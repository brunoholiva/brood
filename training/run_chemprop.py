#!/usr/bin/env python3
"""Train a Chemprop v2 MPNN and save predictions.

Pipeline: AtomMessagePassing → MeanAggregation → BinaryClassificationFFN.

Optionally tune hyperparameters with Optuna (--tune).

Usage
-----
    python training/run_chemprop.py --dataset stokes
    python training/run_chemprop.py --dataset stokes --tune --n-trials 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
)
from chemprop.models import MPNN, save_model
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.message_passing import AtomMessagePassing
from chemprop.nn.metrics import BinaryAUPRC
from chemprop.nn.predictors import BinaryClassificationFFN

from src.splitter import split_by_clusters
from src.training_utils import (
    load_split_data,
    parse_training_args,
    predict_lightning,
    run_optuna_tuning,
    save_predictions,
    train_lightning_model,
)

RDLogger.logger().setLevel(RDLogger.ERROR)


def _make_datapoints(smiles_list, targets_list) -> list[MoleculeDatapoint]:
    """Build a list of MoleculeDatapoints from SMILES and targets."""
    datapoints = []
    for smi, tgt in zip(smiles_list, targets_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES: {smi}")
            continue
        datapoints.append(
            MoleculeDatapoint(mol=mol, y=np.array([tgt], dtype=np.float32))
        )
    return datapoints


def _build_model(**kwargs) -> MPNN:
    """Build an MPNN with message passing, aggregation, and predictor."""
    mp = AtomMessagePassing(
        d_h=kwargs.get("message_hidden_dim", 300),
        depth=kwargs.get("depth", 3),
        dropout=kwargs.get("dropout", 0.0),
    )
    agg = MeanAggregation()
    predictor = BinaryClassificationFFN(
        input_dim=mp.output_dim,
        hidden_dim=kwargs.get("ffn_hidden_dim", 300),
        n_layers=kwargs.get("ffn_num_layers", 1),
        dropout=kwargs.get("dropout", 0.0),
    )
    return MPNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        metrics=[BinaryAUPRC()],
        max_lr=kwargs.get("max_lr", 1e-3),
        warmup_epochs=kwargs.get("warmup_epochs", 2),
    )


def _default_params() -> dict:
    return {
        "depth": 3,
        "message_hidden_dim": 300,
        "dropout": 0.0,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "max_lr": 1e-3,
        "batch_size": 64,
        "epochs": 30,
    }


def _objective(
    trial,
    train_df: pd.DataFrame,
) -> float:
    """Optuna objective: mean AP from 5-fold Butina GroupKFold CV.

    Uses GroupKFold with cluster_id as groups to ensure no structural
    leakage between train and validation folds.
    """
    params = {
        "depth": trial.suggest_int("depth", 2, 6),
        "message_hidden_dim": trial.suggest_categorical(
            "message_hidden_dim", [128, 256, 512]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "ffn_hidden_dim": trial.suggest_categorical(
            "ffn_hidden_dim", [128, 256, 512]
        ),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "max_lr": trial.suggest_float("max_lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    all_smiles = train_df["standardized_smiles"].tolist()
    all_targets = train_df["target"].tolist()
    all_clusters = train_df["cluster_id"].tolist()

    gkf = GroupKFold(n_splits=5)
    X = np.arange(len(train_df))
    y = np.array(all_targets)
    groups = np.array(all_clusters)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_smiles_fold = [all_smiles[i] for i in train_idx]
        train_targets_fold = [all_targets[i] for i in train_idx]
        val_smiles_fold = [all_smiles[i] for i in val_idx]
        val_targets_fold = [all_targets[i] for i in val_idx]

        val_clusters = set([all_clusters[i] for i in val_idx])

        logger.debug(
            f"  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}, "
            f"val_clusters={sorted(val_clusters)}"
        )

        train_dps = _make_datapoints(train_smiles_fold, train_targets_fold)
        val_dps = _make_datapoints(val_smiles_fold, val_targets_fold)

        train_dataset = MoleculeDataset(train_dps)
        val_dataset = MoleculeDataset(val_dps)

        train_loader = build_dataloader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        model = _build_model(**params)
        train_lightning_model(model, train_loader, val_loader, epochs=15)

        val_preds = predict_lightning(model, val_loader)
        val_targets_arr = np.array([dp.y[0] for dp in val_dps])

        if len(set(val_targets_arr)) < 2:
            logger.warning(
                f"Fold {fold}: only one class in validation, skipping"
            )
            continue

        ap = average_precision_score(val_targets_arr, val_preds)
        fold_scores.append(ap)
        logger.debug(f"  Fold {fold}: AP = {ap:.4f}")

    if not fold_scores:
        return 0.0

    mean_ap = float(np.mean(fold_scores))
    logger.debug(f"Mean CV AP: {mean_ap:.4f}")
    return mean_ap


def _train_and_predict(
    train_df: pd.DataFrame,
    test_smiles: list[str],
    params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """Build model, train on full training data, predict on test set.

    Uses Butina OOD split (smallest clusters as validation) for early stopping.
    """
    inner_train_df, inner_val_df = split_by_clusters(train_df, train_size=0.9)

    n_train = len(inner_train_df)
    n_val = len(inner_val_df)
    train_clusters = sorted(set(inner_train_df["cluster_id"]))
    val_clusters = sorted(set(inner_val_df["cluster_id"]))

    logger.info(
        f"OOD inner split: train={n_train} (clusters {train_clusters}), "
        f"val={n_val} (clusters {val_clusters})"
    )

    if n_val == 0:
        logger.warning(
            "Validation set is empty after OOD split! "
            "Falling back to using training set for validation."
        )
        inner_train_df, inner_val_df = train_df, train_df

    train_dps = _make_datapoints(
        inner_train_df["standardized_smiles"].tolist(),
        inner_train_df["target"].tolist(),
    )
    val_dps = _make_datapoints(
        inner_val_df["standardized_smiles"].tolist(),
        inner_val_df["target"].tolist(),
    )

    train_dataset = MoleculeDataset(train_dps)
    val_dataset = MoleculeDataset(val_dps)
    test_dataset = MoleculeDataset(
        _make_datapoints(test_smiles, [0] * len(test_smiles))
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = _build_model(**params)
    train_lightning_model(
        model,
        train_loader,
        val_loader,
        epochs=params.get("epochs", 30),
        restore_best_weights=True,
    )

    probs = predict_lightning(model, test_loader)

    model_path = out_dir / "chemprop.pt"
    save_model(str(model_path), model)

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": probs,
        }
    )


def main() -> None:
    """Train Chemprop, predict, save predictions."""
    args = parse_training_args(description="Train Chemprop MPNN and predict.")
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    train_df, test_df = load_split_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
        require_cluster_id=True,
    )

    params = _default_params()
    if args.tune:
        n_clusters = train_df["cluster_id"].nunique()
        params = run_optuna_tuning(
            lambda trial: _objective(trial, train_df),
            dataset,
            model_name="chemprop",
            n_trials=args.n_trials,
            random_state=args.random_state,
            n_clusters=n_clusters,
        )

    params["epochs"] = 30

    preds = _train_and_predict(
        train_df,
        test_df["standardized_smiles"].tolist(),
        params,
        data_dir / "predictions" / dataset,
    )

    save_predictions(preds, data_dir / "predictions" / dataset, "chemprop.csv")


if __name__ == "__main__":
    main()
