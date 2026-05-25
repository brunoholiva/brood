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

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from lightning import pytorch as pl
from loguru import logger
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score

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


def _train_model(
    model: MPNN,
    train_loader,
    val_loader,
    epochs: int,
    checkpoint_dir: Path | None = None,
) -> pl.Trainer:
    """Train an MPNN with Lightning and return the trainer."""
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        ),
    ]
    if checkpoint_dir is not None:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer


def _predict(model: MPNN, test_loader) -> np.ndarray:
    """Return predicted probabilities for the test set.

    ``BinaryClassificationFFN.forward`` already applies sigmoid,
    so the model outputs raw probabilities in [0, 1].
    """
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    outputs = trainer.predict(model, dataloaders=test_loader)
    return torch.cat(outputs).squeeze().numpy()


def _load_data(
    train_path: Path, test_path: Path, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate train/test CSVs."""
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run split_datasets.py first.")
        sys.exit(1)
    if not test_path.exists():
        logger.error(f"{test_path} not found. Run split_datasets.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    logger.info(
        f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
    )
    logger.info(f"{dataset}: test={len(test_df)} ({n_test_actives} actives)")

    return train_df, test_df


def _objective(
    trial,
    train_smiles,
    train_targets,
    random_state: int,
) -> float:
    """Optuna objective: val AP after training on a 0.9 split."""
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

    datapoints = _make_datapoints(train_smiles, train_targets)
    n = len(datapoints)
    n_train = int(0.9 * n)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    train_dps = [datapoints[i] for i in indices[:n_train]]
    val_dps = [datapoints[i] for i in indices[n_train:]]

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
    _train_model(model, train_loader, val_loader, epochs=15)

    val_preds = _predict(model, val_loader)
    val_targets = np.array([dp.y[0] for dp in val_dps])
    return float(average_precision_score(val_targets, val_preds))


def _run_tuning(
    train_smiles,
    train_targets,
    dataset: str,
    n_trials: int,
    random_state: int,
) -> dict:
    """Run Optuna hyperparameter search and log results to MLflow."""
    logger.info(f"Tuning with Optuna ({n_trials} trials, 15 epochs each)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        lambda trial: _objective(
            trial, train_smiles, train_targets, random_state
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val_ap = study.best_value

    logger.success(f"Tuning done: best val AP = {best_val_ap:.4f}")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    with mlflow.start_run(run_name=f"tuning/chemprop/{dataset}", nested=False):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_average_precision", best_val_ap)
        mlflow.log_param("n_trials", n_trials)
        mlflow.set_tags(
            {"dataset": dataset, "model": "chemprop", "stage": "tuning"}
        )

    return best_params


def _train_and_predict(
    train_smiles,
    train_targets,
    test_smiles,
    params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """Build model, train on full training data, predict on test set."""
    datapoints = _make_datapoints(train_smiles, train_targets)
    n = len(datapoints)
    n_train = int(0.9 * n)
    indices = np.random.RandomState(42).permutation(n)
    train_dps = [datapoints[i] for i in indices[:n_train]]
    val_dps = [datapoints[i] for i in indices[n_train:]]

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
    _train_model(
        model,
        train_loader,
        val_loader,
        epochs=params.get("epochs", 30),
    )

    probs = _predict(model, test_loader)

    model_path = out_dir / "chemprop.pt"
    save_model(str(model_path), model)

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": probs,
        }
    )


def _save_predictions(
    preds: pd.DataFrame, out_dir: Path, filename: str
) -> Path:
    """Write predictions CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Chemprop MPNN and predict."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. stokes)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state (default: 42)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of tuning trials (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    """Train Chemprop, predict, save predictions."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    train_df, test_df = _load_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
    )

    params = _default_params()
    if args.tune:
        params = _run_tuning(
            train_df["standardized_smiles"],
            train_df["target"],
            dataset,
            args.n_trials,
            args.random_state,
        )

    params["epochs"] = 30

    preds = _train_and_predict(
        train_df["standardized_smiles"],
        train_df["target"],
        test_df["standardized_smiles"],
        params,
        data_dir / "predictions" / dataset,
    )

    _save_predictions(
        preds, data_dir / "predictions" / dataset, "chemprop.csv"
    )


if __name__ == "__main__":
    main()
