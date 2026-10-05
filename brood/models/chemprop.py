"""Chemprop v2 MPNN backend.

Chemprop is the molecular-graph benchmark: it consumes SMILES directly, not an
interchangeable fixed representation, so it never joins the representation
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from brood.core.artifacts import prediction_dir, run_dir, save_metadata
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data, save_predictions
from brood.data.splitter import split_train_test
from brood.models.base import ModelBackend
from brood.models.lightning import predict_lightning, train_lightning_model
from brood.tuning.optuna import run_optuna_tuning

RDLogger.logger().setLevel(RDLogger.ERROR)

BATCH_SIZE = 256
NUM_WORKERS = 3
TUNE_EPOCHS = 15
TUNE_PATIENCE = 5
INNER_SPLITS = 5


def _import_chemprop():
    from chemprop.data import (
        MoleculeDatapoint,
        MoleculeDataset,
        build_dataloader,
    )
    from chemprop.models import MPNN, save_model
    from chemprop.nn.agg import NormAggregation
    from chemprop.nn.message_passing import BondMessagePassing
    from chemprop.nn.metrics import ArrayLike, BinaryAUPRC, ChempropMetric
    from chemprop.nn.predictors import BinaryClassificationFFN

    return {
        "MoleculeDatapoint": MoleculeDatapoint,
        "MoleculeDataset": MoleculeDataset,
        "build_dataloader": build_dataloader,
        "MPNN": MPNN,
        "save_model": save_model,
        "NormAggregation": NormAggregation,
        "BondMessagePassing": BondMessagePassing,
        "ArrayLike": ArrayLike,
        "BinaryAUPRC": BinaryAUPRC,
        "ChempropMetric": ChempropMetric,
        "BinaryClassificationFFN": BinaryClassificationFFN,
    }


def _import_torch():
    import torch

    return torch


def _make_weighted_bce(pos_weight: float, task_weights: Any = 1.0):
    """Return a Chemprop metric subclass with per-class weighting."""
    cp = _import_chemprop()

    class _WeightedBCELoss(cp["ChempropMetric"]):
        def __init__(self, pos_weight=pos_weight, task_weights=task_weights):
            super().__init__(task_weights)
            self.pos_weight = pos_weight

        def _calc_unreduced_loss(self, preds, targets, *args):
            torch = _import_torch()
            pw = torch.tensor(
                [self.pos_weight], device=preds.device, dtype=preds.dtype
            )
            return torch.nn.functional.binary_cross_entropy_with_logits(
                preds, targets, pos_weight=pw, reduction="none"
            )

    return _WeightedBCELoss()


def _default_params() -> dict:
    return {
        "depth": 3,
        "message_hidden_dim": 300,
        "dropout": 0.0,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "max_lr": 1e-3,
        "batch_size": BATCH_SIZE,
        "epochs": 30,
        "patience": 10,
    }


def _build_model(pos_weight: float | None = None, **kwargs):
    cp = _import_chemprop()
    mp = cp["BondMessagePassing"](
        d_h=kwargs.get("message_hidden_dim", 300),
        depth=kwargs.get("depth", 3),
        dropout=kwargs.get("dropout", 0.0),
        undirected=True,
    )
    agg = cp["NormAggregation"]()
    pos_weight = kwargs.get("pos_weight")
    criterion = (
        _make_weighted_bce(pos_weight) if pos_weight is not None else None
    )
    predictor = cp["BinaryClassificationFFN"](
        input_dim=mp.output_dim,
        hidden_dim=kwargs.get("ffn_hidden_dim", 300),
        n_layers=kwargs.get("ffn_num_layers", 1),
        dropout=kwargs.get("dropout", 0.0),
        criterion=criterion,
    )
    return cp["MPNN"](
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        metrics=[cp["BinaryAUPRC"]()],
        max_lr=kwargs.get("max_lr", 1e-3),
        warmup_epochs=kwargs.get("warmup_epochs", 2),
    )


def _make_datapoints(smiles_list, targets_list):
    cp = _import_chemprop()
    datapoints = []
    for smi, tgt in zip(smiles_list, targets_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES: {smi}")
            continue
        datapoints.append(
            cp["MoleculeDatapoint"](
                mol=mol, y=np.array([tgt], dtype=np.float32)
            )
        )
    return datapoints


def _make_loader(dataset, *, shuffle: bool, batch_size: int = BATCH_SIZE):
    cp = _import_chemprop()
    return cp["build_dataloader"](
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )


@dataclass
class _Fold:
    train_loader: object
    val_loader: object
    pos_weight: float
    val_targets: np.ndarray


def _build_folds(train_df: pd.DataFrame) -> list[_Fold]:
    cp = _import_chemprop()
    all_smiles = train_df[SMILES_COL].tolist()
    all_targets = train_df[TARGET_COL].tolist()
    groups = np.array(train_df["scaffold"].tolist())
    y = np.array(all_targets)

    n_splits = max(
        2,
        min(
            INNER_SPLITS,
            len(set(groups.tolist())),
            int(np.bincount(y).min()),
        ),
    )
    gkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )
    folds = []
    for train_idx, val_idx in gkf.split(np.arange(len(train_df)), y, groups):
        train_targets = [all_targets[i] for i in train_idx]
        train_dps = _make_datapoints(
            [all_smiles[i] for i in train_idx], train_targets
        )
        val_dps = _make_datapoints(
            [all_smiles[i] for i in val_idx],
            [all_targets[i] for i in val_idx],
        )
        n_pos = sum(train_targets)
        n_neg = len(train_targets) - n_pos
        folds.append(
            _Fold(
                train_loader=_make_loader(
                    cp["MoleculeDataset"](train_dps), shuffle=True
                ),
                val_loader=_make_loader(
                    cp["MoleculeDataset"](val_dps), shuffle=False
                ),
                pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
                val_targets=np.array([dp.y[0] for dp in val_dps]),
            )
        )
    return folds


def _objective(trial, folds: list[_Fold]) -> float:
    params = {
        "depth": trial.suggest_int("depth", 2, 6),
        "message_hidden_dim": trial.suggest_categorical(
            "message_hidden_dim", [64, 128, 256, 512]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "ffn_hidden_dim": trial.suggest_categorical(
            "ffn_hidden_dim", [64, 128, 256, 512]
        ),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "max_lr": trial.suggest_float("max_lr", 1e-4, 1e-2, log=True),
    }

    fold_scores = []
    for index, fold in enumerate(folds):
        if len(set(fold.val_targets)) < 2:
            logger.warning(
                f"Fold {index}: only one class in validation, skipping"
            )
            continue
        model = _build_model(pos_weight=fold.pos_weight, **params)
        train_lightning_model(
            model,
            fold.train_loader,
            fold.val_loader,
            epochs=TUNE_EPOCHS,
            patience=TUNE_PATIENCE,
            monitor="val/prc",
            monitor_mode="max",
        )
        val_preds = predict_lightning(model, fold.val_loader)
        fold_scores.append(
            average_precision_score(fold.val_targets, val_preds)
        )

    if not fold_scores:
        return 0.0
    return float(np.mean(fold_scores))


def _train_and_predict(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    cp = _import_chemprop()
    inner_train_df, inner_val_df = split_train_test(train_df, train_size=0.9)

    n_val = len(inner_val_df)
    if n_val == 0:
        logger.warning(
            "Validation set empty after OOD split; using training set."
        )
        inner_train_df, inner_val_df = train_df, train_df

    test_smiles = test_df[SMILES_COL].tolist()
    train_dps = _make_datapoints(
        inner_train_df[SMILES_COL].tolist(),
        inner_train_df[TARGET_COL].tolist(),
    )
    val_dps = _make_datapoints(
        inner_val_df[SMILES_COL].tolist(),
        inner_val_df[TARGET_COL].tolist(),
    )
    test_dps = _make_datapoints(test_smiles, [0] * len(test_smiles))

    train_dataset = cp["MoleculeDataset"](train_dps)
    val_dataset = cp["MoleculeDataset"](val_dps)
    test_dataset = cp["MoleculeDataset"](test_dps)

    train_loader = _make_loader(
        train_dataset, shuffle=True, batch_size=params["batch_size"]
    )
    val_loader = _make_loader(
        val_dataset, shuffle=False, batch_size=params["batch_size"]
    )
    test_loader = _make_loader(
        test_dataset, shuffle=False, batch_size=params["batch_size"]
    )

    train_targets_arr = inner_train_df[TARGET_COL].values
    n_pos = int(train_targets_arr.sum())
    n_neg = len(train_targets_arr) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = _build_model(pos_weight=pos_weight, **params)
    _, best_metric = train_lightning_model(
        model,
        train_loader,
        val_loader,
        epochs=params.get("epochs", 30),
        patience=params.get("patience", 10),
        restore_best_weights=True,
        enable_progress_bar=True,
        monitor="val/prc",
        monitor_mode="max",
    )

    probs = predict_lightning(model, test_loader)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.pt"
        cp["save_model"](str(model_path), model)
        if best_metric is not None:
            logger.success(
                f"Chemprop model saved to {model_path} "
                f"(best val_loss={best_metric:.4f})"
            )
        else:
            logger.success(f"Chemprop model saved to {model_path}")

    return pd.DataFrame({SMILES_COL: test_smiles, PREDICTION_COL: probs})


class ChempropBackend(ModelBackend):
    """Backend for Chemprop v2 MPNN."""

    name = "chemprop"
    supports_tuning = True
    needs_features = False

    def default_params(self) -> dict[str, Any]:
        """Return Chemprop architecture and training defaults."""
        return _default_params()

    def _load_params(self, data_dir: Path, dataset: str) -> dict[str, Any]:
        import json

        path = run_dir(data_dir, dataset, self.name) / "params.json"
        params = self.default_params()
        if path.exists():
            params.update(json.loads(path.read_text()))
            logger.info(f"Using tuned params from {path}")
        return params

    def tune(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Search Chemprop hyperparameters with inner scaffold CV."""
        train_df, _ = load_split_data(
            data_dir / "splits" / f"{dataset}_train.csv",
            data_dir / "splits" / f"{dataset}_test.csv",
            dataset,
            require_scaffold=True,
        )
        tuned = run_optuna_tuning(
            lambda trial: _objective(trial, _build_folds(train_df)),
            dataset,
            model_name=self.name,
            n_trials=n_trials,
            random_state=random_state,
            params_path=run_dir(data_dir, dataset, self.name) / "params.json",
        )
        return {**self.default_params(), **tuned}

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Train Chemprop on the development set and predict the test set."""
        train_df, test_df = load_split_data(
            data_dir / "splits" / f"{dataset}_train.csv",
            data_dir / "splits" / f"{dataset}_test.csv",
            dataset,
            require_scaffold=True,
        )
        preds = _train_and_predict(
            dataset,
            train_df,
            test_df,
            params,
            run_dir(data_dir, dataset, self.name),
        )
        save_predictions(
            preds,
            prediction_dir(data_dir, dataset, self.name),
            "test.csv",
        )
        save_metadata(
            run_dir(data_dir, dataset, self.name),
            "metadata",
            {"model": self.name, "artifact": "model.pt", "params": params},
        )
        return preds

    def predict_fold(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Train Chemprop on one CV fold and predict its test slice."""
        return _train_and_predict(
            dataset, train_df, test_df, params, out_dir=None
        )
