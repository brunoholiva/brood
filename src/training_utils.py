"""Shared utilities for model training scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupKFold


def parse_training_args(
    description: str = "Train model and predict.",
) -> argparse.Namespace:
    """Parse standard CLI arguments for training scripts.

    Parameters
    ----------
    description : str
        Description for the argument parser.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: dataset, data_dir, random_state, tune, n_trials.
    """
    parser = argparse.ArgumentParser(description=description)
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


def load_split_data(
    train_path: Path,
    test_path: Path,
    dataset: str,
    require_cluster_id: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate train/test CSVs from split files.

    Parameters
    ----------
    train_path : Path
        Path to training CSV.
    test_path : Path
        Path to test CSV.
    dataset : str
        Dataset name for logging.
    require_cluster_id : bool
        If True, error if ``cluster_id`` column is missing. If False, warn
        and continue. Default: True.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with loaded data.
    """
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

    if "cluster_id" in train_df.columns:
        n_clusters = train_df["cluster_id"].nunique()
        logger.info(
            f"{dataset}: train={len(train_df)} ({n_train_actives} actives, "
            f"{n_clusters} Butina clusters)"
        )
    else:
        if require_cluster_id:
            logger.error(
                f"{train_path} missing 'cluster_id' column. Re-run "
                "scripts/split_datasets.py to regenerate splits."
            )
            sys.exit(1)
        else:
            logger.info(
                f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
            )
    logger.info(f"{dataset}: test={len(test_df)} ({n_test_actives} actives)")

    return train_df, test_df


def save_predictions(
    preds: pd.DataFrame, out_dir: Path, filename: str
) -> Path:
    """Write predictions CSV and return the path.

    Parameters
    ----------
    preds : pd.DataFrame
        DataFrame with predictions (standardized_smiles,
        predicted_probability columns).
    out_dir : Path
        Output directory (created if needed).
    filename : str
        Output filename (e.g. "rf.csv").

    Returns
    -------
    Path
        Full path to the saved CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")
    return out_path


def run_optuna_tuning(
    objective: Callable[[optuna.Trial], float],
    dataset: str,
    model_name: str,
    n_trials: int,
    random_state: int,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search and log results to MLflow.

    Creates a study with TPESampler, runs optimization, logs best params
    and metrics to MLflow, and returns the best parameters.

    Parameters
    ----------
    objective : Callable[[optuna.Trial], float]
        Optuna objective function (maximized).
    dataset : str
        Dataset name for logging.
    model_name : str
        Model name for MLflow tags (e.g. "rf", "chemprop", "molformer").
    n_trials : int
        Number of Optuna trials.
    random_state : int
        Random seed for TPESampler.
    n_clusters : int | None
        Number of Butina clusters (for logging). If provided, logs
        "cv_strategy" as "Butina GroupKFold".

    Returns
    -------
    dict[str, Any]
        Best hyperparameters from Optuna.
    """
    if n_clusters is not None:
        logger.info(
            f"Tuning with Optuna ({n_trials} trials, "
            f"5-fold Butina GroupKFold, {n_clusters} clusters)..."
        )
    else:
        logger.info(f"Tuning with Optuna ({n_trials} trials, 5-fold CV)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_ap = study.best_value

    logger.success(f"Tuning done: best CV AP = {best_cv_ap:.4f}")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    with mlflow.start_run(
        run_name=f"tuning/{model_name}/{dataset}", nested=False
    ):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_average_precision", best_cv_ap)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", 5)
        if n_clusters is not None:
            mlflow.log_param("cv_strategy", "Butina GroupKFold")
        mlflow.set_tags(
            {"dataset": dataset, "model": model_name, "stage": "tuning"}
        )

    return best_params


def train_lightning_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 30,
    restore_best_weights: bool = True,
):
    """Train a LightningModule with EarlyStopping and best-weight restoration.

    Parameters
    ----------
    model : pl.LightningModule
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int
        Maximum number of epochs. Default: 30.
    restore_best_weights : bool
        If True, restore model weights from best val_loss epoch.
        If False, keep weights from last epoch. Default: True.
    """
    import tempfile

    import lightning.pytorch as pl
    import torch

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]

    temp_ckpt_dir = None
    if restore_best_weights:
        temp_ckpt_dir = tempfile.mkdtemp()
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(temp_ckpt_dir),
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
        enable_checkpointing=restore_best_weights,
    )
    trainer.fit(model, train_loader, val_loader)

    if restore_best_weights and temp_ckpt_dir is not None:
        checkpoint_callback = trainer.checkpoint_callback
        if (
            checkpoint_callback is not None
            and checkpoint_callback.best_model_path
        ):
            best_path = checkpoint_callback.best_model_path
            logger.debug(f"Restoring best weights from: {best_path}")
            ckpt = torch.load(best_path, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])

        import shutil

        shutil.rmtree(temp_ckpt_dir)

    return trainer


def predict_lightning(model, test_loader) -> np.ndarray:
    """Return predicted probabilities using a Lightning Trainer.

    Parameters
    ----------
    model : pl.LightningModule
        Trained model.
    test_loader : DataLoader
        Test data loader.

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    import lightning.pytorch as pl
    import torch

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    outputs = trainer.predict(model, dataloaders=test_loader)
    all_probs = []
    for out in outputs:
        if isinstance(out, np.ndarray):
            all_probs.append(torch.tensor(out))
        else:
            all_probs.append(out)
    return torch.cat(all_probs).squeeze().numpy()


def butina_group_kfold(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate GroupKFold splits using ``cluster_id`` as groups.

    Ensures no structural leakage between train/val folds by keeping
    entire Butina clusters together.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``cluster_id`` column.
    n_splits : int
        Number of CV folds. Default: 5.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_idx, val_idx) arrays for each fold.

    Raises
    ------
    KeyError
        If ``cluster_id`` column is missing from df.
    """
    if "cluster_id" not in df.columns:
        raise KeyError("DataFrame missing 'cluster_id' column for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(len(df))
    y = df["target"].to_numpy()
    groups = df["cluster_id"].to_numpy()

    return list(gkf.split(X, y, groups))
