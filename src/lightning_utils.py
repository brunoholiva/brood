"""Lightning training and prediction utilities."""

from __future__ import annotations

import numpy as np
from loguru import logger


def _suppress_lightning_noise() -> None:
    """Suppress Lightning tips, deprecations, and low-value warnings."""
    import logging
    import warnings

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="lightning"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning.pytorch.trainer.connectors.data_connector",
    )
    warnings.filterwarnings(
        "ignore", message="Found .* module\\(s\\) in eval mode"
    )
    warnings.filterwarnings("ignore", message="Tensor Cores")
    warnings.filterwarnings("ignore", message="LOCAL_RANK")
    warnings.filterwarnings(
        "ignore", message="You are using a CUDA device with"
    )
    warnings.filterwarnings("ignore", message="Asking to truncate")
    warnings.filterwarnings("ignore", message="No maximum length")
    warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")


def train_lightning_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 30,
    patience: int = 10,
    restore_best_weights: bool = True,
    monitor: str = "val_loss",
    monitor_mode: str = "min",
    enable_progress_bar: bool = False,
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
    patience : int
        Early stopping patience. Default: 10.
    restore_best_weights : bool
        If True, restore model weights from best monitored epoch.
    monitor : str
        Metric to monitor for early stopping / checkpoint. Default: "val_loss".
    monitor_mode : str
        "min" or "max". Default: "min".
    enable_progress_bar : bool
        Show Lightning tqdm epoch progress bar. Default: False.
    """
    import tempfile

    import lightning.pytorch as pl
    import torch

    _suppress_lightning_noise()

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=monitor_mode,
        ),
    ]

    temp_ckpt_dir = None
    if restore_best_weights:
        temp_ckpt_dir = tempfile.mkdtemp()
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(temp_ckpt_dir),
                monitor=monitor,
                mode=monitor_mode,
                save_top_k=1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=restore_best_weights,
    )
    trainer.fit(model, train_loader, val_loader)

    best_metric = None
    if restore_best_weights:
        checkpoint_callback = trainer.checkpoint_callback
        if (
            checkpoint_callback is not None
            and checkpoint_callback.best_model_path
        ):
            best_path = checkpoint_callback.best_model_path
            if hasattr(checkpoint_callback, "best_model_score"):
                best_metric = float(checkpoint_callback.best_model_score)
            logger.debug(f"Restoring best weights from: {best_path}")
            ckpt = torch.load(best_path, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])

        if temp_ckpt_dir is not None:
            import shutil

            shutil.rmtree(temp_ckpt_dir)

    return trainer, best_metric


def predict_lightning(model, test_loader) -> np.ndarray:
    """Return predicted probabilities using a Lightning Trainer."""
    import lightning.pytorch as pl
    import torch

    _suppress_lightning_noise()

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
