#!/usr/bin/env python3
"""Fine-tune MoLFormer with a trainable MLP head (frozen backbone).

Pipeline: MoLFormer (frozen) -> pooler_output -> Skip-connection MLP
-> binary classification.

Optionally tune hyperparameters with Optuna (--tune).

Usage
-----
    python training/run_molformer.py --dataset stokes
    python training/run_molformer.py --dataset stokes --tune --n-trials 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from loguru import logger
from rdkit import RDLogger
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

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

MOLFORMER_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_EMBED_DIM = 768


class SkipConnectionMLP(nn.Module):
    """2-layer MLP with skip connections.

    From original MoLFormer finetune code.

    Architecture:
        x -> Linear -> Dropout -> GELU -> (+ residual x)
          -> Linear -> Dropout -> GELU -> (+ residual)
          -> Linear -> sigmoid (for inference)

    Uses BCEWithLogitsLoss during training, so outputs logits.
    """

    def __init__(self, input_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu2 = nn.GELU()
        self.final = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (apply sigmoid for probabilities)."""
        x_out = self.fc1(x)
        x_out = self.dropout1(x_out)
        x_out = self.gelu1(x_out)
        x_out = x_out + x

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.gelu2(z)
        z = self.final(z + x_out)

        return z


class MolFormerClassifier(pl.LightningModule):
    """PyTorch Lightning module for frozen MoLFormer + trainable MLP head.

    MoLFormer backbone is frozen (requires_grad=False) during training.
    Only the SkipConnectionMLP head is trained.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.molformer = AutoModel.from_pretrained(
            MOLFORMER_MODEL_NAME,
            trust_remote_code=True,
            deterministic_eval=True,
        )

        for param in self.molformer.parameters():
            param.requires_grad = False

        self.head = SkipConnectionMLP(
            input_dim=MOLFORMER_EMBED_DIM, dropout=dropout
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: MoLFormer pooler output -> MLP head -> logits."""
        self.molformer.eval()
        with torch.no_grad():
            outputs = self.molformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = outputs.pooler_output

        logits = self.head(embeddings)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["target"].float().unsqueeze(1)

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, targets)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Run one validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["target"].float().unsqueeze(1)

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, targets)
        probs = torch.sigmoid(logits)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )

        return {
            "val_loss": loss,
            "probs": probs.detach().cpu().numpy().flatten(),
            "targets": targets.detach().cpu().numpy().flatten(),
        }

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Return predicted probabilities (not logits)."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        return probs.flatten()

    def configure_optimizers(self):
        """Configure the optimizer (AdamW)."""
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


class SmilesDataset(Dataset):
    """Simple Dataset for SMILES strings and binary targets."""

    def __init__(
        self,
        smiles_list: list[str],
        targets_list: list[int] | None = None,
    ):
        self.smiles = smiles_list
        self.targets = (
            targets_list
            if targets_list is not None
            else [0] * len(smiles_list)
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Return the idx-th sample (smiles, target)."""
        return self.smiles[idx], self.targets[idx]


def _create_collate_fn(tokenizer):
    """Create a collate function that tokenizes SMILES."""

    def collate_fn(batch: list[tuple[str, int]]) -> dict:
        smiles_list = [smi for smi, _ in batch]
        targets = [tgt for _, tgt in batch]

        tokens = tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "target": torch.tensor(targets, dtype=torch.long),
        }

    return collate_fn


def _build_model(**kwargs) -> MolFormerClassifier:
    """Build a MolFormerClassifier with frozen backbone + MLP head."""
    return MolFormerClassifier(
        learning_rate=kwargs.get("learning_rate", 1e-3),
        dropout=kwargs.get("dropout", 0.2),
        weight_decay=kwargs.get("weight_decay", 0.0),
    )


def _default_params() -> dict:
    """Return default hyperparameters."""
    return {
        "learning_rate": 1e-3,
        "dropout": 0.2,
        "weight_decay": 0.0,
        "batch_size": 64,
        "epochs": 50,
    }


def _predict(
    model: MolFormerClassifier, test_loader: DataLoader
) -> np.ndarray:
    """Return predicted probabilities for the test set."""
    return predict_lightning(model, test_loader)


def _objective(
    trial,
    train_df: pd.DataFrame,
    tokenizer,
) -> float:
    """Optuna objective: mean AP from 5-fold Butina GroupKFold CV.

    Uses GroupKFold with cluster_id as groups to ensure no structural
    leakage between train and validation folds.
    """
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
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
    collate_fn = _create_collate_fn(tokenizer)

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

        train_dataset = SmilesDataset(train_smiles_fold, train_targets_fold)
        val_dataset = SmilesDataset(val_smiles_fold, val_targets_fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        model = _build_model(**params)
        train_lightning_model(model, train_loader, val_loader, epochs=15)

        val_preds = _predict(model, val_loader)
        val_targets_arr = np.array(val_targets_fold)

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
    tokenizer,
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

    train_dataset = SmilesDataset(
        inner_train_df["standardized_smiles"].tolist(),
        inner_train_df["target"].tolist(),
    )
    val_dataset = SmilesDataset(
        inner_val_df["standardized_smiles"].tolist(),
        inner_val_df["target"].tolist(),
    )
    test_dataset = SmilesDataset(test_smiles, [0] * len(test_smiles))

    collate_fn = _create_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = _build_model(**params)
    train_lightning_model(
        model,
        train_loader,
        val_loader,
        epochs=params.get("epochs", 50),
    )

    probs = _predict(model, test_loader)

    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "molformer_head.pt"
    torch.save(model.head.state_dict(), head_path)
    logger.success(f"MLP head weights saved to {head_path}")

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": probs,
        }
    )


def main() -> None:
    """Fine-tune MoLFormer, predict, save predictions."""
    args = parse_training_args(
        description="Fine-tune MoLFormer with trainable MLP head."
    )
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    logger.info(f"Loading MoLFormer tokenizer from {MOLFORMER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME, trust_remote_code=True
    )

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
            lambda trial: _objective(trial, train_df, tokenizer),
            dataset,
            model_name="molformer",
            n_trials=args.n_trials,
            random_state=args.random_state,
            n_clusters=n_clusters,
        )

    params["epochs"] = 50

    preds = _train_and_predict(
        train_df,
        test_df["standardized_smiles"].tolist(),
        params,
        data_dir / "predictions" / dataset,
        tokenizer,
    )

    save_predictions(
        preds, data_dir / "predictions" / dataset, "molformer.csv"
    )


if __name__ == "__main__":
    main()
