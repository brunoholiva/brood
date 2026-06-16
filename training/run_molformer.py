#!/usr/bin/env python3
"""Fine-tune MoLFormer with a trainable skip-connection MLP head.

Supports full, head-only, or partial unfreezing of the MoLFormer backbone.
Default mode is partial unfreezing of the last transformer layers.

Optionally tune hyperparameters with Optuna (--tune).

Usage
-----
    python training/run_molformer.py --dataset stokes
    python training/run_molformer.py --dataset stokes --tune --n-trials 50
    python training/run_molformer.py --dataset stokes \
        --finetune-mode partial --unfreeze-last-n-layers 4
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

from src.argparse_utils import build_base_arg_parser
from src.data_utils import load_split_data, save_predictions
from src.lightning_utils import predict_lightning, train_lightning_model
from src.splitter import split_train_test
from src.tuning_utils import run_optuna_tuning

RDLogger.logger().setLevel(RDLogger.ERROR)

MOLFORMER_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_EMBED_DIM = 768
VALID_FINETUNE_MODES = {"partial", "full", "head_only"}


class SkipConnectionMLP(nn.Module):
    """Two-layer MLP with residual connections."""

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
        """Forward pass returning logits."""
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
    """Lightning module for MoLFormer fine-tuning + MLP head."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        dropout: float = 0.2,
        weight_decay: float = 0.01,
        finetune_mode: str = "partial",
        unfreeze_last_n_layers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay

        self.molformer = _build_backbone(MOLFORMER_MODEL_NAME)
        _set_backbone_trainability(
            self.molformer,
            finetune_mode=finetune_mode,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )

        self.head = SkipConnectionMLP(
            input_dim=MOLFORMER_EMBED_DIM,
            dropout=dropout,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.validation_probs: list[torch.Tensor] = []
        self.validation_targets: list[torch.Tensor] = []

        trainable = _count_trainable_parameters(self)
        total = _count_total_parameters(self)
        logger.info(
            f"MolFormer trainable parameters: {trainable:,}/{total:,} "
            f"({100 * trainable / max(total, 1):.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: backbone embeddings -> MLP logits."""
        outputs = self.molformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embeddings = _pool_backbone_output(outputs, attention_mask)
        logits = self.head(embeddings)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Single training step with BCEWithLogits loss."""
        logits, targets = _forward_batch(self, batch)
        loss = self.loss_fn(logits, targets)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Single validation step returning loss, probs, and targets."""
        logits, targets = _forward_batch(self, batch)
        loss = self.loss_fn(logits, targets)
        probs = torch.sigmoid(logits)

        self.validation_probs.append(probs.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )
        return {
            "val_loss": loss,
            "probs": probs.detach().cpu().numpy().flatten(),
            "targets": targets.detach().cpu().numpy().flatten(),
        }

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation AP from collected predictions."""
        if not self.validation_probs:
            return

        all_probs = torch.cat(self.validation_probs).numpy().ravel()
        all_targets = (
            torch.cat(self.validation_targets).numpy().ravel().astype(int)
        )

        self.validation_probs.clear()
        self.validation_targets.clear()

        if len(np.unique(all_targets)) >= 2:
            ap = average_precision_score(all_targets, all_probs)
            self.log("val_ap", ap, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Prediction step returning probabilities."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        return probs.flatten()

    def configure_optimizers(self):
        """AdamW with separate groups for backbone and head."""
        groups = _build_optimizer_param_groups(
            model=self,
            backbone_lr=self.backbone_lr,
            head_lr=self.head_lr,
            weight_decay=self.weight_decay,
        )
        return torch.optim.AdamW(groups)


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
        """Return number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Return (smiles, target) at index."""
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


def _build_backbone(model_name: str) -> nn.Module:
    """Load MoLFormer backbone from HuggingFace."""
    return AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        deterministic_eval=True,
    )


def _pool_backbone_output(
    outputs, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool transformer outputs into one embedding per molecule."""
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    if not hasattr(outputs, "last_hidden_state"):
        raise AttributeError(
            "Backbone output missing both pooler_output and last_hidden_state."
        )

    token_embeddings = outputs.last_hidden_state
    expanded_mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    )
    expanded_mask = expanded_mask.float()
    summed = torch.sum(token_embeddings * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Return the list of encoder layers for common HF structures."""
    candidate_paths = [
        ("encoder", "layer"),
        ("transformer", "layer"),
        ("bert", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
    ]

    for path in candidate_paths:
        current = model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and isinstance(current, (nn.ModuleList, list, tuple)):
            return list(current)

    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return list(module)

    return []


def _unfreeze_module(module: nn.Module | None) -> None:
    """Set requires_grad=True for all module parameters."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = True


def _set_backbone_trainability(
    backbone: nn.Module,
    finetune_mode: str,
    unfreeze_last_n_layers: int,
) -> None:
    """Configure trainable backbone parameters for fine-tuning mode."""
    if finetune_mode not in VALID_FINETUNE_MODES:
        raise ValueError(
            f"Invalid finetune_mode '{finetune_mode}'. "
            f"Expected one of: {sorted(VALID_FINETUNE_MODES)}"
        )

    if finetune_mode == "full":
        for param in backbone.parameters():
            param.requires_grad = True
        return

    if finetune_mode == "head_only":
        for param in backbone.parameters():
            param.requires_grad = False
        return

    for param in backbone.parameters():
        param.requires_grad = False

    layers = _find_transformer_layers(backbone)
    if not layers:
        logger.warning(
            "Could not locate explicit transformer layers. "
            "Falling back to full unfreezing of backbone."
        )
        for param in backbone.parameters():
            param.requires_grad = True
        return

    n_total = len(layers)
    n_unfreeze = max(1, min(unfreeze_last_n_layers, n_total))
    for layer in layers[-n_unfreeze:]:
        _unfreeze_module(layer)

    for optional_name in ("pooler", "ln_f", "final_layer_norm"):
        attr = getattr(backbone, optional_name, None)
        if isinstance(attr, nn.Module):
            _unfreeze_module(attr)

    logger.info(
        f"Partial fine-tuning: unfroze last {n_unfreeze}/{n_total} "
        "backbone layers"
    )


def _split_weight_decay_named_params(
    named_params: list[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split parameters into decay and no-decay groups."""
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


def _build_optimizer_param_groups(
    model: MolFormerClassifier,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Create AdamW groups with separate LRs for backbone and head."""
    backbone_named = list(model.molformer.named_parameters())
    head_named = list(model.head.named_parameters())

    b_decay, b_no_decay = _split_weight_decay_named_params(backbone_named)
    h_decay, h_no_decay = _split_weight_decay_named_params(head_named)

    groups = [
        {
            "params": b_decay,
            "weight_decay": weight_decay,
            "lr": backbone_lr,
        },
        {
            "params": b_no_decay,
            "weight_decay": 0.0,
            "lr": backbone_lr,
        },
        {
            "params": h_decay,
            "weight_decay": weight_decay,
            "lr": head_lr,
        },
        {
            "params": h_no_decay,
            "weight_decay": 0.0,
            "lr": head_lr,
        },
    ]
    return [g for g in groups if len(g["params"]) > 0]


def _forward_batch(
    model: MolFormerClassifier,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward and return logits/targets tensors."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["target"].float().unsqueeze(1)
    logits = model(input_ids, attention_mask)
    return logits, targets


def _count_trainable_parameters(module: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_total_parameters(module: nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in module.parameters())


def _build_model(**kwargs) -> MolFormerClassifier:
    """Build a MolFormerClassifier."""
    return MolFormerClassifier(
        learning_rate=kwargs.get("learning_rate", 1e-4),
        backbone_lr=kwargs.get("backbone_lr", 1e-5),
        head_lr=kwargs.get("head_lr", 1e-4),
        dropout=kwargs.get("dropout", 0.2),
        weight_decay=kwargs.get("weight_decay", 0.01),
        finetune_mode=kwargs.get("finetune_mode", "partial"),
        unfreeze_last_n_layers=kwargs.get("unfreeze_last_n_layers", 4),
    )


def _default_params() -> dict:
    """Return default hyperparameters for fine-tuning."""
    return {
        "learning_rate": 1e-4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "batch_size": 32,
        "epochs": 30,
        "patience": 8,
        "finetune_mode": "partial",
        "unfreeze_last_n_layers": 4,
    }


def _predict(
    model: MolFormerClassifier,
    test_loader: DataLoader,
) -> np.ndarray:
    """Return predicted probabilities for the test set."""
    probs = predict_lightning(model, test_loader)
    return np.atleast_1d(probs)


def _make_loaders(
    train_smiles: list[str],
    train_targets: list[int],
    val_smiles: list[str],
    val_targets: list[int],
    batch_size: int,
    collate_fn,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders."""
    train_dataset = SmilesDataset(train_smiles, train_targets)
    val_dataset = SmilesDataset(val_smiles, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def _objective(
    trial,
    train_df: pd.DataFrame,
    tokenizer,
    base_params: dict,
) -> float:
    """Optuna objective: mean AP from 5-fold Butina GroupKFold CV."""
    finetune_mode = base_params["finetune_mode"]

    params = {
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-1, log=True
        ),
        "backbone_lr": trial.suggest_float(
            "backbone_lr",
            1e-6,
            5e-5,
            log=True,
        ),
        "head_lr": trial.suggest_float(
            "head_lr",
            1e-5,
            1e-3,
            log=True,
        ),
        "finetune_mode": finetune_mode,
        "unfreeze_last_n_layers": base_params["unfreeze_last_n_layers"],
    }

    if finetune_mode == "partial":
        params["unfreeze_last_n_layers"] = trial.suggest_categorical(
            "unfreeze_last_n_layers",
            [2, 4, 6, 8],
        )

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

        logger.info(
            "  Fold {}/5  |  train={}, val={}".format(
                fold + 1, len(train_idx), len(val_idx)
            )
        )

        train_loader, val_loader = _make_loaders(
            train_smiles_fold,
            train_targets_fold,
            val_smiles_fold,
            val_targets_fold,
            32,
            collate_fn,
        )

        model = _build_model(**params)
        train_lightning_model(
            model,
            train_loader,
            val_loader,
            epochs=12,
            patience=5,
            restore_best_weights=True,
            monitor="val_ap",
            monitor_mode="max",
        )

        val_preds = _predict(model, val_loader)
        val_targets_arr = np.array(val_targets_fold)

        if len(set(val_targets_arr)) < 2:
            logger.warning(
                f"Fold {fold}: only one class in validation, skipping"
            )
            continue

        ap = average_precision_score(val_targets_arr, val_preds)
        fold_scores.append(ap)

    if not fold_scores:
        return 0.0

    mean_ap = float(np.mean(fold_scores))
    return mean_ap


def _train_and_predict(
    dataset: str,
    train_df: pd.DataFrame,
    test_smiles: list[str],
    params: dict,
    out_dir: Path,
    tokenizer,
) -> pd.DataFrame:
    """Train model and return test-set predictions."""
    inner_train_df, inner_val_df = split_train_test(train_df, train_size=0.9)

    n_train = len(inner_train_df)
    n_val = len(inner_val_df)
    n_train_clusters = inner_train_df["cluster_id"].nunique()
    n_val_clusters = inner_val_df["cluster_id"].nunique()

    logger.info(
        "{}: inner split — train={} ({} clust), val={} ({} clust)".format(
            dataset,
            n_train,
            n_train_clusters,
            n_val,
            n_val_clusters,
        )
    )

    if n_val == 0:
        logger.warning(
            "Validation set is empty after OOD split! "
            "Falling back to using training set for validation."
        )
        inner_train_df, inner_val_df = train_df, train_df

    collate_fn = _create_collate_fn(tokenizer)

    train_loader, val_loader = _make_loaders(
        inner_train_df["standardized_smiles"].tolist(),
        inner_train_df["target"].tolist(),
        inner_val_df["standardized_smiles"].tolist(),
        inner_val_df["target"].tolist(),
        params["batch_size"],
        collate_fn,
    )

    test_dataset = SmilesDataset(test_smiles, [0] * len(test_smiles))
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = _build_model(**params)
    _, best_metric = train_lightning_model(
        model,
        train_loader,
        val_loader,
        epochs=params.get("epochs", 30),
        patience=params.get("patience", 8),
        restore_best_weights=True,
        monitor="val_ap",
        monitor_mode="max",
        enable_progress_bar=True,
    )

    probs = _predict(model, test_loader)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), out_dir / "molformer_head.pt")
    torch.save(model.state_dict(), out_dir / "molformer_finetuned.pt")

    metric_name = "val_AP"
    if best_metric is not None:
        logger.success(
            "MoLFormer model saved to {} (best {}={:.4f})".format(
                out_dir / "molformer_finetuned.pt", metric_name, best_metric
            )
        )
    else:
        logger.success(
            f"MoLFormer model saved to {out_dir / 'molformer_finetuned.pt'}"
        )

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": probs,
        }
    )


def _parse_args():
    """Parse CLI arguments for MoLFormer training."""
    parser = build_base_arg_parser(
        description="Fine-tune MoLFormer with trainable MLP head."
    )
    parser.add_argument(
        "--finetune-mode",
        default="partial",
        choices=sorted(VALID_FINETUNE_MODES),
        help="Backbone fine-tuning mode (default: partial)",
    )
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=4,
        help="Layers to unfreeze for partial mode (default: 4)",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Backbone learning rate override",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Head learning rate override",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Head dropout override",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs override",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience override",
    )
    return parser.parse_args()


def _apply_cli_overrides(params: dict, args) -> dict:
    """Apply optional CLI overrides to parameter dictionary."""
    overrides = {
        "finetune_mode": args.finetune_mode,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    params["learning_rate"] = params["head_lr"]
    return params


def main() -> None:
    """Fine-tune MoLFormer, predict, and save predictions."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    logger.info(f"Loading MoLFormer tokenizer from {MOLFORMER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME,
        trust_remote_code=True,
    )

    train_df, test_df = load_split_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
        require_cluster_id=True,
    )

    params = _default_params()
    params = _apply_cli_overrides(params, args)

    if args.tune:
        n_clusters = train_df["cluster_id"].nunique()
        best_params = run_optuna_tuning(
            lambda trial: _objective(trial, train_df, tokenizer, params),
            dataset,
            model_name="molformer",
            n_trials=args.n_trials,
            random_state=args.random_state,
            n_clusters=n_clusters,
        )
        params = {**params, **best_params}
        params["learning_rate"] = params["head_lr"]

    preds = _train_and_predict(
        dataset,
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
