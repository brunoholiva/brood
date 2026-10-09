"""Small adapter for running models against Polaris benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def benchmark_id(dataset: str) -> str:
    """Return the Polaris Hub ID for a local benchmark config key."""
    return f"tdcommons/{dataset.replace('_', '-')}"


def prediction_arguments(
    metric: str, predictions: np.ndarray
) -> dict[str, Any]:
    """Return the Polaris prediction argument required by ``metric``."""
    if metric in {"roc_auc", "pr_auc"}:
        return {"y_prob": predictions}
    return {"y_pred": predictions}


def benchmark_inputs(benchmark):
    """Load the documented train inputs/targets and test inputs."""
    train, test = benchmark.get_train_test_split()
    return train.inputs, train.targets, test.inputs


def evaluate_predictions(
    benchmark,
    predictions: np.ndarray,
    metric: str,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
):
    """Evaluate predictions with Polaris without submitting the result."""
    results = benchmark.evaluate(
        **prediction_arguments(metric, np.asarray(predictions))
    )
    if name is not None:
        results.name = name
    if description is not None:
        results.description = description
    if tags is not None:
        results.tags = tags
    return results


def save_run_artifacts(
    output_dir: Path,
    dataset: str,
    representation: str,
    benchmark,
    result,
    test_smiles,
    predictions: np.ndarray,
    *,
    task: str,
    metric: str,
    n_train: int,
) -> Path:
    """Save local metrics and unlabeled test predictions for inspection."""
    run_dir = output_dir / dataset / f"tabpfn_{representation}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "standardized_smiles": list(test_smiles),
            "prediction": np.asarray(predictions),
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    result.results.to_csv(run_dir / "metrics.csv", index=False)
    metadata = {
        "benchmark": benchmark_id(dataset),
        "dataset": dataset,
        "representation": representation,
        "model": "tabpfn_v3.5",
        "task": task,
        "metric": metric,
        "n_train": n_train,
        "n_test": len(test_smiles),
        "result": result.model_dump(mode="json"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir
