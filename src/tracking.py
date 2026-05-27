"""MLflow experiment logger for brood."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from typing import Optional

import mlflow

from .types import BroodResult, EvalConfig, SplitConfig


def log_experiment(
    run_name: str,
    result: BroodResult,
    split_cfg: Optional[SplitConfig] = None,
    eval_cfg: Optional[EvalConfig] = None,
    tags: Optional[dict] = None,
    extra_params: Optional[dict] = None,
) -> None:
    """Log a brood screening experiment to MLflow.

    Parameters
    ----------
    run_name : str
        Name for the MLflow run.
    result : BroodResult
        The screening result to log.
    split_cfg : SplitConfig, optional
        Split configuration logged as parameters.
    eval_cfg : EvalConfig, optional
        Evaluation configuration logged as parameters.
    tags : dict, optional
        Tags set on the MLflow run.
    extra_params : dict, optional
        Additional parameters to log (e.g. training set stats).
    """
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)

        if split_cfg is not None:
            mlflow.log_params(
                {f"split_{k}": v for k, v in asdict(split_cfg).items()}
            )

        if eval_cfg is not None:
            params = {}
            if eval_cfg.bedroc_alpha is not None:
                params["eval_bedroc_alpha"] = eval_cfg.bedroc_alpha
            if eval_cfg.distance_bins is not None:
                for i, (lo, hi, name) in enumerate(eval_cfg.distance_bins):
                    params[f"eval_bin_{i}_name"] = name
                    params[f"eval_bin_{i}_low"] = lo
                    params[f"eval_bin_{i}_high"] = hi
            mlflow.log_params(params)

        if extra_params:
            mlflow.log_params(extra_params)

        mlflow.log_param("global_n", result.global_.n)
        mlflow.log_param("n_test", result.global_.n)
        mlflow.log_metric(
            "global_average_precision", result.global_.average_precision
        )
        mlflow.log_metric("global_bedroc", result.global_.bedroc)

        for b in result.by_bin:
            mlflow.log_param(f"bin_{b.bin}_n", b.n)
            mlflow.log_param(f"bin_{b.bin}_n_actives", b.n_actives)
            mlflow.log_metric(
                f"bin_{b.bin}_average_precision", b.average_precision
            )
            mlflow.log_metric(f"bin_{b.bin}_bedroc", b.bedroc)

        tmp_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name
        try:
            result.test_df.to_csv(tmp_path, index=False)
            mlflow.log_artifact(tmp_path, artifact_path="test_df")
        finally:
            os.unlink(tmp_path)
