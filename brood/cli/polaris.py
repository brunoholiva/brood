#!/usr/bin/env python3
"""Train and locally evaluate a model on an official Polaris benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from brood.core.logging import setup_logging
from brood.models.tabpfn import (
    resolve_device,
    train_and_predict_tabpfn,
    train_and_predict_tabpfn_regression,
)
from brood.polaris import (
    benchmark_id,
    benchmark_inputs,
    evaluate_predictions,
    save_run_artifacts,
)
from brood.representations.registry import get_representation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="data/polaris_results",
        help="Local result directory (default: data/polaris_results)",
    )
    return parser.parse_args()


def _load_benchmark(dataset: str):
    import polaris as po

    return po.load_benchmark(benchmark_id(dataset))


def _features(representation: str, train_smiles, test_smiles, device: str):
    kwargs = (
        {"device": resolve_device(device)}
        if representation == "chemeleon"
        else {}
    )
    featurizer = get_representation(representation, **kwargs)
    return (
        featurizer.transform(train_smiles),
        featurizer.transform(test_smiles),
    )


def run(
    config: dict,
    dataset: str,
    representation: str,
    device: str,
    output_dir: str = "data/polaris_results",
):
    """Run one configured TabPFN representation on one benchmark."""
    spec = config["datasets"][dataset]
    benchmark = _load_benchmark(dataset)
    train_smiles, targets, test_smiles = benchmark_inputs(benchmark)
    x_train, x_test = _features(
        representation, train_smiles, test_smiles, device
    )
    model_kwargs = {"device": resolve_device(device)}
    if spec["task"] == "regression":
        predictions = train_and_predict_tabpfn_regression(
            dataset,
            x_train,
            targets,
            x_test,
            list(test_smiles),
            tabpfn_params=model_kwargs,
        )["predicted_probability"].to_numpy()
    else:
        predictions = train_and_predict_tabpfn(
            dataset,
            x_train,
            targets,
            x_test,
            list(test_smiles),
            tabpfn_params=model_kwargs,
        )["predicted_probability"].to_numpy()
    result = evaluate_predictions(
        benchmark,
        predictions,
        spec["metric"],
        name=f"tabpfn-{representation}",
        description=(f"TabPFN v3.5 + {representation}."),
        tags=["tabpfn", representation],
    )
    path = save_run_artifacts(
        Path(output_dir),
        dataset,
        representation,
        benchmark,
        result,
        test_smiles,
        predictions,
        task=spec["task"],
        metric=spec["metric"],
        n_train=len(train_smiles),
    )
    logger.info(f"Evaluated {benchmark_id(dataset)}: {result}")
    logger.info(f"Local artifacts saved to {path}")


def main() -> None:
    """Load configuration and evaluate one Polaris benchmark locally."""
    setup_logging()
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run(
        config,
        args.dataset,
        args.representation,
        args.device,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
