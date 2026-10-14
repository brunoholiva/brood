#!/usr/bin/env python3
"""Unified training entry point for all brood models.

The model and representation are resolved through the central registries in
:mod:`brood.models` and :mod:`brood.representations`.

Usage
-----
    brood-train --dataset GNEtolC --model rf \
        --representation morgan2048 --tune
    brood-train --dataset GNEtolC --model rf \
        --representation morgan2048 --cv
    brood-train --dataset GNEtolC --model chemprop --cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from brood.core.artifacts import prediction_dir
from brood.core.logging import setup_logging
from brood.cv.loop import load_params, run_cv_model
from brood.data.loader import save_predictions
from brood.models import get_model, is_sklearn_model
from brood.models.registry import list_models
from brood.representations import list_representations

_TABPFN_MODELS = ("tabpfn", "tabpfn_regression")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one model and write predictions."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
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
        "--tune-only",
        action="store_true",
        help="Tune and save params without fitting or predicting",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run repeated scaffold CV (5x5) into the model run directory. "
        "Tunable models re-tune on each CV training portion.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of tuning trials per Optuna study (default: 30). "
        "Under --cv, one study runs per CV training portion.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Model to train",
    )
    parser.add_argument(
        "--representation",
        default=None,
        choices=list_representations(),
        help="Molecular representation (required for feature-based models)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for TabPFN (auto/cpu/cuda)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, backend) -> None:
    if backend.needs_features and args.representation is None:
        sys.exit(
            f"{args.model} requires --representation. "
            f"Available: {list_representations()}"
        )
    if not backend.needs_features and args.representation is not None:
        logger.warning(
            f"{args.model} ignores --representation; ignoring "
            f"{args.representation}"
        )
    if args.tune and not backend.supports_tuning:
        sys.exit(
            f"{args.model} does not support --tune. "
            "Remove --tune and run again."
        )
    if args.tune and args.cv:
        sys.exit(
            "--tune and --cv are separate steps: tuning writes params.json "
            "and scoring reuses it. Run without --cv first, then with --cv."
        )


def main() -> None:
    """Tune, cross-validate, or predict depending on the flags."""
    setup_logging()
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    backend_kwargs = {}
    if args.model in _TABPFN_MODELS:
        backend_kwargs["device"] = args.device

    backend = get_model(args.model, **backend_kwargs)
    _validate_args(args, backend)

    representation = args.representation
    run_name = backend.run_name(representation)

    if args.tune:
        backend.tune(
            dataset=dataset,
            data_dir=data_dir,
            representation=representation,
            n_trials=args.n_trials,
            random_state=args.random_state,
        )
        if args.tune_only:
            return

    params = load_params(data_dir, dataset, run_name, backend.default_params())
    if args.model in _TABPFN_MODELS:
        params["random_state"] = args.random_state

    if args.cv:

        def _predict_fold(train_df, test_df):
            fold_params = params
            if backend.supports_tuning:
                fold_params = backend.tune_on(
                    dataset=dataset,
                    data_dir=data_dir,
                    train_df=train_df,
                    representation=representation,
                    n_trials=args.n_trials,
                    random_state=args.random_state,
                )
            return backend.predict_fold(
                dataset=dataset,
                data_dir=data_dir,
                train_df=train_df,
                test_df=test_df,
                representation=representation,
                params=fold_params,
            )

        run_cv_model(dataset, data_dir, run_name, _predict_fold)
        return

    preds = backend.predict_held_out(
        dataset=dataset,
        data_dir=data_dir,
        representation=representation,
        params=params,
    )
    if is_sklearn_model(args.model):
        # Sklearn backend returns the dataframe but does not save it;
        # non-sklearn backends save inside predict_held_out for historical
        # reasons. Align behaviour here.
        save_predictions(
            preds,
            prediction_dir(data_dir, dataset, run_name),
            "test.csv",
        )


if __name__ == "__main__":
    main()
