#!/usr/bin/env python3
"""Unified training entry point for all brood models.

The model and representation are resolved through the central registries in
:mod:`brood.models` and :mod:`brood.representations`.

Usage
-----
    python scripts/train.py --dataset GNEtolC --model rf \
        --representation morgan2048 --tune
    python scripts/train.py --dataset GNEtolC --model rf \
        --representation morgan2048 --cv
    python scripts/train.py --dataset GNEtolC --model tabpfn \
        --representation morgan2048 --cv
    python scripts/train.py --dataset GNEtolC --model chemprop --cv
    python scripts/train.py --dataset GNEtolC --model similarity --cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from brood.core.argparse import build_base_arg_parser
from brood.core.artifacts import prediction_dir, run_dir
from brood.cv.loop import run_cv_model
from brood.data.loader import save_predictions
from brood.models import get_model, is_sklearn_model
from brood.models.registry import list_models
from brood.representations import list_representations


def _parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser(
        description="Train one model and write predictions."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Model family to train",
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
    parser.add_argument(
        "--tabpfn-version",
        default=None,
        help="TabPFN checkpoint version (default: v3.5)",
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
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    backend_kwargs = {}
    if args.model == "tabpfn":
        backend_kwargs["device"] = args.device
        if args.tabpfn_version:
            backend_kwargs["version"] = args.tabpfn_version

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

    params = backend.default_params()
    params_path = run_dir(data_dir, dataset, run_name) / "params.json"
    if params_path.exists() and backend.supports_tuning:
        import json

        params.update(json.loads(params_path.read_text()))
        logger.info(f"Using tuned params from {params_path}")

    if args.cv:

        def _predict_fold(train_df, test_df):
            return backend.predict_fold(
                dataset=dataset,
                data_dir=data_dir,
                train_df=train_df,
                test_df=test_df,
                representation=representation,
                params=params,
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
