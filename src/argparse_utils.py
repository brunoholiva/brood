"""Argument parsing utilities for training scripts."""

from __future__ import annotations

import argparse


def build_base_arg_parser(
    description: str = "Train model and predict.",
) -> argparse.ArgumentParser:
    """Build the base argument parser for training scripts.

    Returns the parser so callers can add custom arguments
    before calling ``.parse_args()``.

    Parameters
    ----------
    description : str
        Description for the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with dataset, data_dir, random_state, tune, n_trials.
    """
    parser = argparse.ArgumentParser(description=description)
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
        "--n-trials",
        type=int,
        default=30,
        help="Number of tuning trials (default: 30)",
    )
    return parser


def parse_training_args(
    description: str = "Train model and predict.",
) -> argparse.Namespace:
    """Parse standard CLI arguments for training scripts.

    Convenience wrapper around :func:`build_base_arg_parser` for
    scripts that don't need extra arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: dataset, data_dir, random_state, tune, n_trials.
    """
    return build_base_arg_parser(description).parse_args()
