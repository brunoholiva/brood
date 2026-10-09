"""Logging configuration for brood entry points."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Route stdlib library logs to stderr at *level*.

    Library modules log through the standard library; entry points call
    this once so those records are visible alongside their own output.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
