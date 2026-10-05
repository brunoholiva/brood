#!/usr/bin/env python3
"""Batch split all datasets in ``data/raw/`` into train/test.

Thin wrapper around :mod:`brood.cli.split`.
"""

from brood.cli.split import main

if __name__ == "__main__":
    main()
