#!/usr/bin/env python3
"""Drive one brood pipeline stage from a YAML config.

Thin wrapper around :mod:`brood.pipeline.runner`.
"""

from brood.pipeline.runner import main

if __name__ == "__main__":
    main()
