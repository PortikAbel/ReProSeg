"""Deprecated: replaced by dedicated per-scenario entry points.

Use instead:
- `uv run python -m train`     (training)
- `uv run python -m visualize` (prototype visualization)
- `uv run python -m evaluate`  (interpretability metrics, e.g. consistency score)
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "src/scripts/run.py is deprecated and no longer functional.\n"
        "Use `python -m train`, `python -m visualize`, or `python -m evaluate` instead."
    )

