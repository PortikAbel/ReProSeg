"""Interpretability-metric evaluation for trained segmentation models.

Generic over any SupportedModel (ReProSeg, PPNet, ...); see consistency.py.
"""

from . import consistency  # noqa: F401  (registers the "consistency" metric)
