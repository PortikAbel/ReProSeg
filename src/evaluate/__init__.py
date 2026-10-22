"""Interpretability-metric evaluation for trained segmentation models.

Generic over any SupportedModel (ReProSeg, PPNet, ...); see consistency.py.
"""

from . import consistency, stability

__all__ = ["consistency", "stability"]
