"""
Utility functions for segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.image import add_margins_to_image as add_margins_to_image
from model.utils.model import get_params as get_params


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales is not None:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, height, width = logits.shape

        def interpolate(logit):
            return F.interpolate(logit, size=(height, width), mode="bilinear", align_corners=False)

        if len(self.scales) == 0:
            return logits

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interpolate(logit) for logit in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max
