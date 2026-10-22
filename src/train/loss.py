from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config.schema.model import LossWeights
from model.model import TrainPhase


@dataclass
class Loss:
    total: torch.Tensor
    jsd: torch.Tensor
    tanh: torch.Tensor
    classification: torch.Tensor

    @classmethod
    def on_device(cls, device: torch.device) -> "Loss":
        """Create a Loss instance with zero tensors on the specified device."""
        return cls(
            total=torch.tensor(0.0, device=device),
            jsd=torch.tensor(0.0, device=device),
            tanh=torch.tensor(0.0, device=device),
            classification=torch.tensor(0.0, device=device),
        )


def calculate_loss(
    pooled: torch.Tensor,
    out: torch.Tensor,
    ys: torch.Tensor,
    weights: LossWeights,
    train_phase: TrainPhase,
    criterion: torch.nn.Module,
) -> Loss:
    jsd_loss = jensen_shannon_divergence(pooled)
    tanh_loss = log_tanh_loss(pooled)

    loss = torch.tensor(0.0, device=pooled.device)
    if train_phase is not TrainPhase.FINETUNE:
        loss += weights.jsd * jsd_loss
        loss += weights.tanh * tanh_loss

    scores = torch.log1p(out**2)
    class_loss = criterion(scores, ys.squeeze())
    loss += weights.classification * class_loss

    # Create loss result structure
    return Loss(
        total=loss,
        jsd=jsd_loss,
        tanh=tanh_loss,
        classification=class_loss,
    )


def jensen_shannon_divergence(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 4
    # Flatten to (batch_size, num_concepts, height*width)
    x = x.flatten(start_dim=2)
    x_log = F.log_softmax(x, dim=-1)

    # Compute log of the mean concept distribution: log(mean_i p_i)
    num_concepts = x_log.size(1)
    m_log = torch.logsumexp(x_log, dim=1, keepdim=True) - torch.log(
        torch.tensor(float(num_concepts), device=x_log.device, dtype=x_log.dtype)
    )
    m_log = m_log.expand_as(x_log)

    # JSD over spatial distributions: average concept-wise KLs
    jsd = F.kl_div(x_log, m_log, reduction="none", log_target=True).sum(dim=-1).mean(dim=-1)

    return torch.exp(-jsd).mean()


def log_tanh_loss(x: torch.Tensor, EPS=1e-10) -> torch.Tensor:
    return -torch.log(torch.tanh(torch.sum(torch.amax(x, dim=(2, 3)), dim=0)) + EPS).mean()
