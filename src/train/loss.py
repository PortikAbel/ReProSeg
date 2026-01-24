from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config.schema.model import LossWeights
from model.model import TrainPhase


@dataclass
class Loss:
    total: torch.Tensor
    alignment: torch.Tensor
    jsd: torch.Tensor
    tanh: torch.Tensor
    classification: torch.Tensor
    
    @classmethod
    def on_device(cls, device: torch.device) -> "Loss":
        """Create a Loss instance with zero tensors on the specified device."""
        return cls(
            total=torch.tensor(0.0, device=device),
            alignment=torch.tensor(0.0, device=device),
            jsd=torch.tensor(0.0, device=device),
            tanh=torch.tensor(0.0, device=device),
            classification=torch.tensor(0.0, device=device),
        )

def calculate_loss(
    aspp_features: torch.Tensor,
    pooled: torch.Tensor,
    out: torch.Tensor,
    ys: torch.Tensor,
    weights: LossWeights,
    train_phase: TrainPhase,
    criterion: torch.nn.Module,
) -> Loss:
    # Data preparation
    af1, af2 = aspp_features.chunk(2)
    pooled1, pooled2 = pooled.chunk(2)

    embv2 = af2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = af1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.0
    jsd_loss = (jensen_shannon_divergence(pooled1) + jensen_shannon_divergence(pooled2)) / 2.0
    tanh_loss = (log_tanh_loss(pooled1) + log_tanh_loss(pooled2)) / 2.0
    class_loss = torch.tensor(0.0, device=aspp_features.device)

    loss = torch.tensor(0.0, device=aspp_features.device)
    if train_phase is not TrainPhase.FINETUNE:
        loss += weights.alignment * a_loss_pf
        loss += weights.jsd * jsd_loss
        loss += weights.tanh * tanh_loss

    if train_phase is not TrainPhase.PRETRAIN:
        softmax_inputs = torch.log1p(out**2)
        class_loss = criterion(softmax_inputs, ys.squeeze())

        loss += weights.classification * class_loss

    # Create loss result structure
    return Loss(
        total=loss,
        alignment=a_loss_pf,
        jsd=jsd_loss,
        tanh=tanh_loss,
        classification=class_loss,
    )


def jensen_shannon_divergence(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 4
    x = x.flatten(start_dim=2)  # Flatten to (batch_size, num_prototypes, height*width)

    w = F.softmax(x.sum(dim=-1, keepdim=True), dim=1)
    m = x.mul(w).sum(dim=1, keepdim=True).expand_as(x)
    m = F.log_softmax(m, dim=-1)

    x = F.log_softmax(x, dim=-1)
    jsd = F.kl_div(x, m, reduction="none", log_target=True).sum(dim=-1).mul(w.squeeze()).sum(dim=-1)

    return torch.exp(-jsd).mean()


def log_tanh_loss(x: torch.Tensor, EPS=1e-10) -> torch.Tensor:
    return -torch.log(torch.tanh(torch.sum(x, dim=(0, 2, 3))) + EPS).mean()


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs: torch.Tensor, targets: torch.Tensor, EPS=1e-12) -> torch.Tensor:
    assert inputs.shape == targets.shape
    assert not targets.requires_grad

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss
