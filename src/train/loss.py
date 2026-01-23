import torch
import torch.nn.functional as F
import tqdm

from config.schema.model import LossWeights
from model.model import TrainPhase
from utils.log import Log


def calculate_loss(
    log: Log,
    aspp_features: torch.Tensor,
    pooled: torch.Tensor,
    out: torch.Tensor,
    ys: torch.Tensor,
    weights: LossWeights,
    train_phase: TrainPhase,
    criterion: torch.nn.Module,
    train_iter: tqdm.tqdm,
    iteration=0,
    print=True,
) -> torch.Tensor:
    af1, af2 = aspp_features.chunk(2)
    pooled1, pooled2 = pooled.chunk(2)

    embv2 = af2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = af1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.0
    jsd_loss = (jensen_shannon_divergence(pooled1) + jensen_shannon_divergence(pooled2)) / 2.0
    tanh_loss = (log_tanh_loss(pooled1) + log_tanh_loss(pooled2)) / 2.0
    class_loss = torch.tensor(0.0)

    loss = torch.tensor(0.0, device=aspp_features.device)
    if train_phase is not TrainPhase.FINETUNE:
        loss += weights.alignment * a_loss_pf
        loss += weights.jsd * jsd_loss
        loss += weights.tanh * tanh_loss

    if train_phase is not TrainPhase.PRETRAIN:
        softmax_inputs = torch.log1p(out**2)
        class_loss = criterion(softmax_inputs, ys.squeeze())

        loss += weights.classification * class_loss

    if print:
        with torch.no_grad():
            train_iter.set_postfix_str(
                (
                    f"LA:{a_loss_pf.item():.2f}, "
                    + f"LJ:{jsd_loss.item():.3f}, "
                    + f"LT:{tanh_loss.item():.3f}, "
                    + f"LC:{class_loss.item():.3f}, "
                    + f"L:{loss.item():.3f}, "
                    + f"num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}"
                ),
                refresh=False,
            )
            phase_string = "pretrain" if train_phase == TrainPhase.PRETRAIN else "train"
            log.tb_scalar(f"loss-{phase_string}/LA", a_loss_pf.item(), iteration)
            log.tb_scalar(f"loss-{phase_string}/LJ", jsd_loss.item(), iteration)
            log.tb_scalar(f"loss-{phase_string}/LT", tanh_loss.item(), iteration)
            log.tb_scalar(f"loss-{phase_string}/LC", class_loss.item(), iteration)
            log.tb_scalar(f"loss-{phase_string}/L", loss.item(), iteration)

    return loss


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
