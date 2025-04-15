import torch
import torch.nn.functional as F

from utils.log import Log

def calculate_loss(
    log: Log,
    aspp_features,
    pooled,
    out,
    ys1,
    align_pf_weight,
    t_weight,
    unif_weight,
    var_weigth,
    cl_weight,
    pretrain,
    finetune,
    criterion,
    train_iter,
    iteration=0,
    print=True,
):
    ys = torch.cat([ys1, ys1])
    af1, af2 = aspp_features.chunk(2)
    pooled1, pooled2 = pooled.chunk(2)

    embv2 = af2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = af1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.0
    tanh_loss = (log_tanh_loss(pooled1) + log_tanh_loss(pooled2)) / 2.0
    uni_loss = (uniform_loss(pooled1) + uniform_loss(pooled2)) / 2.0
    var_loss = (variance_loss(embv1) + variance_loss(embv2)) / 2.0

    if not finetune:
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss
        loss += unif_weight * uni_loss
        loss += var_weigth * var_loss

    acc = 0.0
    if not pretrain:
        softmax_inputs = torch.log1p(out**2)
        class_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys.squeeze())

        if finetune:
            loss = cl_weight * class_loss
        else:
            loss += cl_weight * class_loss

        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(torch.prod(torch.tensor(ys.shape)))

    if print:
        with torch.no_grad():
            train_iter.set_postfix_str(
                (
                    f"LA:{a_loss_pf.item():.2f}, " +
                    f"LT:{tanh_loss.item():.3f}, " +
                    f"LU:{uni_loss.item():.3f}, " +
                    f"LV:{var_loss.item():.3f}, " +
                    (f"LC:{class_loss.item():.3f}, " if not pretrain else "") +
                    f"L:{loss.item():.3f}, " +
                    f"num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}"  # noqa +
                    (f", Ac:{acc:.3f}" if not pretrain else "")
                ),
                refresh=False,
            )
            phase_string = "pretrain" if pretrain else "train"
            log.tb_scalar(f"Loss/{phase_string}/LA", a_loss_pf.item(), iteration)
            log.tb_scalar(f"Loss/{phase_string}/LT", tanh_loss.item(), iteration)
            log.tb_scalar(f"Loss/{phase_string}/LU", uni_loss.item(), iteration)
            log.tb_scalar(f"Loss/{phase_string}/LV", var_loss.item(), iteration)
            if not pretrain:
                log.tb_scalar(f"Loss/{phase_string}/LC", class_loss.item(), iteration)
            log.tb_scalar(f"Loss/{phase_string}/L", loss.item(), iteration)

    return loss, acc


def log_tanh_loss(x, EPS=1e-10):
    return -torch.log(torch.tanh(torch.sum(x, dim=0)) + EPS).mean()


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/.
def uniform_loss(x, t=2, EPS=1e-10):
    # print(
    #   "sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape,
    #   torch.sum(torch.pow(x,2), dim=1),
    # ) #--> should be ones
    x = x.permute(1, 0, 2, 3)
    x = x.reshape(x.size(0), -1)
    x = F.normalize(x + EPS, dim=0)
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + EPS).log()
    return loss


def variance_loss(x, gamma=1, EPS=1e-12):
    return (gamma - (x.var(dim=0) + EPS).sqrt()).clamp(min=0).mean()


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert not targets.requires_grad

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss
