import torch


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("sigmoid"))