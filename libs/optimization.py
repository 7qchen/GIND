import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class LinearScheduler(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, t_total, warmup_steps=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class VariationalHidDropout(nn.Module):
    def __init__(self, dropout=0.0):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        :param temporal: Whether the dropout mask is the same across the temporal dimension (or only the depth dimension)
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def reset_mask(self, z):
        m = torch.zeros_like(z).bernoulli_(1 - self.dropout)

        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, f"You need to reset mask before using {self.__class__.__name__}"
        assert self.mask.size() == x.size()  # Make sure the dimension matches
        return self.mask * x