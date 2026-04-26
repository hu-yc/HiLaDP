import torch
import torch.nn as nn


def mlp(dims_list, batchNorm=False, dropout_ratio=0.0, nonlinearity='leaky_relu', negative_slope=0.1):
    layers = []
    pre_dim = dims_list[0]
    hidden_dims = dims_list[1:]
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(pre_dim, hidden_dim))
        if batchNorm:
             layers.append(nn.BatchNorm1d(hidden_dim, track_running_stats=True))
        if nonlinearity == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinearity == 'leaky_relu':
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        if dropout_ratio:
            layers.append(nn.Dropout(dropout_ratio))
        pre_dim = hidden_dim
    return nn.Sequential(*layers)


def gumbel_softmax(logits, tau=2 / 3, hard=False, dim=-1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)
