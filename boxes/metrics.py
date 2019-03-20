import torch
from torch import Tensor
from .box_operations import *
import scipy.stats as spstats # For Spearman r


def pearson_r(p: Tensor, q: Tensor) -> Tensor:
    """Pearson r statistic
    Implementation is translated from scipy.stats.pearsonr
    """
    mp = p.mean()
    mq = q.mean()
    pm, qm = p-mp, q-mq
    r_num = torch.sum(pm * qm)
    r_den = torch.sqrt(torch.sum(pm**2) * torch.sum(qm**2))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)

    # The rest is leftover from the SciPy function, but we don't need it
    # n = p.shape[0]
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    # return r, prob
    return r


def spearman_r(p: Tensor, q: Tensor) -> float:
    """Spearman r statistic"""
    # TODO: Make a pytorch tensor version of this
    p = p.cpu().detach().numpy()
    q = q.cpu().detach().numpy()
    sr, _ = spstats.spearmanr(p, q)
    return sr


def metric_pearson_r(model, data_in, data_out):
    return pearson_r(model(data_in)["P(B|A)"], data_out).detach().cpu().item()


def metric_spearman_r(model, data_in, data_out):
    return spearman_r(model(data_in)['P(B|A)'], data_out)

def metric_num_needing_push(model, data_in, data_out):
    return needing_push_mask(model(data_in)['boxes'], data_out).sum().detach().cpu().item()


def metric_num_needing_pull(model, data_in, data_out):
    return needing_pull_mask(model(data_in)['boxes'], data_out).sum().detach().cpu().item()
