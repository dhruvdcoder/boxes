from .box_operations import *
import torch
from torch import Tensor
import scipy.stats as spstats # For Spearman r


def pearson_r(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    """Pearson r statistic
    Implementation is translated from scipy.stats.pearsonr
    """
    mp = p.mean()
    mq = q.mean()
    pm, qm = p-mp, q-mq
    r_num = torch.sum(pm * qm)
    r_den = torch.sqrt(torch.sum(pm**2) * torch.sum(qm**2))
    r = r_num / (r_den + 1e-38)

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
    return pearson_r(model(data_in)["P(B|A)"], data_out)


def metric_spearman_r(model, data_in, data_out):
    return spearman_r(model(data_in)["P(B|A)"], data_out)


def metric_num_needing_push(model, data_in, data_out):
    model_out = model(data_in)
    return needing_push_mask(model_out["A"], model_out["B"], data_out).sum()


def metric_num_needing_pull(model, data_in, data_out):
    model_out = model(data_in)
    return needing_pull_mask(model_out["A"], model_out["B"], data_out).sum()


def metric_hard_accuracy(model, data_in, data_out):
    hard_pred = model(data_in)["P(B|A)"] > 0.5
    return (data_out == hard_pred.float()).float().mean()


def metric_hard_f1(model, data_in, data_out):
    hard_pred = model(data_in)["P(B|A)"] > 0.5
    true_pos = data_out[hard_pred==1].sum()
    total_pred_pos = (hard_pred==1).sum().float()
    total_actual_pos = data_out.sum().float()
    precision = true_pos / total_pred_pos
    recall = true_pos / total_actual_pos
    return 2 * (precision*recall) / (precision + recall)
