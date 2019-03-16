import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *
from .learner import Learner, Recorder
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import pandas as pd


def func_list_to_dict(func_list) -> Dict[str, Callable]:
    """

    :param func_list: List of functions or tuples (weight, function)
    :type func_list: Union[Collection[Union[Callable, Tuple[float, Callable]]], Dict[str, Callable]]

    :return: Ordered Dictionary of {name: function}
    :rtype: OrderedDict[str, Callable]
    """
    if type(func_list) == OrderedDict or type(func_list) == dict:
        return func_list
    func_dict = OrderedDict()
    for f in func_list:
        if type(f) is tuple:
            func_dict[f"{f[0]}*{f[1].__name__}"] = lambda *x: f[0] * f[1](*x)
            func_dict[f[0]] = f[1]
        else:
            func_dict[f.__name__] = f
    return func_dict


class LossPieces:

    def __init__(self, recorder: Recorder, *loss_funcs):
        """

        :param functions: List of functions or tuples (weight, function)
        :type functions: Collection[Union[Callable, Tuple[float, Callable]]]
        """
        self.recorder = recorder
        self.loss_funcs = func_list_to_dict(loss_funcs)
        self.recorder.data["LossPieces"] = defaultdict(pd.DataFrame)
        self.r = self.recorder.data["LossPieces"]

    def loss_func(self, model_out: Tensor, true_out: Tensor, name: str = "default", learner: Learner = None) -> Tensor:
        """
        Weighted sum of all loss functions. Tracks values in learner.

        """
        grad_status = torch.is_grad_enabled()
        if learner is None or learner.status != "train":
            torch.set_grad_enabled(False)
        try:
            loss_pieces = {k: l(model_out, true_out) for k, l in self.loss_funcs.items()}
            loss = sum(loss_pieces.values())
            loss_pieces['loss'] = loss
            loss_pieces = {k: [t.detach().cpu().item()] for k, t in loss_pieces.items()}
            self.r[name] = self.r[name].append(
                pd.DataFrame(loss_pieces, [learner.progress.partial_epoch_progress()]), sort=False)
        finally:
            torch.set_grad_enabled(grad_status)
        return loss


def mean_unary_kl_loss(unary, eps=1e-38):
    def mean_unary_kl_loss(model_out, _):
        return kl_div_sym(model_out["unary_vol"], unary, eps).mean()
    return mean_unary_kl_loss


def mean_cond_kl_loss(model_out, target, eps=1e-38):
    return kl_div_sym(model_out["P(B|A)"], target, eps).mean()


def kl_div_sym(p, q, eps=1e-38):
    p = p.clamp(1e-7, 1-1e-7)
    return kl_div_term(p,q, eps) + kl_div_term(1-p, 1-q, eps)

# return F.kl_div(torch.log(p.clamp_min(eps)), q.clamp_min(eps), reduction="none") + \
#        F.kl_div(torch.log((1 - p).clamp_min(eps)), (1 - q).clamp_min(eps), reduction="none")


def kl_div_term(p, q, eps=1e-38):
    return F.kl_div(torch.log(p.clamp_min(eps)), q.clamp_min(eps), reduction="none")
