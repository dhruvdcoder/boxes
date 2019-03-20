import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *
from .learner import Learner, Recorder
from collections import OrderedDict
from .box_operations import *


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

    def __init__(self, *loss_funcs):
        """

        :param functions: List of functions or tuples (weight, function)
        :type functions: Collection[Union[Callable, Tuple[float, Callable]]]
        """
        self.loss_funcs = func_list_to_dict(loss_funcs)

    def loss_func(self, model_out: Tensor, true_out: Tensor, learner: Learner = None, recorder: Recorder = None) -> Tensor:
        """
        Weighted sum of all loss functions. Tracks values in Recorder.

        """
        grad_status = torch.is_grad_enabled()
        if learner is None:
            torch.set_grad_enabled(False)
        try:
            loss_pieces = {k: l(model_out, true_out) for k, l in self.loss_funcs.items()}
            # Note: don't want to detach / move / unwrap tensors here because we have to sum the loss first:
            loss = sum(loss_pieces.values())
            loss_pieces['loss'] = loss
            loss_pieces = {k: t.detach().cpu().item() for k, t in loss_pieces.items()}
            if learner is not None:
                if recorder is not None:
                    recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
                else:
                    self.recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
        finally:
            torch.set_grad_enabled(grad_status)
        return loss


def mean_unit_cube_loss(model_out, _):
    return ((model_out['all_boxes'] - 1).clamp(0) + (-model_out['all_boxes']).clamp(0)).sum(dim=[-2, -1]).mean()


def mean_unary_kl_loss(unary, eps=1e-38):
    def mean_unary_kl_loss(model_out, _):
        return kl_div_sym(model_out["unary_vol"], unary, eps).mean()
    return mean_unary_kl_loss


def mean_cond_kl_loss(model_out, target, eps=1e-38):
    return kl_div_sym(model_out["P(B|A)"], target, eps).mean()


def kl_div_sym(p, q, eps=1e-38):
    return kl_div_term(p,q, eps) + kl_div_term(1-p, 1-q, eps)


def kl_div_term(p, q, eps=1e-38):
    return F.kl_div(torch.log(p.clamp_min(eps)), q.clamp_min(eps), reduction="none")


def mean_pull_loss(model_out, target, eps=1e-6):
    """
    Pulls together boxes which are disjoint but should overlap.
    """
    boxes = model_out['boxes']
    A = boxes[:,:,0]
    B = boxes[:,:,1]
    penalty = ((A[:,:,0] - B[:,:,1] + eps).clamp(0) + (B[:,:,0] - A[:,:,1] + eps).clamp(0)).sum(dim=-1)
    _needing_pull_mask = needing_pull_mask(boxes, target)
    return penalty[_needing_pull_mask].sum() / _needing_pull_mask.sum()


def mean_push_loss(model_out, target, eps=1e-6):
    boxes = model_out['boxes']
    A = boxes[:,:,0]
    B = boxes[:,:,1]
    penalty = ((A[:,:,1] - B[:,:,1] + eps).clamp(0) * (A[:,:,0] - B[:,:,0]).clamp(0)).prod(dim=-1)
    _needing_push_mask = needing_push_mask(boxes, target)
    return penalty[_needing_push_mask].sum() / _needing_push_mask.sum()
