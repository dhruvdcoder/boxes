from .box_operations import *
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *


ModelOutput = Dict[str, Tensor]


def mean_unit_cube_loss(model_out: ModelOutput, _) -> Tensor:
    return ((model_out["box_embeddings"] - 1).clamp(0) + (-model_out["box_embeddings"]).clamp(0)).sum(dim=[-2, -1]).mean()


def mean_unary_kl_loss(unary: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Callable:
    """ Factory Function to create the actual mean_unary_kl_loss function with the given set of unary probabilities. """
    def mean_unary_kl_loss(model_out: ModelOutput, _) -> Tensor:
        return kl_div_sym(model_out["unary_probs"], unary, eps).mean()
    return mean_unary_kl_loss


def mean_cond_kl_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_sym(model_out["P(A|B)"], target, eps).mean()


def kl_div_sym(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return kl_div_term(p, q, eps) + kl_div_term(1-p, 1-q, eps)


def kl_div_term(p: Tensor, q: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    return F.kl_div(torch.log(p.clamp_min(eps)), q.clamp_min(eps), reduction="none")


def mean_pull_loss(model_out: ModelOutput, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Pulls together boxes which are disjoint but should overlap.
    """
    A, B = model_out["A"], model_out["B"]
    _needing_pull_mask = needing_pull_mask(A, B, target)
    num_needing_pull_mask = _needing_pull_mask.sum()
    if num_needing_pull_mask == 0:
        return 0
    else:
        penalty = ((A[:,:,0] - B[:,:,1] + eps).clamp(0) + (B[:,:,0] - A[:,:,1] + eps).clamp(0)).sum(dim=-1)
        return penalty[_needing_pull_mask].sum() / num_needing_pull_mask


def mean_push_loss(model_out: ModelOutput, target: Tensor, eps: float = 1e-6) -> Tensor:
    A, B = model_out["A"], model_out["B"]
    _needing_push_mask = needing_push_mask(A, B, target)
    num_needing_push_mask = _needing_push_mask.sum()
    if num_needing_push_mask == 0:
        return 0
    else:
        penalty = torch.min((A[:,:,1] - B[:,:,0] + eps).clamp(0).min(dim=-1)[0], (B[:,:,1] - A[:,:,0] + eps).clamp(0).min(dim=-1)[0])
        return penalty[_needing_push_mask].sum() / num_needing_push_mask


def mean_surrogate_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    """
    Note: surrogate loss is only used with clamp_volume, so it is hardcoded here.
    """
    A, B = model_out["A"], model_out["B"]
    A_join_B = join(A, B)
    _needing_pull_mask = needing_pull_mask(A, B, target)
    num_needing_pull_mask = _needing_pull_mask.sum()
    if num_needing_pull_mask == 0:
        return 0
    else:
        surrogate = clamp_volume(A_join_B) - clamp_volume(A) - clamp_volume(B)
        penalty = torch.log(surrogate.clamp_min(eps)) - torch.log(clamp_volume(A).clamp_min(eps))
        return (target[_needing_pull_mask[0]] * penalty[_needing_pull_mask]).sum() / num_needing_pull_mask
