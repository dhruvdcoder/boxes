import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *
from .learner import Learner
from dataclasses import dataclass


@dataclass
class LossPieces:
    functions: Collection

    def __post_init__(self):
        # Make the functions into a Dict[String, Callable]
        # Want to be able to pass something quickly like a list of functions,
        # Also want to be able to quickly override the weight without creating a lambda function
        # May also want to specify the name (not sure about that)
        raise NotImplemented

    def loss_func(self, model_out: Tensor, true_out: Tensor, name: str = "default", learner: Learner = None) -> Tensor:
        # Run all functions
        raise NotImplemented



def kl_div(output, target, eps=1e-38):
    return F.kl_div(torch.log(output.clamp_min(eps)), target.clamp_min(eps), reduction="none") + \
           F.kl_div(torch.log((1-output).clamp_min(eps)), (1-target).clamp_min(eps), reduction="none")
