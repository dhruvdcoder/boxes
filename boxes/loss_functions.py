import torch
import torch.nn.functional as F

def kl_div(output, target, eps=1e-38):
    return F.kl_div(torch.log(output.clamp_min(eps)), target.clamp_min(eps), reduction="none") + \
           F.kl_div(torch.log((1-output).clamp_min(eps)), (1-target).clamp_min(eps), reduction="none")
