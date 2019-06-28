import torch
import math

_log1mexp_switch = math.log(0.5)


def log1mexp(x: torch.Tensor, split_point=_log1mexp_switch) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).

    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0

    For details, see

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    """
    logexpm1 = torch.log(-torch.expm1(x))
    logp1exp = torch.log1p(-torch.exp(x))
    Z = torch.empty_like(logexpm1)
    logexpm1_switch = x > split_point
    Z[logexpm1_switch] = logexpm1[logexpm1_switch]
    Z[1 - logexpm1_switch] = logp1exp[1 - logexpm1_switch]
    return Z
