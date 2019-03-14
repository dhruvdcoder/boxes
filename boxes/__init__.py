import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .delta_boxes import *
from .learner import *
from .probability_dataset import *
from .loss_functions import *
from .box_operations import *
import numpy as np
import itertools




def load_numpy_num_lines(path, num = "all", dtype=np.float):
    with open(path) as f:
        if num != "all":
            f = itertools.islice(f, 0, num)
        n = np.loadtxt(f, dtype=dtype)
    return n


def load_from_julia(path, pos_name, neg_name, weight_name=None, num_pos = "all", num_neg = "all", ratio_neg = None):
    pos_ids = load_numpy_num_lines(f"{path}{pos_name}", num_pos, dtype=np.int)
    num_pos = pos_ids.shape[0]
    if ratio_neg is not None:
        num_neg = num_pos * ratio_neg
    neg_ids = load_numpy_num_lines(f"{path}{neg_name}", num_neg, dtype=np.int)
    num_neg = neg_ids.shape[0]
    ids = torch.from_numpy(np.concatenate((pos_ids, neg_ids)))
    probs = np.zeros(ids.shape[0])
    if weight_name is not None:
        probs[0:num_pos] = load_numpy_num_lines(f"{path}{weight_name}", num_pos, dtype=np.float)
    else:
        probs[0:num_pos] = 1
    probs = torch.from_numpy(probs).float()
    return ids, probs

