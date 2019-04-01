from __future__ import annotations
from .box_operations import *
from learner import Callback
from dataclasses import dataclass, field
import torch
from typing import *

if TYPE_CHECKING:
    from learner import Learner, Recorder


@dataclass
class MinBoxSize(Callback):
    """Prevents boxes from getting too small during training.
    """
    min_vol: float = 1e-6
    recorder: Union[None, Recorder] = None
    name: str = "Small Boxes"
    eps: float = 1e-6 # added to the size of the boxes due to floating point precision

    def learner_post_init(self, learner: Learner):
        if self.recorder is None:
            self.recorder = learner.recorder

    def batch_end(self, l: Learner):
        with torch.no_grad():
            boxes = l.model.box_embedding.boxes
            small_boxes = detect_small_boxes(boxes, l.model.vol_func, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_({self.name + f" (<{self.min_vol} before MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_({self.name + f" (<{self.min_vol - self.eps} before MinBoxSize)": detect_small_boxes(boxes, l.model.vol_func, self.min_vol - self.eps).sum()}, l.progress.partial_epoch_progress())
            if num_min_boxes > 0:
                replace_Z_by_cube_(boxes, small_boxes, self.min_vol + self.eps)
            small_boxes = detect_small_boxes(boxes, l.model.vol_func, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_({self.name + f" (<{self.min_vol} after MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_({self.name + f" (<{self.min_vol - self.eps} after MinBoxSize)": detect_small_boxes(boxes, l.model.vol_func, self.min_vol - self.eps).sum()}, l.progress.partial_epoch_progress())
