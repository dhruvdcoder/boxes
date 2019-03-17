from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from .box_operations import *
if TYPE_CHECKING:
    from .learner import Learner, Recorder


class Callback:
    def learner_post_init(self, learner: Learner):
        pass

    def epoch_begin(self, learner: Learner):
        pass

    def batch_begin(self, learner: Learner):
        pass

    def backward_end(self, learner: Learner):
        pass

    def batch_end(self, learner: Learner):
        pass

    def epoch_end(self, learner: Learner):
        pass


class CallbackCollection:

    def __init__(self, *callbacks: Collection[Callback]):
        self._callbacks = callbacks

    def __call__(self, action: str, *args, **kwargs):
        for c in self._callbacks:
            getattr(c, action)(*args, **kwargs)

    def __getattr__(self, action: str):
        return lambda *args, **kwargs: self.__call__(action, *args, **kwargs)



@dataclass
class GradientClipping(Callback):
    min: float = None
    max: float = None

    def backward_end(self, learner: Learner):
        for param in learner.model.parameters():
            param.grad = param.grad.clamp(self.min, self.max)


@dataclass
class MinBoxSize(Callback):
    """Prevents boxes from getting too small during training.
    """
    min_vol: float = 1e-6
    recorder: Union[None, Recorder] = None
    section: str = "MinBoxSize"
    name: str = "Small Boxes"
    eps: float = 1e-6 # added to the size of the boxes due to floating point precision

    def batch_end(self, l: Learner):
        with torch.no_grad():
            boxes = l.model.boxes.boxes
            small_boxes = detect_small_boxes(boxes, l.model.vol, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_(self.section, {self.name + f" (<{self.min_vol} before MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_(self.section, {self.name + f" (<{self.min_vol - self.eps} before MinBoxSize)": detect_small_boxes(boxes, l.model.vol, self.min_vol - self.eps).sum().detach().cpu().item()}, l.progress.partial_epoch_progress())
            if num_min_boxes > 0:
                replace_Z_by_cube_(boxes, small_boxes, self.min_vol + self.eps)
            small_boxes = detect_small_boxes(boxes, l.model.vol, self.min_vol)
            num_min_boxes = small_boxes.sum().detach().cpu().item()
            if self.recorder is not None:
                self.recorder.update_(self.section, {self.name + f" (<{self.min_vol} after MinBoxSize)": num_min_boxes}, l.progress.partial_epoch_progress())
                self.recorder.update_(self.section, {self.name + f" (<{self.min_vol - self.eps} after MinBoxSize)": detect_small_boxes(boxes, l.model.vol, self.min_vol - self.eps).sum().detach().cpu().item()}, l.progress.partial_epoch_progress())


@dataclass
class LossCallback(Callback):
    recorder: Recorder
    ds: Dataset
    name: str = "Loss"
    section: str = "LossCallback"

    def __post_init__(self):
        self.name = self.recorder.get_unique_name(self.section, self.name)

    def epoch_end(self, l: Learner):
        with torch.no_grad():
            data_in, data_out = self.ds[:]
            output = l.model(data_in)
            loss = l.loss_fn(output, data_out, self.name, l)
            self.recorder.update_(self.section, {self.name: loss.item()}, l.progress.current_epoch_iter)


