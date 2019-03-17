from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
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
    """Prevents boxes from getting too small during training."""
    min_vol: float = 1e-20

    def on_batch_end(self, learner: Learner):
        with torch.no_grad():
            small_boxes = learner.model.vol(learner.model.boxes.boxes) < self.min_vol
            if small_boxes.sum() > 0:
                learner.model.boxes.boxes[:,:,1][small_boxes] = learner.model.boxes.boxes[:,:,0][small_boxes] + self.min_vol ** (
                        1 / learner.model.boxes.boxes.shape[-1])


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


