from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from .box_operations import *
from .exceptions import *
import numpy as np
import ipywidgets as widgets
from IPython.core.display import HTML, display
if TYPE_CHECKING:
    from .learner import Learner, Recorder


class Callback:
    def learner_post_init(self, learner: Learner):
        pass

    def train_begin(self, learner: Learner):
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


@dataclass
class LossCallback(Callback):
    recorder: Recorder
    ds: Dataset
    weighted: bool = True

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.epoch_end(learner)

    @torch.no_grad()
    def epoch_end(self, l: Learner):
        data_in, data_out = self.ds[:]
        output = l.model(data_in)
        l.loss_fn(output, data_out, l, self.recorder, weighted=self.weighted) # this logs the data to the recorder


@dataclass
class MetricCallback(Callback):
    recorder: Recorder
    ds: Dataset
    metric: Callable
    name: Union[str, None] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.metric.__name__
        self.name = self.recorder.get_unique_name(self.name)

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.epoch_end(learner)

    @torch.no_grad()
    def epoch_end(self, l: Learner):
        data_in, data_out = self.ds[:]
        metric_val = self.metric(l.model, data_in, data_out)
        self.recorder.update_({self.name: metric_val}, l.progress.current_epoch_iter)


@dataclass
class DisplayTable(Callback):
    recorder: Union[Recorder, None] = None

    def learner_post_init(self, learner: Learner):
        if self.recorder is None:
            self.recorder = learner.recorder

    @torch.no_grad()
    def train_begin(self, learner: Learner):
        self.out = widgets.Output()

    @torch.no_grad()
    def epoch_end(self, learner: Learner):
        self.out.clear_output()
        with self.out:
            display(self.recorder)


@dataclass
class StopAtMaxLoss(Callback):
    max_loss: float = 100.

    @torch.no_grad()
    def batch_end(self, learner: Learner):
        if learner.loss > self.max_loss:
            raise MaxLoss(learner.loss, self.max_loss)


@dataclass
class PercentIncreaseEarlyStopping(Callback):
    rec: Recorder
    metric_name: str
    percent_inc: float
    epoch_count: int = 0
    flag: Optional[str] = None

    def __post_init__(self):
        if self.epoch_count == 0:
            self.epoch_end = self._epoch_end_percent_only
        else:
            self.epoch_end = self._epoch_end_both

    @torch.no_grad()
    def _epoch_end_percent_only(self, learner: Learner):
        vals = self.rec[self.metric_name]
        min_val = vals.min()
        cur_val = vals.tail(1).item()
        if  cur_val > (1 + self.percent_inc) * min_val:
            if self.flag is not None:
                self.rec.update_({self.flag: True}, vals.tail(1).index.item())
            else:
                raise EarlyStopping(f"{self.metric_name} is now {cur_val}, which is more than {1 + self.percent_inc} times it's minimum of {min_val}.")

    @torch.no_grad()
    def _epoch_end_both(self, learner: Learner):
        vals = self.rec[self.metric_name]
        min_idx = vals.idxmin()
        cur_idx = vals.tail(1).index.item()
        if cur_idx >= min_idx + self.epoch_count and vals[cur_idx] > (1 + self.percent_inc) * vals[min_idx]:
            if self.flag is not None:
                self.rec.update_({self.flag: True}, cur_idx)
            else:
                raise EarlyStopping(f"{self.metric_name} is now {vals[cur_idx]}, which is more than {1 + self.percent_inc} times it's minimum of {vals[min_idx]}, which occurred {cur_idx - min_idx} >= {self.epoch_count} epochs ago.")


@dataclass
class ModelHistory(Callback):
    state_dict :List[dict] = field(default_factory=list)

    @torch.no_grad()
    def batch_end(self, learner: Learner):
        self.state_dict.append({k: v.detach().cpu().clone() for k, v in learner.model.state_dict().items()})

    def __getitem__(self, item):
        return self.state_dict[item]
