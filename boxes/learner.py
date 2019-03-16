from __future__ import annotations
from dataclasses import dataclass, field
from typing import *
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm, trange
from collections import defaultdict


@dataclass
class Progress:
    current_epoch_iter: int = 0
    current_batch_iter: int = 0
    num_batches: int = 0

    def increment(self):
        self.current_batch_iter += 1
        if self.current_batch_iter == self.num_batches:
            self.current_batch_iter = 0
            self.current_epoch_iter += 1

    def percent_epoch_complete(self):
        return self.current_batch_iter / self.num_batches

    def partial_epoch_progress(self):
        return self.current_epoch_iter + self.percent_epoch_complete()


@dataclass
class Learner:
    train_dl: DataLoader
    model: nn.Module
    loss_fn: Callable
    opt: optim.Optimizer
    callbacks: Collection[Callback] = field(default_factory=list)
    progress: Progress = field(default_factory=Progress)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.progress = Progress(0,0,len(self.train_dl))
        for c in self.callbacks:
            c.learner_post_init(self)

    def train(self, epochs):
        self.status = "train"
        for epoch in trange(epochs, desc="Overall Training:"):
            for c in self.callbacks:
                c.epoch_begin(self)
            for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False)):
                self.batch_in, self.batch_out = batch
                self.progress.increment()
                for c in self.callbacks:
                    c.batch_begin(self)
                self.opt.zero_grad()
                self.model_output = self.model(self.batch_in)
                self.loss = self.loss_fn(self.model_output, self.batch_out, "train", self)
                self.loss.backward()
                for c in self.callbacks:
                    c.backward_end(self)
                self.opt.step()
                for c in self.callbacks:
                    c.batch_end(self)
            for c in self.callbacks:
                c.epoch_end(self)


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


@dataclass
class Recorder:
    _data: Dict[str, pd.DataFrame] = field(default_factory=lambda: defaultdict(pd.DataFrame))

    def update_(self, section:str, data: Dict[str, Any], index: Union[int, float]):
        self[section] = self[section].combine_first(
            pd.DataFrame(data, [index])
        )

    def get_unique_name(self, section:str, name:str):
        i = 1
        while name in self[section].columns:
            name = f"{name}_{i}"
            i += 1
        self[section][name] = [] # adds this column to DataFrame
        return name

    def __getitem__(self, name:str):
        return self._data[name]

    def __setitem__(self, name:str, val:pd.DataFrame):
        self._data[name] = val
