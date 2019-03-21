from __future__ import annotations
from dataclasses import dataclass, field
from typing import *
import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .exceptions import *
from collections import defaultdict
import pandas as pd

from .callbacks import CallbackCollection


@dataclass
class Recorder:
    _data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def update_(self, data_in: Dict[str, Any], index: Union[int, float]):
        data_no_tensors = {k: v if type(v) is not torch.Tensor else v.detach().cpu().item() for k,v in data_in.items()}
        self._data = self._data.combine_first(
            pd.DataFrame(data_no_tensors, [index])
        )

    def get_unique_name(self, name:str):
        i = 1
        while name in self._data.columns:
            name = f"{name}_{i}"
            i += 1
        self._data[name] = [] # adds this column to DataFrame
        return name

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def _repr_html_(self):
        return self._data._repr_html_()

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()


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
    model: Module
    loss_fn: Callable
    opt: optim.Optimizer
    callbacks: CallbackCollection = field(default_factory=CallbackCollection)
    recorder: Recorder = field(default_factory=Recorder)

    def __post_init__(self):
        self.progress = Progress(0,0,len(self.train_dl))
        self.callbacks.learner_post_init(self)

    def train(self, epochs):
        try:
            self.callbacks.train_begin(self)
            for epoch in trange(epochs, desc="Overall Training:"):
                self.callbacks.epoch_begin(self)
                for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False)):
                    self.batch_in, self.batch_out = batch
                    self.progress.increment()
                    self.callbacks.batch_begin(self)
                    self.opt.zero_grad()
                    self.model_out = self.model(self.batch_in)
                    self.loss = self.loss_fn(self.model_out, self.batch_out, self, self.recorder)
                    self.loss.backward()
                    self.callbacks.backward_end(self)
                    self.opt.step()
                    self.callbacks.batch_end(self)
                self.callbacks.epoch_end(self)
        except StopTrainingError as e:
            print(e)
        except KeyboardInterrupt:
            print(f"Stopped training at {self.progress.partial_epoch_progress()} epochs due to keyboard interrupt.")
