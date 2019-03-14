from __future__ import annotations
from dataclasses import dataclass, field
from typing import *
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm, trange


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
        return self.current_batch_iter / num_batches

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
        for epoch in trange(epochs, desc="Overall Training:"):
            for c in self.callbacks:
                c.epoch_begin(self)
            for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False)):
                self.progress.increment()
                for c in self.callbacks:
                    c.batch_begin(self)
                self.opt.zero_grad()
                self.output = self.model(batch[0])
                self.loss = self.loss_fn(self.output, batch[1], self)
                self.loss.backward()
                self.opt.step()
                for c in self.callbacks:
                    c.batch_end(self)
            for c in self.callbacks:
                c.epoch_end(self)


class Callback:
    def learner_post_init(self, learner:Learner):
        pass

    def epoch_begin(self, learner:Learner):
        pass

    def batch_begin(self, learner:Learner):
        pass

    def batch_end(self, learner:Learner):
        pass

    def epoch_end(self, learner:Learner):
        pass


@dataclass
class LossCallback(Callback):
    ds: Dataset
    name: str = "Loss"

    def learner_post_init(self, l):
        if "loss" not in l.metadata.keys():
            l.metadata["loss"] = pd.DataFrame()
        name = self.name
        i = 1
        while name in l.metadata["loss"].columns:
            name = f"{self.name} ({i})"
            i += 1
        self.name = name
        l.metadata["loss"][self.name] = []

    def epoch_end(self, l):
        with torch.no_grad():
            data = self.ds[:]
            output = l.model(data[0])
            loss = l.loss_fn(output, data[1])
            l.metadata["loss"] = l.metadata["loss"].combine_first(
                pd.DataFrame({self.name: loss.item()}, [l.progress.current_epoch_iter])
            )
