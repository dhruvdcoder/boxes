from __future__ import annotations
from dataclasses import dataclass, field
from typing import *
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.autonotebook import tqdm, trange


@dataclass
class Learner:
    train_dl: DataLoader
    model: nn.Module
    loss_fn: Callable
    opt: optim.Optimizer

    def train(self, epochs):
        for epoch in trange(epochs, desc="Overall Training:"):
            for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False)):
                self.opt.zero_grad()
                output = self.model(batch[0])
                loss = self.loss_fn(output, batch[1])
                loss.backward()
                self.opt.step()
            tqdm.write(f"{loss.item()}")










