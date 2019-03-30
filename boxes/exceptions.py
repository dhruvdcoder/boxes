from dataclasses import dataclass

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class StopTrainingError(Error):
    """Base class for exceptions which should stop training in this module."""
    pass

@dataclass
class MaxLoss(StopTrainingError):
    """Max Loss Exceeded"""
    loss: float
    max_loss: float

    def __str__(self):
        return f"Max Loss Exceeded: {self.loss} > {self.max_loss}"


@dataclass
class EarlyStopping(StopTrainingError):
    """Max Value Exceeded"""
    condition: str

    def __str__(self):
        return f"EarlyStopping: {self.condition}"

