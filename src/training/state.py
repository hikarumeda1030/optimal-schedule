from dataclasses import dataclass, field
import torch
from .steps import Steps


@dataclass
class TrainingState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    steps: Steps
    lr_scheduler: object
    lr_step_type: str
    bs_scheduler: object
    bs_step_type: str
    criterion: object
    epoch: int = 0
    eps: float = 0.1

@dataclass
class TrainingResults:
    train: list = field(default_factory=list)
    test: list = field(default_factory=list)
    lr_bs: list = field(default_factory=list)
    norm: list = field(default_factory=list)
