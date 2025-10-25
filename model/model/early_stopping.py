from typing import Literal
import torch
import copy

ESMode = Literal["min", "max"]

class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        mode: ESMode = "min",
        delta: float = 1e-4,
    ):
        self._patience = patience
        self._mode = mode
        self._delta = delta
        self._best = None
        self._counter = 0
        self._best_state = None

    def step(
        self,
        metric: float,
        model: torch.nn.Module,
    ) -> bool:
        if self._best is None:
            self._best = metric
            self._best_state = copy.deepcopy(model.state_dict())
            return False
        improvement = (metric - self._best) if self._mode == "max" else (self._best - metric)
        if improvement > self._delta:
            self._best = metric
            self._best_state = copy.deepcopy(model.state_dict())
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self._patience

    def load_best(
        self,
        model: torch.nn.Module,
    ) -> None:
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
