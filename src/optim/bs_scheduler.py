import math
from typing import Optional, Callable


class BSScheduler:
    def __init__(self, initial_bs: int, max_bs=None, last_step: int = -1):
        self.initial_bs = initial_bs
        self.last_step = last_step
        self.batch_size = initial_bs
        self.max_bs = max_bs

        self._initial_step()

    def _initial_step(self):
        """Perform initial step and update batch size"""
        self.step()

    def get_batch_size(self) -> int:
        raise NotImplementedError

    def step(self, step: Optional[int] = None):
        """Advance step count and update batch size"""
        if step is None:
            self.last_step += 1
        else:
            self.last_step = step
        self.batch_size = self.get_batch_size()

    def state_dict(self):
        return {
            'initial_bs': self.initial_bs,
            'last_step': self.last_step,
            'batch_size': self.batch_size
        }

    def load_state_dict(self, state_dict):
        self.initial_bs = state_dict['initial_bs']
        self.last_step = state_dict['last_step']
        self.batch_size = state_dict['batch_size']


class LambdaBS(BSScheduler):
    def __init__(
        self,
        initial_bs: int,
        bs_lambda: Callable[[int], float],
        max_bs=None,
        last_step: int = -1
    ):
        self.bs_lambda = bs_lambda
        super().__init__(initial_bs, max_bs, last_step)

    def get_batch_size(self) -> int:
        if self.max_bs is not None:
            return min(self.max_bs, math.ceil(self.initial_bs * self.bs_lambda(self.last_step)))
        else:
            return math.ceil(self.initial_bs * self.bs_lambda(self.last_step))


class LambdaBSEps(BSScheduler):
    def __init__(
        self,
        initial_bs: int,
        bs_lambda: Callable[[int], float],
        initial_eps: float = None,
        max_bs=None,
        last_step: int = -1
    ):
        self.bs_lambda = bs_lambda
        self.initial_eps = initial_eps
        super().__init__(initial_bs, max_bs, last_step)

    def get_batch_size(self) -> int:
        if self.max_bs is not None:
            return min(self.max_bs, math.ceil(self.initial_bs * self.bs_lambda(self.last_step)))
        else:
            return math.ceil(self.initial_bs * self.bs_lambda(self.last_step))

    def get_epsilon(self) -> float:
        if self.initial_eps == None:
            return None
        else:
            return self.initial_eps * math.sqrt(self.initial_bs / self.get_batch_size())
