from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed
        self.state = np.random.RandomState(self.seed)

    @abstractmethod
    def sample(self) -> int | float: ...

    def reset(self):
        self.state = np.random.RandomState(self.seed)


class PositiveNormal(Distribution):
    def __init__(
        self, mean: float = 0.0, variance: float = 1.0, seed: int | None = None
    ) -> None:
        super().__init__(seed)
        self.mean = mean
        self.variance = variance

    def sample(self):
        return int(np.clip(self.state.normal(self.mean, self.variance), 0, np.inf))


class Exponential(Distribution):
    def __init__(self, _lambda: float = 0.5, seed: int | None = None) -> None:
        super().__init__(seed)
        self._lamblda = _lambda

    def sample(self) -> int:
        return int(self.state.exponential(self._lamblda))
