import dataclasses

import numpy as np


@dataclasses.dataclass
class CyclicAnnealing:
    n_cycles: int = 4
    n_iterations: int = 1_000
    R: float = 0.5
    beta: float = 1.0

    def __post_init__(self):
        self.reset()
        self._func = lambda t: 1.0 / (1.0 + np.exp(-(t * 18 - 3)))

    def step(self) -> float:
        self._iteration += 1
        t_over_m = self.n_iterations / self.n_cycles
        tau = np.mod((self._iteration - 1), np.ceil(t_over_m)) / t_over_m

        if self._iteration >= self.n_iterations:
            return self.beta

        if tau <= self.R:
            return self._func(tau) * self.beta
        else:
            return self.beta

    def reset(self):
        self._iteration = 0
        print(f"Resetting cyclic annealing: {self._iteration}")
