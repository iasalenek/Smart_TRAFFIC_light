
import numpy as np
from collections import deque
import torch as th
rng = np.random.default_rng()


class ReplayBuffer():
    def __init__(self, max_size):
        self.s = deque(maxlen=max_size)
        self.a = deque(maxlen=max_size)
        self.r = deque(maxlen=max_size)
        self.s_prime = deque(maxlen=max_size)
        self.done = deque(maxlen=max_size)

    def put(self, transition):
        self.s.append(transition[0])
        self.a.append(transition[1])
        self.r.append(transition[2])
        self.s_prime.append(transition[3])
        self.done.append(transition[4])

    def __len__(self):
        return len(self.s)

    def sample(self, n):
        inds = rng.choice([i for i in range(len(self.s))],
                          size=n, replace=False)
        return th.tensor(
            self.s, dtype=th.float)[inds], th.tensor(
            self.a)[inds], th.tensor(
            self.r)[inds], th.tensor(
                self.s_prime, dtype=th.float)[inds], th.tensor(
                    self.done)[inds]
