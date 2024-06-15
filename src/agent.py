import torch
import numpy as np


class Agent:
    def __init__(self):
        torch.manual_seed(42)
        self.model = torch.load(__file__[:-8] + "agent.pkl", map_location="cpu")

    def act(self, state):
        return np.argmax(self.model(torch.Tensor(state)).cpu().data.numpy())

