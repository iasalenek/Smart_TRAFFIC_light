import numpy as np
import torch
from gymnasium import make
from torch import nn, optim
from torch.nn import functional as F
from typing import Tuple
from collections import deque
import random

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 1
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
DEVICE = 'mps'


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self._step = 0
        self._replay_max_len = 1000000
        self.replay_buffer = deque(maxlen=self._replay_max_len)
        self.local = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)).to(DEVICE)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)).to(DEVICE)
        torch.manual_seed(42)
        self.optimizer = optim.Adam(self.local.parameters(), lr=LEARNING_RATE)
        self._lr = LEARNING_RATE

    def consume_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for _ in range(BATCH_SIZE):
            state, action, next_state, reward, done = self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return torch.Tensor(np.array(states)).to(DEVICE), torch.LongTensor(np.array(actions)).to(DEVICE), torch.Tensor(
            np.array(next_states)).to(DEVICE), torch.Tensor(np.array(rewards)).to(DEVICE), torch.Tensor(
            np.array(dones)).to(DEVICE)

    def train_step(self, batch) -> float:
        states, actions, next_states, rewards, dones = batch
        q_targets_next = self.model(next_states).detach().max(1)[0]
        q_targets = (rewards + GAMMA * q_targets_next * (1 - dones)).unsqueeze(1)
        q_expected = self.local(states).gather(1, actions.unsqueeze(1))

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.model.load_state_dict(self.local.state_dict())

    def act(self, state, target=False):
        if target:
            self.model.eval()
            with torch.no_grad():
                action_values = self.model(torch.from_numpy(np.array(state)).float().to(DEVICE))
            self.model.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            self.local.eval()
            with torch.no_grad():
                action_values = self.local(torch.from_numpy(np.array(state)).float().to(DEVICE))
            self.local.train()
            return np.argmax(action_values.cpu().data.numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        loss = None
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            loss = self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1
        return loss if loss else None

    def save(self):
        torch.save(self.model, "agent.pkl")

if __name__ == "__main__":
    env = make("LunarLander-v2")

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()[0]

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()[0]

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()[0]

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            dqn.save()
