import numpy as np
from torch import nn
import torch as th
import torch.nn.functional as F
import random
from torch import optim
from .replayBuffer import ReplayBuffer

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

class QNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(20,3125)
        self.fc5 = nn.Linear(3125,3125)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc5(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 3124)
        else:
            return out.argmax().item()

def train(q, q_target, replay_buffer, optimizer, batch_size, gamma, updates_number=10):
    for _ in range(updates_number):

        s, a, r, s_prime, done_mask = replay_buffer.sample(batch_size)

        # полезность
        q_out = q(s)
        a = a.unsqueeze(1)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r.unsqueeze(1) +  gamma * max_q_prime * done_mask.unsqueeze(1)

        loss = F.smooth_l1_loss(q_a, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def run(initEnv, learning_rate, gamma, buffer_max_size, batch_size, target_update_interval,
        replay_buffer_start_size, print_interval=20, n_episodes=10000):

    q = QNetwork()
    q_target = QNetwork()

    q_target.load_state_dict(q.state_dict())

    replay_buffer = ReplayBuffer(max_size=buffer_max_size)

    score = 0.0

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        env = initEnv()
        s, _ = env.step(0)
        s = np.array(s)
        s = s.reshape(-1)
        for _ in range(1000):
            
            a = q.sample_action(th.from_numpy(s).float(), epsilon)
            s_prime, r = env.step(a)
            r *= -1
            done_mask = 1.0
            replay_buffer.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
        env.close()
        print(len(replay_buffer))
        if len(replay_buffer) > replay_buffer_start_size:
            train(q, q_target, replay_buffer, optimizer, batch_size, gamma)

        if n_epi % target_update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        print("# of episode :{}, abg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%"
                .format(n_epi, score/ print_interval, len(replay_buffer), epsilon * 100))
        score = 0.0

if __name__ == "__main__":
    print(2)