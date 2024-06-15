from typing import List, Optional

import os
import random
import sys

import tqdm

import matplotlib.pyplot as plt
import numpy

import matplotlib.pyplot as plt
import sumolib
import traci

from src.policy import BasePolicy, FixedSpeedPolicy, MyPolicy
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime

from src.env import RoadSimulationEnv

from src.dqn import DQN

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 1
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 200
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
DEVICE = 'cpu'

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def evaluate_policy(env, agent, episodes=1):
    returns = []
    speeds = []
    for _ in range(episodes):
        done = False
        state = env.reset()[0]
        total_reward = 0.

        while not done:
            action = agent.act(state)
            speeds.append(action)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return returns, sum(speeds) / len(speeds) + 15

if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    env = RoadSimulationEnv(net, sim_time=1000, use_gui=False)

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()[0]

    print("doing initial steps")
    for _ in tqdm.tqdm(range(INITIAL_STEPS), total=INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()[0]

    i = 0

    epoch_count = 200
    epoch_size = TRANSITIONS // epoch_count

    speeds = []
    res = []
    for i in range(epoch_count):
        print("epoch #", epoch_count)
        for _ in tqdm.tqdm(range(epoch_size), total=epoch_size):
            # Epsilon-greedy policy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)

            next_state, reward, done, _, _ = env.step(action)
            dqn.update((state, action, next_state, reward, done))

            state = next_state if not done else env.reset()[0]

        rewards, speed = evaluate_policy(env, dqn, 1)
        res.append(rewards[0])
        speeds.append(speed)

        figure, axis = plt.subplots(1, 2)
        axis[0].plot(res)
        axis[1].plot(speeds)

        plt.show()
        plt.savefig('res.png')

    for metricListner in env.metrics_listeners:
        print(metricListner)
