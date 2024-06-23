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

INITIAL_STEPS = 1000
TRANSITIONS = 500000

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def evaluate_policy(env, agent, episodes=1):
    speeds = []
    rewards = []
    for i in range(episodes):
        done = False
        random.seed(42 + i)
        state = env.reset()[0]

        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            speeds.append(action)
            rewards.append(reward)

    return sum(rewards) / len(rewards), sum(speeds) / len(speeds) + 15


if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    env = RoadSimulationEnv(net, sim_time=300, use_gui=False)

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

    epoch_count = 100
    epoch_size = TRANSITIONS // epoch_count

    speeds = []
    res = []
    for i in range(epoch_count):
        random.seed()
        print("epoch #", i)
        for _ in tqdm.tqdm(range(epoch_size), total=epoch_size):
            # Epsilon-greedy policy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)

            next_state, reward, done, _, _ = env.step(action)
            dqn.update((state, action, next_state, reward, done))

            state = next_state if not done else env.reset()[0]

        reward, speed = evaluate_policy(env, dqn, 3)
        print("eval episode 1: rew:", reward, "speed:", speed)
        res.append(reward)
        speeds.append(speed)

        figure, axis = plt.subplots(1, 2)
        axis[0].plot(res)
        axis[1].plot(speeds)

        plt.show()
        plt.savefig('res.png')

        if i % 10 == 0:
            dqn.save()

    dqn.save()

    for metricListner in env.metrics_listeners:
        print(metricListner)
