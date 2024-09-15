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

from stable_baselines3.dqn import DQN

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


class FixedPolicyAgent:
    def __init__(self):
        return

    def act(self, state) -> int:
        return 45


if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    env = RoadSimulationEnv(net, sim_time=300, use_gui=False)

    # model = DQN("MlpPolicy", env, verbose=1)

    # model.learn(total_timesteps=10000, log_interval=4)

    # model.save("dqn_road_simulation")

    evaluate_policy(env, FixedPolicyAgent())

    # for metricListner in env.metrics_listeners:
    #     print(metricListner)
