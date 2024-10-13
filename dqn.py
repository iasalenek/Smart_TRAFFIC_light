import os
import pathlib
import random
import sys
import time
import numpy as np

from typing import Callable

from pathlib import Path

import sumolib
import src.env
from src.env import RoadSimulationEnv
from src import evaluate, agents
from stable_baselines3 import DQN

from src import helper

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":

    callback = helper.SaveOnBestTrainingRewardCallback(check_freq=20, log_dir="./dqn", verbose=1)

    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    env = RoadSimulationEnv(net, sim_time=1200, use_gui=False, glosa_range=0)

    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.5,
                tensorboard_log="./log")

    totrain = False
    toload = False

    if toload and Path("./dqn.pkl").exists:
        print("loading existing weights")
        model.load("./dqn.pkl")

        if totrain:
            total_timesteps = int(1e3)
            model.learn(total_timesteps=total_timesteps, callback=callback)

        rw, fuel = evaluate.evaluate_policy(env, agents.StableBaselinesAgent(model), episodes=5)
        print("res", rw, fuel, sep='\n')
        exit(0)

    total_timesteps = int(1e5)

    print("start training")

    model.learn(total_timesteps=total_timesteps)
    env.close()

    model.save("dqn" + str(total_timesteps) + str(time.time()))

    rw, fuel = evaluate.evaluate_policy(env, agents.StableBaselinesAgent(model), episodes=5)
    print("res", rw, fuel, sep='\n')
    env.close()
