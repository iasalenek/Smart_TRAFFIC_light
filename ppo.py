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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src import helper, benchmark

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def get_env_agent(p_vehicle: float, p_connected: float):
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')
    cur_env = RoadSimulationEnv(net,
                                sim_time=1200,
                                use_gui=False,
                                glosa_range=0,
                                p_vehicle=p_vehicle,
                                p_connected=p_connected,
                                )
    model = PPO('MlpPolicy', cur_env, verbose=1, learning_rate=linear_schedule(0.8),
                tensorboard_log="./log", n_epochs=10)
    model.load("./ppo.pkl")
    return cur_env, agents.StableBaselinesAgent(model)


if __name__ == "__main__":
    train = False
    train_further = False
    train_further_steps = 100000

    callback = helper.SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="./ppo", verbose=1)

    if not train and Path("./ppo.pkl").exists:
        if train_further:
            net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')
            cur_env = RoadSimulationEnv(net,
                                        sim_time=1200,
                                        use_gui=False,
                                        glosa_range=0,
                                        )

            custom_objects = {'learning_rate': linear_schedule(0.03)}

            model = PPO.load("./ppo.pkl", custom_objects=custom_objects)
            model.set_env(cur_env)

            total_timesteps = int(1e5)

            print("start further training")

            model.learn(total_timesteps=total_timesteps, callback=callback)
            model.save("ppo" + str(total_timesteps) + str(time.time()))

            exit(0)

        ps_vehicle = [0.1, 0.3, 0.5]
        ps_connected = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        res = benchmark.benchmark(get_env_agent, 10, ps_vehicle=ps_vehicle, ps_connected=ps_connected)
        benchmark.save_res("ppo" + str(time.time()) + ".csv", res)
        exit(0)

    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')
    cur_env = RoadSimulationEnv(net,
                                sim_time=1200,
                                use_gui=False,
                                glosa_range=0,
                                )

    cur_env = Monitor(cur_env, "./ppo")

    model = PPO('MlpPolicy', cur_env, verbose=1, learning_rate=linear_schedule(0.1),
                tensorboard_log="./log", n_epochs=10, gamma=1)

    total_timesteps = int(1e5)

    print("start training")

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo" + str(total_timesteps) + str(time.time()))
    cur_env.close()

    rw, fuel, _ = evaluate.evaluate_policy(RoadSimulationEnv(net,
                                                             sim_time=1200,
                                                             use_gui=False,
                                                             glosa_range=0,
                                                             ), agents.StableBaselinesAgent(model), episodes=5)
