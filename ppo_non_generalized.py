import os
import sys
import time

from pathlib import Path

import sumolib
from src.env import RoadSimulationEnv
from src import agents
from stable_baselines3 import PPO

from src import benchmark

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_env_agent(p_vehicle: float, p_connected: float):
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')
    cur_env = RoadSimulationEnv(net,
                                sim_time=1200,
                                use_gui=False,
                                glosa_range=0,
                                p_vehicle=p_vehicle,
                                p_connected=p_connected,
                                )
    model = PPO.load("./ppo.pkl", cur_env)
    return cur_env, agents.StableBaselinesAgent(model)


if __name__ == "__main__":
    if not Path("./ppo_const_05_lr.pkl").exists():
        print("weights file does not exist")
        exit(0)

    ps_vehicle = [0.3]
    ps_connected = [0.2]
    res = benchmark.benchmark(get_env_agent, 10, ps_vehicle=ps_vehicle, ps_connected=ps_connected)

    print(res)

    benchmark.save_res("ppo_non_generalized" + str(time.time()) + ".csv", res)