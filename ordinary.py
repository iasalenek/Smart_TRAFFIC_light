# this is a debug file and it should be ignored
import os
import sys
import time

import sumolib
import src.env
from src.env import RoadSimulationEnv
from src import evaluate, agents, benchmark

import csv

GLOSA_RANGE = 0
INITIAL_STEPS = 1000
TRANSITIONS = 500000

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
    return cur_env, agents.DummyAgent()


if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    agent = agents.DummyAgent()

    results = benchmark.benchmark(get_env_agent, episodes=5)

    print(results)

    benchmark.save_res("ordinary" + str(time.time()) + ".csv", results)
