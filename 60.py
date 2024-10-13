# this is a debug file and it should be ignored
import os
import sys
import sumolib
import src.env
from src import agents, benchmark

INITIAL_STEPS = 1000
TRANSITIONS = 500000

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    agent = agents.FixedPolicyAgent(60, src.env.MIN_SPEED)

    results = benchmark.benchmark(agent, sim_time=1200, glosa_range=0, episodes=5)

    benchmark.save_res("60", results)
