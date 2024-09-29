import os
import random
import sys
import sumolib
import src.env
from src.env import RoadSimulationEnv

GLOSA_RANGE = 0
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

    fuel = {
        "connected": 0,
        "ordinary": 0,
        "All": 0,
    }

    for i in range(episodes):
        done = False
        random.seed(42 + i)
        state = env.reset()[0]

        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            speeds.append(action)
            rewards.append(reward)

        cur = env.get_total_fuel()
        fuel["connected"] += cur["connected"]
        fuel["ordinary"] += cur["ordinary"]
        fuel["All"] += cur["All"]

    fuel["connected"] /= episodes
    fuel["ordinary"] /= episodes
    fuel["All"] /= episodes

    return (sum(rewards) / len(rewards),
            sum(speeds) / len(speeds) + src.env.MIN_SPEED,
            fuel
            )


class FixedPolicyAgent:
    def __init__(self, speed: int, min_speed: int = 0):
        self._speed = speed
        self._min_speed = min_speed
        return

    def act(self, state) -> int:
        return self._speed - self._min_speed


if __name__ == "__main__":
    net = sumolib.net.readNet('./nets/single_agent/500m/net.xml')

    # model = DQN("MlpPolicy", env, verbose=1)

    # model.learn(total_timesteps=10000, log_interval=4)

    # model.save("dqn_road_simulation")

    env_with_glosa = RoadSimulationEnv(net, sim_time=1200, use_gui=False, glosa_range=10000)
    rw_with_glosa, _, fuel_with_glosa = evaluate_policy(env_with_glosa, FixedPolicyAgent(30, src.env.MIN_SPEED), episodes=5)
    env_with_glosa.close()

    env = RoadSimulationEnv(net, sim_time=1200, use_gui=False, glosa_range=0)
    rw_without_glosa, _, fuel_without_glosa = evaluate_policy(env, FixedPolicyAgent(30, src.env.MIN_SPEED), episodes=5)
    print("with", rw_with_glosa, fuel_with_glosa, sep='\n')
    print("without glosa", rw_without_glosa, fuel_without_glosa, sep='\n')
    env.close()
