import random
from typing import Tuple, Dict


def evaluate_policy(env, agent, episodes=1) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    reward, _, fuel, time = evaluate_policy_with_speeds(env, agent, episodes=episodes)
    return reward, fuel, time


# evaluate_policy_with_speeds returns average_speed=-1 if no speed were recommended
def evaluate_policy_with_speeds(env, agent, episodes=1) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    total_speeds = []
    total_rewards = []

    fuel = {
        "connected": 0,
        "ordinary": 0,
        "All": 0,
    }

    time_cons = {
        "connected": 0,
        "ordinary": 0,
        "All": 0,
    }

    for i in range(episodes):
        done = False
        random.seed(42 + i)
        states = env.reset_all(random_probabilities=False)[0]
        while not done:
            actions = []
            for state in states:
                actions.append(agent.act(state))
            states, rewards, done, _, _ = env.step_all(actions)
            total_speeds.extend(actions)
            total_rewards.extend(rewards)

        cur_fuel = env.get_total_fuel()
        fuel["connected"] += cur_fuel["connected"]
        fuel["ordinary"] += cur_fuel["ordinary"]
        fuel["All"] += cur_fuel["All"]

        cur_time = env.get_total_time()
        time_cons["connected"] += cur_time["connected"]
        time_cons["ordinary"] += cur_time["ordinary"]
        time_cons["All"] += cur_time["All"]

    fuel["connected"] /= episodes
    fuel["ordinary"] /= episodes
    fuel["All"] /= episodes

    time_cons["connected"] /= episodes
    time_cons["ordinary"] /= episodes
    time_cons["All"] /= episodes

    average_reward = sum(total_rewards) / len(total_rewards)
    total_speeds = list(filter(lambda x: x is not None, total_speeds))
    average_recommended_speed = -1
    if len(total_speeds) > 0:
        average_recommended_speed = sum(total_speeds) / len(total_speeds)
    return average_reward, average_recommended_speed, fuel, time_cons
