from src.env import RoadSimulationEnv
from src.evaluate import evaluate_policy

from typing import Callable, List

import csv


def benchmark(get_agent_env: Callable, episodes: int, ps_vehicle: List[float], ps_connected: List[float]):
    results = []

    for p_vehicle in ps_vehicle:
        for p_connected in ps_connected:
            env, agent = get_agent_env(p_vehicle, p_connected)
            _, fuel, time = evaluate_policy(env, agent, episodes=episodes)
            env.close()
            results.append({
                "p_vehicle": p_vehicle,
                "p_connected": p_connected,
                "connected_fuel": fuel["connected"],
                "ordinary_fuel": fuel["ordinary"],
                "overall_fuel": fuel["All"],
                "connected_time": time["connected"],
                "ordinary_time": time["ordinary"],
                "overall_time": time["All"],
            })

    return results


def save_res(filename: str, results):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
