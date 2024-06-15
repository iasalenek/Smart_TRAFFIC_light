from typing import List, Optional

import sumolib.net.node

import random

import traci
from traci import StepListener

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import torch

import time

from src.dqn import DQN
from src.metrics import MeanEdgeFuelConsumption



class BasePolicy(StepListener):
    def __init__(
            self,
            edge_ids: List[str],
            min_speed: int,
            max_speed: int,
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(BasePolicy, self).__init__(**kwargs)
        self.edge_ids = edge_ids
        self._min_speed = min_speed
        self._max_speed = max_speed
        self.tf_ids = tf_ids
        # Проверяем наличие всех ребер
        assert set(self.edge_ids).issubset(traci.edge.getIDList())
        # Проверяем наличие всех светофоров
        if tf_ids is not None:
            assert set(self.tf_ids).issubset(traci.trafficlight.getIDList())

    def step(self, t=0):
        return super().step(t)

    def apply_action(self, vehicleID: str, speed: float):
        assert (
                traci.vehicle.getTypeID(vehicleID) == "connected"
        ), f"vehicle {vehicleID} is not connected"
        assert (speed >= self._min_speed) and (
                speed <= self._max_speed
        ), "The speed is beyond the limit"
        traci.vehicle.setSpeed(vehID=vehicleID, speed=speed / 3.6)


class FixedSpeedPolicy(BasePolicy):
    def __init__(
            self,
            speed: int,
            min_speed: int,
            max_speed: int,
            edge_ids: List[str],
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(FixedSpeedPolicy, self).__init__(edge_ids, tf_ids, **kwargs)
        self._min_speed = min_speed
        self._max_speed = max_speed
        assert (speed >= self._min_speed) and (speed <= self._max_speed)
        self.speed = speed

    def step(self, t=0):
        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edge_ids:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    self.apply_action(vehicleID, self.speed)
        return super().step(t)


class MyPolicy(BasePolicy):
    def __init__(
            self,
            min_speed: int,
            max_speed: int,
            edge_ids: List[str],
            net: sumolib.net.Net,
            step_length: int,
            fuel_metric: MeanEdgeFuelConsumption,
            initial_steps_count: int,
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        self._step_count = 0
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._initial_steps_count = initial_steps_count

        self.fuel_metric: MeanEdgeFuelConsumption = fuel_metric
        self.agent = DQN(2, self._max_speed - self._min_speed)
        self.obs, self.done, self.losses, self.ep_len, self.rew = [], False, 0, 0, 0

        self.eps = 0.1

        self.step_length = step_length

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = [], [], [], []
        self.rewards = []
        self.index = 128
        self.episodes = 10000
        self.epsilon = 0.3
        self.tf_ids = tf_ids
        self.net = net
        self.is_exploration = {}

        self.figure = plt.figure()

        self.prev_states: Dict[int, int] = {}
        self.prev_actions: Dict[int, int] = {}
        self.prev_fuel: Dict[int, int] = {}
        plt.ion()
        self._graph = plt.plot(self.losses)[0]
        plt.pause(1)

        super(MyPolicy, self).__init__(edge_ids=edge_ids, tf_ids=tf_ids, min_speed=min_speed, max_speed=max_speed,
                                       **kwargs)

    def get_obs(self, veh_id: int) -> np.ndarray:
        car_pos = traci.vehicle.getPosition(veh_id)[0]
        closest_red_light_tf = _INF_PROX
        time_to_next_switch = 0
        for tf_id in self.tf_ids:
            tf_pos = self.net.getNode(tf_id).getCoord()[0]
            if tf_pos >= car_pos and traci.trafficlight.getPhase(tf_id) == 0:
                time_to_next_switch = traci.trafficlight.getNextSwitch(tf_id) - traci.simulation.getTime()
                closest_red_light_tf = min(closest_red_light_tf, tf_pos - car_pos)
        return np.array([time_to_next_switch, closest_red_light_tf])

    def get_total_fuel_cons(self, veh_id: int):
        return self.fuel_metric._vehicleIdFuelDict[veh_id]

    def get_rew(self, veh_id: int):
        return -(self.get_total_fuel_cons(veh_id) - self.prev_fuel[veh_id])

    def step(self, t=0):
        # finishing previous transitions
        self.losses = 0
        cnt = 0
        cumulative_reward = 0
        rewardee_cnt = 0
        if len(self.prev_states) != 0:
            for edgeID in self.edge_ids:
                for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                    if traci.vehicle.getTypeID(vehicleID) == "connected":
                        if vehicleID in self.prev_states and vehicleID in self.prev_actions and vehicleID in self.prev_fuel:
                            obs = self.get_obs(vehicleID)
                            reward = self.get_rew(vehicleID)

                            transition = (self.prev_states[vehicleID], self.prev_actions[vehicleID], obs, reward, 0)

                            rewardee_cnt += 1
                            cumulative_reward += reward

                            if self._step_count < self._initial_steps_count:
                                self.agent.consume_transition(
                                    transition
                                )

                                self.prev_fuel[vehicleID] = self.get_total_fuel_cons(vehicleID)
                            else:
                                loss = self.agent.update(transition)
                                if loss and not self.is_exploration[vehicleID]:
                                    cnt += 1
                                    self.losses += loss
        if rewardee_cnt != 0:
            self.rewards.append(cumulative_reward / rewardee_cnt)

        if cnt != 0:
            # print(self.losses / cnt)
            self.losses_list += [self.losses / cnt]

            self._graph.remove()

            self._graph = plt.plot(self.losses_list)[0]

        # doing new transitions
        for edgeID in self.edge_ids:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    state = self.get_obs(vehicleID)

                    exploration = random.random() < self.eps

                    action: int = self.agent.act(np.array(state)) if not exploration else random.randint(0, self._max_speed - self._min_speed)

                    self.is_exploration[vehicleID] = exploration
                    self.prev_actions[vehicleID] = action
                    self.prev_states[vehicleID] = state

                    self.apply_action(vehicleID, self._min_speed + action)

                    try:
                        self.prev_fuel[vehicleID] = self.get_total_fuel_cons(vehicleID)
                    except KeyError:
                        self.prev_fuel[vehicleID] = 0
        self._step_count += 1



        return super().step(t)
