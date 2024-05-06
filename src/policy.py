from typing import List, Optional

import sumolib.net.node


import traci
from traci import StepListener

from typing import Dict

import numpy as np

from Smart_TRAFFIC_light.src.dqn import DQN
from Smart_TRAFFIC_light.src.metrics import MeanEdgeFuelConsumption

POLICY_MIN_SPEED = 30
POLICY_MAX_SPEED = 60

GREEN_LIGHT_TF = 0
RED_LIGHT_TF = 0

INF_PROX = 2000


class BasePolicy(StepListener):
    def __init__(
            self,
            edge_ids: List[str],
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(BasePolicy, self).__init__(**kwargs)
        self.edge_ids = edge_ids
        self.tf_ids = tf_ids
        # Проверяем наличие всех ребер
        assert set(self.edge_ids).issubset(traci.edge.getIDList())
        # Проверяем наличие всех светофоров
        if tf_ids is not None:
            assert set(self.tf_ids).issubset(traci.trafficlight.getIDList())

    def step(self, t=0):
        return super().step(t)

    @staticmethod
    def apply_action(vehicleID: str, speed: float):
        assert (
                traci.vehicle.getTypeID(vehicleID) == "connected"
        ), f"vehicle {vehicleID} is not connected"
        assert (speed >= POLICY_MIN_SPEED) and (
                speed <= POLICY_MAX_SPEED
        ), "The speed is beyond the limit"
        traci.vehicle.setSpeed(vehID=vehicleID, speed=speed / 3.6)


class FixedSpeedPolicy(BasePolicy):
    def __init__(
            self,
            speed: int,
            edge_ids: List[str],
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(FixedSpeedPolicy, self).__init__(edge_ids, tf_ids, **kwargs)
        assert (speed >= POLICY_MIN_SPEED) and (speed <= POLICY_MAX_SPEED)
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
            net: sumolib.net.Net,
            step_length: int,
            edge_ids: List[str],
            fuel_metric: MeanEdgeFuelConsumption,
            tf_ids: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        exp_replay_size = 1
        input_dim = 1
        output_dim = POLICY_MAX_SPEED - POLICY_MIN_SPEED
        self.fuel_metric: MeanEdgeFuelConsumption = fuel_metric
        self.agent = DQN(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                         exp_replay_size=exp_replay_size)
        self.obs, self.done, self.losses, self.ep_len, self.rew = [], False, 0, 0, 0

        self.step_length = step_length

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = [], [], [], []
        self.rew = []
        self.index = 128
        self.episodes = 10000
        self.epsilon = 0.3
        self.tf_ids = tf_ids
        self.net = net

        self.prev_obs: Dict[int, int] = {}
        self.prev_actions: Dict[int, int] = {}
        self.prev_fuel: Dict[int, int] = {}

        super(MyPolicy, self).__init__(edge_ids, tf_ids, **kwargs)

    def get_obs(self, veh_id: int) -> np.ndarray:
        car_pos = traci.vehicle.getPosition(veh_id)[0]
        closest_red_light_tf = INF_PROX
        for tf_id in self.tf_ids:
            tf_pos = self.net.getNode(tf_id).getCoord()[0]
            if tf_pos >= car_pos and traci.trafficlight.getPhase(tf_id) == RED_LIGHT_TF:
                closest_red_light_tf = min(closest_red_light_tf, tf_pos - car_pos)
        return np.array([closest_red_light_tf])

    def get_total_fuel_cons(self, veh_id: int):
        return -self.fuel_metric._vehicleIdFuelDict[veh_id]

    def get_rew(self, veh_id: int):
        return -(self.get_total_fuel_cons(veh_id) - self.prev_fuel[veh_id])

    def step(self, t=0):
        if len(self.prev_obs) != 0:
            for edgeID in self.edge_ids:
                for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                    if traci.vehicle.getTypeID(vehicleID) == "connected":
                        if vehicleID in self.prev_obs and vehicleID in self.prev_actions and vehicleID in self.prev_fuel:
                            obs = self.get_obs(vehicleID)
                            reward = self.get_rew(vehicleID)
                            self.agent.collect_experience(
                                [self.prev_obs[vehicleID], self.prev_actions[vehicleID].item(), reward, obs])
                            self.rew.append(reward)
                            self.prev_fuel[vehicleID] = self.get_total_fuel_cons(vehicleID)

            self.index += 1
            if self.index > 128:
                self.index = 0
                for j in range(4):
                    loss = self.agent.train(batch_size=16)
                    self.losses += loss

        for edgeID in self.edge_ids:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    obs = self.get_obs(vehicleID)
                    action = self.agent.get_action(obs, POLICY_MAX_SPEED - POLICY_MIN_SPEED, self.epsilon)

                    self.prev_actions[vehicleID] = action
                    self.prev_obs[vehicleID] = obs
                    action += POLICY_MIN_SPEED
                    self.apply_action(vehicleID, max(min(POLICY_MAX_SPEED, action.item()), POLICY_MIN_SPEED))
                    try:
                        self.prev_fuel[vehicleID] = self.get_total_fuel_cons(vehicleID)
                    except KeyError:
                        self.prev_fuel[vehicleID] = 0

        return super().step(t)
