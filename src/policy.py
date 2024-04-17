import random
from collections import namedtuple, deque
from typing import Optional, List, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traci
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Box, Discrete
from numpy import ndarray, dtype
from traci import StepListener

from src.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class SumoEnv(gym.Env):
    def __init__(self):

        self.bound_steps = SIM_TIME // STEP_LENGTH
        self.steps = 0
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.action_dim = 8  # split max/min speed in 8 parts
        self.actions = np.linspace(self.min_speed, self.max_speed, self.action_dim)
        self.action_space = Discrete(len(self.actions))
        self.n_traffic_lights = len(TRAFFIC_LIGTS)
        self.car_cur_edge = dict()
        self.stepLength = traci.simulation.getDeltaT()
        #phase for the nearest light, its time before the next phase, distance, distance for the nearest car.
        #I want to count accelerated fuel consumption on a current edge as a reward. (with a '-')

        self.obs_dim = 4
        obs_low = np.zeros((self.obs_dim,))
        self.INFTY = 100000
        obs_high = np.array([4, 100000, 100000, 100000])
        self.observation_space = Box(low=obs_low, high=obs_high)

    def reset(self, **kwargs):
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[dict[Any, ndarray[Any, dtype[Any]]], dict[Any, Any], bool | Any, dict[Any, Any]]:

        vehicleIDs = traci.vehicle.getIDList()
        vehicleIDs = list(filter(lambda x: traci.vehicle.getTypeID(x) == "connected", vehicleIDs))
        # acting
        for vehicleID, speed_index in action.items():
            if vehicleID in vehicleIDs:
                traci.vehicle.setSpeed(vehicleID, speed=(self.actions[int(speed_index)] / 3.6))
                # print("Set speed is: ", self.actions[int(speed_index)])
        # next_obs
        next_obs = dict()
        reward = dict()
        next_car_edge = dict()

        for car_id in vehicleIDs:
            TLS_list = traci.vehicle.getNextTLS(car_id)
            TLS_list = sorted(TLS_list, key=lambda x: x[2])
            if TLS_list:
                nearestTLS = TLS_list[0]
                tlsID, tlsIndex, dist, state = nearestTLS
                phase = traci.trafficlight.getPhase(tlsID)
                remaining_time = traci.trafficlight.getNextSwitch(tlsID)
            else:
                phase = 0
                remaining_time = self.INFTY
                dist = self.INFTY

            leader_id, leader_dist = traci.vehicle.getLeader(car_id)
            if leader_id == "":
                leader_dist = self.INFTY
            next_obs[car_id] = np.array([phase, remaining_time, dist, leader_dist])
            real_edge = traci.vehicle.getLaneID(car_id)
            next_car_edge[car_id] = (real_edge, self.car_cur_edge.get(car_id, ("", 0))[1] +
                                     self.stepLength * traci.vehicle.getFuelConsumption(car_id))
            reward[car_id] = next_car_edge[car_id][1]
        self.car_cur_edge = next_car_edge
        done = self.steps < self.bound_steps
        self.steps += 1
        return next_obs, reward, done, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BasePolicy(StepListener):
    def __init__(
            self,
            edgeIDs: List[str],
            trafficlightIDs: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(BasePolicy, self).__init__(**kwargs)
        self.edgeIDs = edgeIDs
        self.trafficlightIDs = trafficlightIDs
        # Проверяем наличие всех ребер
        assert set(self.edgeIDs).issubset(traci.edge.getIDList())
        # Проверяем наличие всех светофоров
        if trafficlightIDs is not None:
            assert set(self.trafficlightIDs).issubset(traci.trafficlight.getIDList())

    def step(self, t=0):
        return super().step(t)


class MaxSpeedPolicy(BasePolicy):
    def __init__(
            self,
            edgeIDs: List[str],
            trafficlightIDs: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        super(MaxSpeedPolicy, self).__init__(edgeIDs, trafficlightIDs, **kwargs)

    def step(self, t=0):
        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edgeIDs:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    traci.vehicle.setSpeed(vehicleID, speed=60 / 3.6)
        return super().step(t)
