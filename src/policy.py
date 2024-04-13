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
from traci import constants as tc

from src.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class SumoEnv(gym.Env):
    def __init__(self, edge_id, n_traffic_lights):
        traci.edge.subscribe(edge_id, varIDs=[tc.LAST_STEP_VEHICLE_ID_LIST])
        self.bound_steps = SIM_TIME // STEP_LENGTH
        self.steps = 0
        self.last_waiting = 0
        self.waiting_cons_coef = 0.7
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        # self.action_space = Box(low=self.min_speed, high=self.max_speed, dtype=np.float32)
        self.action_dim = 8  # split max/min speed in 8 parts
        self.actions = np.linspace(self.min_speed, self.max_speed, self.action_dim)
        self.action_space = Discrete(len(self.actions))
        self.edgeID = edge_id
        #density + queue + fuel_cons + cur_phases_for_all_lights

        self.obs_dim = 3 + n_traffic_lights
        obs_low = np.zeros((self.obs_dim,))
        obs_high = np.array([1, 1, np.inf] + [4 for _ in range(n_traffic_lights)])
        self.observation_space = Box(low=obs_low, high=obs_high)

    def reset(self, **kwargs):
        null_obs = np.zeros((self.obs_dim,))
        return null_obs, {}

    def compute_queue(self):
        stepVehicleIDs = set(
            traci.edge.getSubscriptionResults(self.edgeID)[tc.LAST_STEP_VEHICLE_ID_LIST]
        )
        queued = np.array([1 if traci.vehicle.getSpeed(car_id) < 0.1 else 0 for car_id in stepVehicleIDs])
        capacity = traci.lane.getLength(self.edgeID + "_0") / 1000.
        return sum(queued)/capacity


    def get_waiting(self):
        stepVehicleIDs = set(
            traci.edge.getSubscriptionResults(self.edgeID)[tc.LAST_STEP_VEHICLE_ID_LIST]
        )
        acc = 0.0
        for ids in stepVehicleIDs:
            acc += traci.vehicle.getAccumulatedWaitingTime(ids)
        return acc

    def step(
            self, action: ActType
    ) -> tuple[ndarray[Any, dtype[Any]], float | Any, bool | Any, dict[Any, Any]]:
        stepVehicleIDs = set(
            traci.edge.getSubscriptionResults(self.edgeID)[tc.LAST_STEP_VEHICLE_ID_LIST]
        )
        #acting
        for vehicleID in stepVehicleIDs:
            if traci.vehicle.getTypeID(vehicleID) == "connected":


                traci.vehicle.setSpeed(vehicleID, speed=(self.actions[int(action)] / 3.6))
                # traci.vehicle.setSpeed(vehicleID, speed=60 / 3.6)
        # next_obs
        num = traci.edge.getLastStepVehicleNumber(self.edgeID)
        density = num / traci.lane.getLength(self.edgeID + "_0") / 1000

        fuel_cons = traci.edge.getFuelConsumption(self.edgeID)
        queue = self.compute_queue()
        phases = [traci.trafficlight.getPhase(light) for light in traci.trafficlight.getIDList()]
        next_obs = np.array([density, queue, fuel_cons] + phases)
        waiting_time = self.get_waiting()
        reward = (1-self.waiting_cons_coef)*(waiting_time-self.last_waiting) - self.waiting_cons_coef * fuel_cons
        self.last_waiting = waiting_time
        done = self.steps < self.bound_steps
        self.steps+=1
        return next_obs, reward, done, {}
        pass

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
        x = x.astype(np.float32)
        x = F.relu(self.layer1(torch.from_numpy(x)))
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
