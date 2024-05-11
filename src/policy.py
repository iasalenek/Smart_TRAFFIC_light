import random
from collections import namedtuple, deque
from math import ceil
from typing import Optional, List, Any, Tuple, Dict, Set

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traci
import sumolib
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Box, Discrete
from numpy import ndarray, dtype
from traci import StepListener
from traci import constants as tc
from vector2d import Vector2D
from src.constants import *

import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# net = sumolib.net.readNet("net.xml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class SumoEnv(gym.Env):
    def __init__(self, edgeID):
        """TODO переписать окружение под одно ребро.
        принимать решения через определенные промежутки, а не на каждом шагу. DONE
        делать шаги оптимизации не на каждом шаге симуляции, а через определенное количество шагов DONE
        сделать ohe-hot-encoding для фаз светофора DONE
        присваивать всем машинам на ребре одну награду DONE
        выставлять done, когда машина заканчивает эпизод . P.s вроде бесполезно
        нарисовать графики
        """
        self.edgeID = edgeID
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.action_dim = 10  # split max/min speed in 8 parts
        self.actions = np.linspace(self.min_speed, self.max_speed, self.action_dim)
        self.action_space = Discrete(len(self.actions))
        self.n_traffic_lights = len(TRAFFIC_LIGTS)
        self.stepLength = traci.simulation.getDeltaT()
        traci.edge.subscribe(self.edgeID, varIDs=[tc.LAST_STEP_VEHICLE_ID_LIST])
        # lanes = list(filter(lambda lane: traci.lane.getEdgeID(lane) == self.edgeID,  traci.lane.getIDList()))
        # self.toNode = net.getEdge(self.edgeID).getToNode().getCoord()
        # self.fromNode = net.getEdge(self.edgeID).getFromNode().getCoord()
        # self.X = self.toNode[0] - self.fromNode[0]
        # self.Y = self.toNode[1] - self.fromNode[1]
        # self.edgeVector = Vector2D(self.X, self.Y)
        # self.vehicleLength = traci.vehicle.getLength("connected")
        print(traci.vehicletype.getIDList())
        self.vehicleLength = (traci.vehicletype.getLength('connected'))
        self.lanes_number = traci.edge.getLaneNumber(self.edgeID)
        self.lane_length = traci.lane.getLength(self.edgeID + '_0')

        self.number_of_fragments = ceil(self.lane_length / self.vehicleLength)
        # self.fractions = [self.edgeVector * self.frac * i for i in range(1, ceil(self.lane_length/self.vehicleLength))]

        #color_state for the nearest light ohe hot encoded, its time before the next color_state, distance, distance for the nearest car
        #and two flags if there is an upcoming light or a leading car ahead.
        #I want to count accelerated fuel consumption on a current edge as a reward. (with a '-')
        self.obs_dim = 6 + self.number_of_fragments


    def mark_fractions(self, carID, carsIDs):
        laneID = traci.vehicle.getLaneID(carID)
        toNode, fromNode = traci.lane.getShape(laneID)
        laneX = fromNode[0] - toNode[0]
        laneY = fromNode[1] - toNode[1]
        laneVector = Vector2D(laneX, laneY)
        on_frac = [0] * self.number_of_fragments
        for car in carsIDs:
            if car != carID and laneID == traci.vehicle.getLaneID(car):
                x, y = traci.vehicle.getPosition(carID)
                carVector = Vector2D(x, y)
                proj = Vector2D.Project(carVector, laneVector)
                index = proj.length / self.vehicleLength
                on_frac[int(index)] = 1
        return on_frac

    def reset(self, **kwargs):
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[dict[Any, ndarray[Any, dtype[Any]]], dict[Any, Any], bool, dict[Any, Any]]:

        vehicleIDs = traci.edge.getLastStepVehicleIDs(self.edgeID)
        vehicleIDs = list(filter(lambda x: traci.vehicle.getTypeID(x) == "connected", vehicleIDs))
        # acting
        for vehicleID, speed_index in action.items():
            if vehicleID in vehicleIDs:
                traci.vehicle.setSpeed(vehicleID, speed=(self.actions[int(speed_index)] / 3.6))
                # print("Set speed is: ", self.actions[int(speed_index)])
        # next_obs
        next_obs = dict()
        reward = dict()
        is_light = False
        # is_leading_car = True
        for car_id in vehicleIDs:
            TLS_list = traci.vehicle.getNextTLS(car_id)
            TLS_list = sorted(TLS_list, key=lambda x: x[2])
            if TLS_list:
                is_light = True
                nearestTLS = TLS_list[0]
                tlsID, tlsIndex, dist, color_state = nearestTLS
                color_state = color_state.upper()
                if color_state == 'G':
                    color_state = [1, 0, 0]
                elif color_state == 'R':
                    color_state = [0, 1, 0]
                elif color_state == 'Y':
                    color_state = [0, 0, 1]
                else:
                    print(color_state)
                    raise Exception("weird color")  # potentially a light can be off ('O')
                    # or in a lowercase color (r g y)

                remaining_time = traci.trafficlight.getNextSwitch(tlsID)
            else:
                color_state = 0
                remaining_time = 0
                dist = 0

            # leader_id, leader_dist = traci.vehicle.getLeader(car_id)
            # if leader_id == "":
            #     is_leading_car = False
            #     leader_dist = 0
            marks = self.mark_fractions(car_id, vehicleIDs)
            next_obs[car_id] = np.array(color_state + [remaining_time, dist, is_light] + marks)
            reward[car_id] = -traci.vehicle.getFuelConsumption(car_id) * self.stepLength
            # reward[car_id] = -traci.edge.getFuelConsumption(self.edgeID)

        done = False
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
