import math
from typing import List, Optional

import os
import random
import sys

import torch
import torch.nn as nn
import sumolib
import traci

from src.policy import (BasePolicy, MaxSpeedPolicy, device,
                        SumoEnv, DQN, ReplayMemory, Transition)
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime
import torch.optim as optim

# CONFIG_PATH = "nets/single_agent/500m/config.sumocfg"
# SIM_TIME = 1 * 60 * 60
# P_VEHICLE = 0.3
# P_CONNECTED = 0.2
# MIN_SPEED = 45
# MAX_SPEED = 60
# STEP_LENGTH = 1
# EDGE_IDS = ["E1"]
# TRAFFIC_LIGTS = None
# VEHICLETYPE_IDS = ["ordinary", "connected"]
# RANDOM_SEED = 42
# SUMO_SEED = 42
# USE_GUI = True
from src.constants import *

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def runSimulation(
        configPath: str = CONFIG_PATH,
        simTime: int = SIM_TIME,
        policyListner: Optional[BasePolicy] = MaxSpeedPolicy,
        pVehicle: float = P_VEHICLE,
        pConnected: float = P_CONNECTED,
        minSpeed: float = MIN_SPEED,
        maxSpeed: float = MAX_SPEED,
        stepLength: float = STEP_LENGTH,
        edgeIDs: List[str] = EDGE_IDS,
        trafficlightIDs: Optional[List[str]] = None,
        vehicletypeIDs: Optional[List[str]] = VEHICLETYPE_IDS,
        randomSeed: int = RANDOM_SEED,
        sumoSeed: int = SUMO_SEED,
        useGUI: bool = USE_GUI,
):
    if useGUI:
        sumoBinary = sumolib.checkBinary("sumo-gui")
    else:
        sumoBinary = sumolib.checkBinary("sumo")

    sumoCmd = [
        sumoBinary,
        "-c", configPath,
        "--start",
        "--step-length", str(stepLength),
        "--quit-on-end",
        "--seed", str(sumoSeed),
        "--time-to-teleport", "-1",  # Телепортация автомобилей отключена
    ]

    traci.start(sumoCmd)

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return policy_net(state).max(1).indices.view(1, 1) //TODO
                return torch.argmax(policy_net(state))

        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.float32)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if policyListner is not None:
        policyListner = policyListner(
            edgeIDs=edgeIDs,
            trafficlightIDs=trafficlightIDs,
        )

    metrics = []
    env = SumoEnv(EDGE_IDS[0], 2)

    # Get number of actions from gym action space
    n_actions = env.action_dim  # TODO
    # Get the number of state observations
    # state, info = env.reset()
    state, info = env.reset()  # TODO
    n_observations = len(state)
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for _ in range(num_episodes):

        metricsListners = []
        for edgeID in edgeIDs:
            metricsListners += [
                MeanEdgeTime(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
                MeanEdgeFuelConsumption(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
            ]

        # traci.addStepListener(policyListner)
        for metricListner in metricsListners:
            traci.addStepListener(metricListner)

        veh_id = 0
        random.seed(randomSeed)

        state, info = env.reset()

        for _ in range(simTime // STEP_LENGTH):

            if random.random() < pVehicle * stepLength:

                if random.random() < pConnected:
                    traci.vehicle.add(
                        vehID=veh_id, routeID="r_0", departLane="best", typeID="connected"
                    )
                else:
                    traci.vehicle.add(
                        vehID=veh_id, routeID="r_0", departLane="best", typeID="ordinary"
                    )

                traci.vehicle.setSpeed(
                    vehID=veh_id, speed=random.randint(minSpeed, maxSpeed) / 3.6
                )
                veh_id += 1

            traci.simulationStep()
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        traci.close()
        metrics = metricsListners

    return metrics





if __name__ == "__main__":

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    steps_done = 0
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4



    metricsListners = runSimulation()

    for metricListner in metricsListners:
        print(metricListner)
