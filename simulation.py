import math
from typing import List, Optional

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import sumolib
import traci
import matplotlib.pyplot as plt

loss_values = []

from src.policy import (BasePolicy, MaxSpeedPolicy, device,
                        SumoEnv, DQN, ReplayMemory, Transition)
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime
import torch.optim as optim

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

    def select_action(state):  # state is actually a tuple of many states
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return policy_net(state).max(1).indices.view(1, 1) //TODO
                # print(policy_net(torch.from_numpy(state)).max(-1).indices.item() == torch.argmax(policy_net(torch.from_numpy(state))))
                return torch.argmax(policy_net(torch.from_numpy(state)))

        else:
            return torch.tensor([[envs[0].action_space.sample()]], device=device, dtype=torch.float32)

    def optimize_model():
        global last_optimization
        if len(memory) < BATCH_SIZE:
            return
        if last_optimization < OPTIMIZATION_FREQUENCY:
            last_optimization += 1
            return
        last_optimization = 0
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)

        #TODO
        non_final_next_states = torch.tensor(np.array(batch.next_state))

        state_batch = torch.tensor(np.array(batch.state))
        # print(state_batch.shape)
        # print(tuple(torch.tensor(arr) for arr in batch.action))
        # action_batch = torch.cat(tuple(torch.tensor(arr).reshape(1) for arr in batch.action))
        action_batch = torch.tensor(tuple(arr.reshape(1).to(torch.int64) for arr in batch.action))

        # reward_batch = torch.cat(tuple(torch.tensor(arr) for arr in batch.reward))
        # reward_batch = np.concatenate(batch.reward)
        reward_batch = torch.tensor(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(state_batch.shape, torch.unsqueeze(action_batch, 1).shape)
        # print(state_batch.shape)
        state_action_values = policy_net(state_batch).gather(1, torch.unsqueeze(action_batch, 1))

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
        loss_values.append(loss.item())
        # print(loss.item())
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
    envs = []
    for edge in edgeIDs:
        envs.append(SumoEnv(edgeID=edge))

    # Get number of actions from gym action space
    n_actions = envs[0].action_dim  # TODO
    # Get the number of state observations
    # state, info = env.reset()
    # state, info = envs[0].reset()  # TODO
    n_observations = envs[0].obs_dim
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    traci.setLegacyGetLeader(False)

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 7

    for k in range(num_episodes):
        steps_done = 0
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
        state = []
        for i in range(len(envs)):
            st, _ = envs[i].reset()
            state.append(st)

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

            stepVehicles = traci.vehicle.getIDList()
            for k in range(len(envs)):
                action = dict()

                for car_id, obs in state[k].items():
                    action[car_id] = select_action(obs)

                next_state, reward, done, _ = envs[k].step(action)
                for car_id, obs in state[k].items():
                    if car_id in next_state:
                        rew = torch.tensor([reward], device=device)
                        memory.push(obs, action[car_id], next_state[car_id], rew)
                    elif car_id not in stepVehicles:
                        memory.push(obs, action[car_id], None, None)

                # reward = torch.tensor([reward], device=device)
                state[k] = next_state

            optimize_model()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            traci.simulationStep()
            steps_done += 1

        print(f"Episode {k} is finished!")

        metrics = metricsListners
    traci.close()
    return metrics, list(policy_net.parameters())


if __name__ == "__main__":

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    steps_done = 0
    last_optimization = 1000
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    metricsListners, weights = runSimulation()

    for metricListner in metricsListners:
        print(metricListner)
    plt.plot(loss_values)
    weights_file = open("weights.txt", 'w')
    weights_file.write(str(weights))
    plt.show()


