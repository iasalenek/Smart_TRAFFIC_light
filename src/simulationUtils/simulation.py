from typing import List, Optional

import os
import random
import sys

import threading
import sumolib
import traci
from .policyTraffic import trainTraffic
from .policy import BasePolicy, NeuroPolicy
from .metrics import MeanEdgeFuelConsumption, MeanEdgeTime


CONFIG_PATH = "nets/single_agent/500m/config.sumocfg"
# SIM_TIME = 1 * 60 * 60
SIM_TIME = 1 * 60 * 60 * 60
P_VEHICLE = 0.3
P_CONNECTED = 0.3
MIN_SPEED = 45
MAX_SPEED = 60

STEP_LENGTH = 1
EDGE_IDS = ["E1"]
TRAFFIC_LIGTS = ["J2"]
VEHICLETYPE_IDS = ["ordinary", "connected"]
RANDOM_SEED = 42
SUMO_SEED = 42
USE_GUI = True

# lck = lock_guard()
# model = trainTraffic(lck)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SimulationTraffic:

    def __init__(self):
        self.veh_id = 0
        self.traci = None
        self.minSpeed = None
        self.maxSpeed = None
        self.pVehicle = None
        self.stepLength = None
        self.pConnected = None
        self.trainTraffic_ = None
        self.metricListner_ = None
        self.policy = None

    def setTrainTraffic(self, trainTraffic_):
        self.trainTraffic_ = trainTraffic_

    def step(self):
        self.applyPolicy()
        if random.random() < self.pVehicle * self.stepLength:

            if random.random() < self.pConnected:
                traci.vehicle.add(
                    vehID=self.veh_id,
                    routeID="r_0",
                    departLane="best",
                    typeID="connected")
            else:
                traci.vehicle.add(
                    vehID=self.veh_id,
                    routeID="r_0",
                    departLane="best",
                    typeID="ordinary")

            traci.vehicle.setSpeed(
                vehID=self.veh_id, speed=random.randint(
                    self.minSpeed, self.maxSpeed) / 3.6)
            self.veh_id += 1

        traci.simulationStep()
        self.applyReward()
        self.setState()

    def applyPolicy(self):
        self.policy.apply_action()

    def setState(self):
        self.policy.step()

    def applyReward(self):
        sum_reward = 0.0
        sz = 0
        ls = []
        for i in self.trainTraffic_.actual_cars():
            try:
                # sum_reward += traci.vehicle.getFuelConsumption(i) * self.stepLength
                ls.append(traci.vehicle.getFuelConsumption(i) * self.stepLength)
                sz += 1
            except:
                sum_reward += 0.0
        # if sz == 0:
        #     self.trainTraffic_.set_reward(0)
        # else:
        #     self.trainTraffic_.set_reward(sum_reward/sz)
        if len(ls) == 0:
            sum_reward = 0
        else:
            sum_reward = max(ls)
            # print(ls)
        self.trainTraffic_.set_reward(sum_reward)
        for metricListner in self.metricListner_:
            metricListner.step()

    def reset(self):
        self.veh_id = 0

    def start(
        self,
        configPath: str = CONFIG_PATH,
        simTime: int = SIM_TIME,
        policyListner: BasePolicy = BasePolicy,
        policyOptions: dict = {},
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
        # init
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.pVehicle = pVehicle
        self.stepLength = stepLength
        self.pConnected = pConnected

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

        if policyListner is not None:
            policyListner = policyListner(
                edgeIDs=edgeIDs,
                trafficlightIDs=trafficlightIDs,
                model=self.trainTraffic_,
                **policyOptions,
            )

        self.metricListner_ = []
        for edgeID in edgeIDs:
            self.metricListner_ += [
                MeanEdgeTime(
                    edgeID=edgeID,
                    vehicletypeIDs=vehicletypeIDs,
                    model=self.trainTraffic_),
                MeanEdgeFuelConsumption(
                    edgeID=edgeID,
                    vehicletypeIDs=vehicletypeIDs,
                    model=self.trainTraffic_),
            ]

    #     metricsListners = []
    # for edgeID in edgeIDs:
    #     metricsListners += [
    #         MeanEdgeTime(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
    #         MeanEdgeFuelConsumption(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
    #     ]


        for metricListner in self.metricListner_:
            traci.addStepListener(metricListner)

        self.policy = policyListner
        random.seed(randomSeed)
        return True

    def init(self):
        tf = trainTraffic(self)
        self.setTrainTraffic(tf)
        self.start(
            policyListner=NeuroPolicy,
            policyOptions={'speed': 30}
        )
        tf.step_while()
    
    def reset(self):
        self.close()
        self = SimulationTraffic()
        self.init()

    def close(self):
        for i in self.metricListner_:
            i.cleanUp()
        print("--->", self.metricListner_)
        traci.close()

        return self.metricListner_


# all_actions = trainTraffic.init_actions()


def initSimulation() -> trainTraffic:
    cm = SimulationTraffic()
    tf = trainTraffic(cm)
    cm.setTrainTraffic(tf)

    cm.start(
        policyListner=NeuroPolicy,
        policyOptions={'speed': 30}
    )

    tf.step_while()
    return tf


if __name__ == "__main__":
    env = SimulationTraffic()
    env.init()
    env.step(0)
    env.step(0)
    env.step(0)
    env.reset()

    # tf = initSimulation()
    # for _ in range(SIM_TIME // STEP_LENGTH):
    #     tf.step(0)
    # tf.close()
