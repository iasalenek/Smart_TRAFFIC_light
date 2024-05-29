from typing import List, Optional

import os
import sys

import sumolib
import traci

from src.policy import BasePolicy
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime
from src.traffic import TrafficGenerator
from src.utils import PositiveNormal, Exponential


CONFIG_PATH = "nets/single_agent/1000m/config.sumocfg"
SIM_TIME = 1 * 60 * 60
P_VEHICLE = 0.3
P_CONNECTED = 0.3
MIN_SPEED = 45
MAX_SPEED = 60

STEP_LENGTH = 1
EDGE_IDS = ["E1"]
TRAFFIC_LIGTS = ["1"]
VEHICLETYPE_IDS = ["ordinary", "connected"]
RANDOM_SEED = 42
SUMO_SEED = 42
USE_GUI = True

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def runSimulation(
    configPath: str = CONFIG_PATH,
    simTime: int = SIM_TIME,
    policyListner: BasePolicy = BasePolicy,
    policyOptions: dict = {},
    trafficListners: list[TrafficGenerator] | None = None,
    stepLength: float = STEP_LENGTH,
    edgeIDs: List[str] = EDGE_IDS,
    trafficlightIDs: Optional[List[str]] = None,
    vehicletypeIDs: Optional[List[str]] = VEHICLETYPE_IDS,
    useGUI: bool = USE_GUI,
    sumoSeed: int = SUMO_SEED,
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

    if policyListner is not None:
        policyListner = policyListner(
            edgeIDs=edgeIDs,
            trafficlightIDs=trafficlightIDs,
            **policyOptions,
        )

    metricsListners = []
    for edgeID in edgeIDs:
        metricsListners += [
            MeanEdgeTime(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
            MeanEdgeFuelConsumption(edgeID=edgeID, vehicletypeIDs=vehicletypeIDs),
        ]

    traci.addStepListener(policyListner)
    for metricListner in metricsListners:
        traci.addStepListener(metricListner)

    if trafficListners is not None:
        for trafficListner in trafficListners:
            traci.addStepListener(trafficListner)

    for _ in range(simTime // STEP_LENGTH):
        traci.simulationStep()

    traci.close()

    return metricsListners


if __name__ == "__main__":

    trafficListner = TrafficGenerator(
        routeID='r_0',
        typeID='PKW_special',
        deltaT=STEP_LENGTH,
        wavesFrequency=40,
        wavesAmplitudeDistribution=PositiveNormal(10, 10, seed=RANDOM_SEED),
        randomCarsDistribution=Exponential(_lambda=0.3, seed=RANDOM_SEED),
    )

    metricsListners = runSimulation(
        trafficListners=[trafficListner],
    )

    for metricListner in metricsListners:
        print(metricListner)
