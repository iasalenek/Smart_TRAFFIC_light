from typing import List, Optional

import os
import random
import sys

import sumolib
import traci

from src.policy import BasePolicy, FixedSpeedPolicy
from src.metrics import MeanEdgeFuelConsumption, MeanEdgesFuelConsumption


CONFIG_PATH = "nets/single_agent/Hamburg/run_simulation.sumocfg"
SIM_TIME = 1 * 60 * 60

STEP_LENGTH = 1
EDGE_IDS = ["E1"]
TRAFFIC_LIGTS = ["J2"]
MIN_EDGE_LENGTH = 200
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
    stepLength: float = STEP_LENGTH,
    edgeIDs: List[str] = EDGE_IDS,
    trafficlightIDs: Optional[List[str]] = None,
    vehicletypeIDs: Optional[List[str]] = VEHICLETYPE_IDS,
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

    if policyListner is not None:
        policyListner = policyListner(
            edgeIDs=edgeIDs,
            trafficlightIDs=trafficlightIDs,
            **policyOptions,
        )

    metricsListners = []
    metricsListners += [
        MeanEdgesFuelConsumption(edgeIDs=edgeIDs, vehicletypeIDs=vehicletypeIDs),
    ]

    traci.addStepListener(policyListner)
    for metricListner in metricsListners:
        traci.addStepListener(metricListner)

    for _ in range(simTime // STEP_LENGTH):
        traci.simulationStep()

    traci.close()

    return metricsListners


def getEdgesAndTrafficLights(
    configPath: str = CONFIG_PATH,
    minEdgeLength: int = MIN_EDGE_LENGTH,
):
    sumoCmd = [
        sumolib.checkBinary('sumo'),
        '-c', configPath,
        '--start',
    ]
    traci.start(sumoCmd)

    edgeIDs = set()

    trafficlightIDs = traci.trafficlight.getIDList()
    trafficlightEdgesDict = dict()
    for trafficlightID in trafficlightIDs:
        incEdges = set(
            link[0][0].split("_")[0]
            for link in traci.trafficlight.getControlledLinks(trafficlightID)
            if link
        )
        trafficlightEdgesDict[trafficlightID] = incEdges
        edgeIDs = edgeIDs.union(incEdges)

    edgeIDsFiltered = sorted(
        edgeID
        for edgeID in edgeIDs
        if traci.lane.getLength(f"{edgeID}_0") > minEdgeLength
    )
    trafficlightIDsFiltered = sorted(
        tlID
        for tlID, edgeIDs in trafficlightEdgesDict.items()
        if edgeIDs.intersection(edgeIDsFiltered)
    )

    traci.close()

    return edgeIDsFiltered, trafficlightIDsFiltered


if __name__ == "__main__":

    edgeIDs, trafficlightIDs = getEdgesAndTrafficLights(
        configPath=CONFIG_PATH,
        minEdgeLength=MIN_EDGE_LENGTH,
    )

    metricsListners = runSimulation(
        edgeIDs=edgeIDs,
        trafficlightIDs=trafficlightIDs,
        policyListner=FixedSpeedPolicy,
        policyOptions={'speed': 30}
    )

    for metricListner in metricsListners:
        print(metricListner)
