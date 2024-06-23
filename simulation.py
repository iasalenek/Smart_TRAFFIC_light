from typing import List, Optional

import os
import random
import sys

import sumolib
import traci

from src.policy import BasePolicy, FixedSpeedPolicy
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime


CONFIG_PATH = "nets/single_agent/500m/config.sumocfg"
SIM_TIME = 1 * 60 * 60
P_VEHICLE = 0.3
P_CONNECTED = 0.2
MIN_SPEED = 45
MAX_SPEED = 60

STEP_LENGTH = 1
EDGE_IDS = ["E1"]
TRAFFIC_LIGTS = ["J2"]
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
    GLOSARange: int = 0,
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
        "--device.glosa.range", str(GLOSARange),
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

    veh_id = 0
    random.seed(randomSeed)

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

    traci.close()

    return metricsListners


if __name__ == "__main__":

    metricsListners = runSimulation(
        policyListner=FixedSpeedPolicy,
        policyOptions={'speed': 30}
    )

    for metricListner in metricsListners:
        print(metricListner)
