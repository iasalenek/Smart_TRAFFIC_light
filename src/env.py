import numpy as np

from typing import List, Dict, Optional, Set
import sumolib

import random

import traci
from src.metrics import MeanEdgeFuelConsumption, MeanEdgeTime
from src.metrics import EdgeMetric
from gymnasium import Env, spaces

# constants
_INF_PROX = 2000

# defaults
CONFIG_PATH = "nets/single_agent/500m/config.sumocfg"
SIM_TIME = 1 * 60 * 5
P_VEHICLE = 0.3
P_CONNECTED = 0.2
MIN_SPEED = 15
MAX_SPEED = 60
STEP_LENGTH = 1
EDGE_IDS = ["E1", "E0", "E2"]
TRAFFIC_LIGTS = ["J1", "J2"]
VEHICLETYPE_IDS = ["ordinary", "connected"]
RANDOM_SEED = 42
SUMO_SEED = 42
USE_GUI = True
GLOSA_RANGE = 0


class ConnectedVehicle:
    def __init__(self, veh_id: int, edge_id: str, fuel_cons: Dict[str, float]):
        self.veh_id = veh_id
        self.edge_id = edge_id
        self.fuel_cons = fuel_cons


class RoadSimulationEnv(Env):
    def __init__(self,
                 net: sumolib.net.Net,
                 config_path: str = CONFIG_PATH,
                 sim_time: int = SIM_TIME,
                 p_vehicle: float = P_VEHICLE,
                 p_connected: float = P_CONNECTED,
                 min_speed: int = MIN_SPEED,
                 max_speed: int = MAX_SPEED,
                 step_length: float = STEP_LENGTH,
                 edge_ids: List[str] = EDGE_IDS,
                 trafficlight_ids: Optional[List[str]] = TRAFFIC_LIGTS,
                 vehicletype_ids: Optional[List[str]] = VEHICLETYPE_IDS,
                 random_seed: int = RANDOM_SEED,
                 sumo_seed: int = SUMO_SEED,
                 use_gui: bool = USE_GUI,
                 glosa_range: int = GLOSA_RANGE,
                 ):
        super(RoadSimulationEnv, self).__init__()

        self.vehicletype_ids = vehicletype_ids
        self.trafficlight_ids = trafficlight_ids
        self.net = net
        self.edge_ids = edge_ids
        self.veh_id = 0
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.sim_time = sim_time
        self.step_length = step_length
        self.cur_step = 0
        self.p_vehicle = p_vehicle
        self.p_connected = p_connected
        self._actionable_vehicles: List[ConnectedVehicle] = []
        self._glosa_range = glosa_range

        self.action_space = spaces.Discrete(self.max_speed - self.min_speed)

        self.output_shape = (5, 1)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.float64)

        sumo_binary = sumolib.checkBinary("sumo-gui") if use_gui else sumolib.checkBinary("sumo")
        self.sumo_cmd = [
            sumo_binary,
            "-c", config_path,
            "--start",
            "--step-length", str(step_length),
            "--quit-on-end",
            "--seed", str(sumo_seed),
            "--time-to-teleport", "-1",  # Телепортация автомобилей отключена
            "--device.glosa.range", str(self._glosa_range),
        ]

        random.seed(random_seed)

        self.metrics_listeners = {}

        self._traci_setup()

    def _traci_setup(self):
        traci.start(self.sumo_cmd)

        for edgeID in self.edge_ids:
            self.metrics_listeners[edgeID] = {
                "time": MeanEdgeTime(edgeID=edgeID, vehicletypeIDs=self.vehicletype_ids),
                "fuel": MeanEdgeFuelConsumption(edgeID=edgeID, vehicletypeIDs=self.vehicletype_ids),
            }

        for edge_id in self.edge_ids:
            for (_, metric_listener) in self.metrics_listeners[edge_id].items():
                metric_listener: EdgeMetric
                traci.addStepListener(metric_listener)

    def reset(self, seed=42):
        observations, _ = self.reset_all(random_probabilities=True)
        return observations[0], {}

    def reset_all(self, random_probabilities=False):
        traci.close()

        if random_probabilities:
            self.p_vehicle = random.random()
            self.p_connected = random.random()

        self.cur_step = 0
        self._traci_setup()

        self._spawn_till_connected()

        self._actionable_vehicles.clear()
        for edge_id in self.edge_ids:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edge_id):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    self._actionable_vehicles.append(ConnectedVehicle(vehicleID, edge_id, {
                        edge_id: self._get_fuel_consumption(edge_id, vehicleID) for edge_id in self.edge_ids
                    }))

        assert len(self._actionable_vehicles) != 0

        random.shuffle(self._actionable_vehicles)

        obs = self._get_obs_all()
        return obs, {}

    def _get_obs_all(self) -> List[np.ndarray]:
        edge_ids: Set[str] = {veh.edge_id for veh in self._actionable_vehicles}

        lane_positions = {
            edge_id: {
                veh_id: traci.vehicle.getLanePosition(veh_id) for veh_id in traci.edge.getLastStepVehicleIDs(edge_id)
            } for edge_id in edge_ids
        }

        res = []
        for veh in self._actionable_vehicles:
            car_pos = traci.vehicle.getPosition(veh.veh_id)[0]
            closest_red_light_tf = _INF_PROX
            time_to_next_switch = 0
            for tf_id in self.trafficlight_ids:
                tf_pos = self.net.getNode(tf_id).getCoord()[0]
                if tf_pos >= car_pos and traci.trafficlight.getPhase(tf_id) == 0:
                    time_to_next_switch = traci.trafficlight.getNextSwitch(tf_id) - traci.simulation.getTime()
                    closest_red_light_tf = min(closest_red_light_tf, tf_pos - car_pos)

            acceleration = traci.vehicle.getAcceleration(veh.veh_id)

            vehicles_on_the_edge = traci.edge.getLastStepVehicleIDs(veh.edge_id)
            edge_vehicle_count = len(vehicles_on_the_edge)

            lane_pos: float = lane_positions[veh.edge_id][veh.veh_id]
            dist_to_next = _INF_PROX
            for other_veh_id, other_lane_pos in lane_positions[veh.edge_id].items():
                if veh.veh_id == other_veh_id or lane_pos >= other_lane_pos:
                    continue

                dist_to_next = min(dist_to_next, other_lane_pos - lane_pos)

            res.append(
                np.array(
                    [time_to_next_switch, closest_red_light_tf, acceleration, edge_vehicle_count, dist_to_next],
                ).reshape((5, 1)))
        return res

    def _get_fuel_consumption(self, edge_id: str, veh_id: int):
        if veh_id in self.metrics_listeners[edge_id]['fuel']._vehicleIdFuelDict:
            return self.metrics_listeners[edge_id]['fuel']._vehicleIdFuelDict[veh_id]
        return 0

    def _get_reward(self):
        return self._get_reward_all()[0]

    def _get_reward_all(self) -> List[float]:
        deltas: List[float] = []
        for veh in self._actionable_vehicles:
            cur_fuel_cons = {
                edge_id: self._get_fuel_consumption(edge_id, veh.veh_id) for edge_id in
                self.edge_ids
            }

            delta = 0
            for edge_id in self.edge_ids:
                prev, cur = veh.fuel_cons[edge_id], cur_fuel_cons[edge_id]

                if cur > prev:
                    delta += cur - prev

            assert delta >= 0

            deltas.append(-delta)

        return deltas

    def get_total_fuel(self):
        fuel_cons = {
            "connected": 0,
            "ordinary": 0,
            "All": 0,
        }
        for edge_id in self.edge_ids:
            cur = self.metrics_listeners[edge_id]["fuel"].get_mean_fuel()
            fuel_cons["connected"] += cur["connected"]
            fuel_cons["ordinary"] += cur["ordinary"]
            fuel_cons["All"] += cur["All"]

        fuel_cons["connected"] /= len(self.edge_ids)
        fuel_cons["ordinary"] /= len(self.edge_ids)
        fuel_cons["All"] /= len(self.edge_ids)
        return fuel_cons

    def get_total_time(self):
        time_cons = {
            "connected": 0,
            "ordinary": 0,
            "All": 0,
        }
        for edge_id in self.edge_ids:
            cur = self.metrics_listeners[edge_id]["time"].get_mean_time()
            time_cons["connected"] += cur["connected"]
            time_cons["ordinary"] += cur["ordinary"]
            time_cons["All"] += cur["All"]

        time_cons["connected"] /= len(self.edge_ids)
        time_cons["ordinary"] /= len(self.edge_ids)
        time_cons["All"] /= len(self.edge_ids)
        return time_cons

    def _spawn_till_connected(self):
        while True:
            if random.random() < self.p_vehicle * self.step_length:
                is_connected = random.random() < self.p_connected

                traci.vehicle.add(
                    vehID=self.veh_id, routeID="r_0", departLane="best",
                    typeID="connected" if is_connected else "ordinary"
                )

                traci.vehicle.setSpeed(
                    vehID=self.veh_id, speed=random.randint(self.min_speed, self.max_speed) / 3.6
                )

                self.veh_id += 1

            traci.simulationStep()

            found = False
            for edge_id in self.edge_ids:
                for vehicleID in traci.edge.getLastStepVehicleIDs(edge_id):
                    if traci.vehicle.getTypeID(vehicleID) == "connected":
                        found = True
            if found:
                return

    def step(self, action: int):
        observations, rewards, terminated, truncated, _ = self.step_all([action])
        return observations[0], rewards[0], terminated, truncated, {}

    def step_all(self, actions: List[np.ndarray]):
        for i, action in enumerate(actions):
            if not action:
                continue
            action += self.min_speed
            action = min(action, self.max_speed)
            action = max(action, self.min_speed)
            assert (
                    traci.vehicle.getTypeID(self._actionable_vehicles[i].veh_id) == "connected"
            ), f"vehicle {self._actionable_vehicles[i].veh_id} is not connected"
            assert action <= MAX_SPEED
            traci.vehicle.setSpeed(vehID=self._actionable_vehicles[i].veh_id, speed=action / 3.6)

        self._spawn_till_connected()

        rewards = self._get_reward_all()

        self._actionable_vehicles.clear()
        for edge_id in self.edge_ids:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edge_id):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    self._actionable_vehicles.append(ConnectedVehicle(vehicleID, edge_id, {
                        edge_id: self._get_fuel_consumption(edge_id, vehicleID) for edge_id in self.edge_ids
                    }))

        random.shuffle(self._actionable_vehicles)

        observations = self._get_obs_all()

        assert len(observations) != 0

        terminated = self.cur_step + 1 > (self.sim_time // self.step_length)
        truncated = False

        self.cur_step += 1

        return observations, rewards, terminated, truncated, {}

    def close(self):
        traci.close()
        super().close()
