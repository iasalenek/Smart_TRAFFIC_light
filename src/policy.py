from typing import List, Optional
import random
import traci
from traci import StepListener
from src.policyTraffic import *
import numpy as np

POLICY_MIN_SPEED = 30
POLICY_MAX_SPEED = 60


class BasePolicy(StepListener):
    def __init__(
        self,
        edgeIDs: List[str],
        trafficlightIDs: Optional[List[str]] = None,
        model=None,
        **kwargs,
    ) -> None:
        super(BasePolicy, self).__init__(**kwargs)

        self.train_model: trainTraffic = model

        self.edgeIDs = edgeIDs
        self.trafficlightIDs = trafficlightIDs
        # Проверяем наличие всех ребер
        assert set(self.edgeIDs).issubset(traci.edge.getIDList())
        # Проверяем наличие всех светофоров
        if trafficlightIDs is not None:
            assert set(
                self.trafficlightIDs).issubset(
                traci.trafficlight.getIDList())

    def step(self, t=0):
        return super().step(t)

    @staticmethod
    def apply_action(vehicleID: str, speed: float):
        assert (
            traci.vehicle.getTypeID(vehicleID) == "connected"
        ), f"vehicle {vehicleID} is not connected"
        assert (speed >= POLICY_MIN_SPEED) and (
            speed <= POLICY_MAX_SPEED
        ), "The speed is beyond the limit"
        traci.vehicle.setSpeed(vehID=vehicleID, speed=speed / 3.6)


class NeuroPolicy(BasePolicy):
    def __init__(
        self,
        speed: int,
        edgeIDs: List[str],
        trafficlightIDs: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super(NeuroPolicy, self).__init__(edgeIDs, trafficlightIDs, **kwargs)
        assert (speed >= POLICY_MIN_SPEED) and (speed <= POLICY_MAX_SPEED)
        self.speed = speed

    def apply_action(self):
        if self.train_model.get_id() is None:
            return
        for edgeID in self.edgeIDs:
            for id, vehicleID in enumerate(self.train_model.get_id()):
                if vehicleID != - \
                        1 and vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                    if traci.vehicle.getTypeID(vehicleID) == "connected":
                        super().apply_action(
                            vehicleID, self.train_model.get_speed()[id])

    def step(self, t=0):

        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edgeIDs:

            all_venicles = [
                i for i in traci.edge.getLastStepVehicleIDs(edgeID)]
            connected_venicles = [
                i for i in all_venicles if traci.vehicle.getTypeID(i) == "connected"]

            inds = [i for i in connected_venicles]
            if len(inds) > NUMBER_OF_CARS:
                inds = np.random.choice(inds, NUMBER_OF_CARS, replace=False)

            def init_vector(vec, d, EXTRA_MIN=0):
                while len(vec) < d:
                    vec.append(EXTRA_MIN)
                return vec[:d]

            if len(inds) != 0:
                # add vector phace
                id_random = np.random.choice(inds, 1)[0]
                tlss = traci.vehicle.getNextTLS(id_random)
                phace = [0 for _ in range(NUMBER_OF_CARS)]
                tlsId = tlss[0][0]
                val = traci.trafficlight.getPhase(tlsId)
                phace[val] = 1
            else:
                phace = [0 for _ in range(NUMBER_OF_CARS)]

            deltaDist = 1000
            deltaSpid = 100
            connected_dist_vector = [
                (traci.vehicle.getDistance(i) + deltaDist) / 2000 for i in inds]
            connected_speed_vector = [
                (traci.vehicle.getSpeed(i) + deltaSpid) / 160 for i in inds]
            connected_dist_light = [
                (abs(
                    traci.vehicle.getDistance(i) -
                    sorted(
                        traci.vehicle.getNextTLS(i),
                        key=lambda x: x[1])[0][2]) +
                    deltaDist) /
                2000 for i in inds]

            connected_ids = init_vector(inds[:], NUMBER_OF_CARS, -1)
            connected_dist_vector = init_vector(
                connected_dist_vector, NUMBER_OF_CARS)
            connected_speed_vector = init_vector(
                connected_speed_vector, NUMBER_OF_CARS)
            connected_dist_light = init_vector(
                connected_dist_light, NUMBER_OF_CARS, 0)

            result_state = [connected_dist_vector,
                            connected_speed_vector,
                            connected_dist_light,
                            phace]
            result_state = np.array(result_state).reshape(-1)
            self.train_model.set_state(result_state)
            self.train_model.set_id(connected_ids)

        return super().step(t)
