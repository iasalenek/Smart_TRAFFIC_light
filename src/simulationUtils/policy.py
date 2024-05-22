from typing import List, Optional
import random
import traci
from traci import StepListener
from .policyTraffic import *
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

class FixedSpeedPolicy(BasePolicy):
    def __init__(
        self,
        speed: int,
        edgeIDs: List[str],
        trafficlightIDs: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super(FixedSpeedPolicy, self).__init__(edgeIDs, trafficlightIDs, **kwargs)
        assert (speed >= POLICY_MIN_SPEED) and (speed <= POLICY_MAX_SPEED)
        self.speed = speed

    def step(self, t=0):
        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edgeIDs:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    self.apply_action(vehicleID, self.speed)
        return super().step(t)

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
        for edgeID in self.edgeIDs:
            for id in self.train_model.use_real_ids:
                if id in traci.edge.getLastStepVehicleIDs(edgeID):
                     if traci.vehicle.getTypeID(id) == "connected":
                        model_id = self.train_model.to_id(id)
                        super().apply_action(id, traci.vehicle.getSpeed(id) * 3.6 + self.train_model.get_speed_diff(model_id))

    def step(self, t=0):

        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edgeIDs:

            all_venicles = [
                i for i in traci.edge.getLastStepVehicleIDs(edgeID)]
            connected_venicles = [
                i for i in all_venicles if traci.vehicle.getTypeID(i) == "connected"]


            self.train_model.set_and_clear_ids(connected_venicles)

            for id_venicle in connected_venicles:
                sorted_tlss =  sorted(traci.vehicle.getNextTLS(id_venicle), key=lambda x: x[2])
                phace = [0 for _ in range(5)]
                val = traci.trafficlight.getPhase(sorted_tlss[0][0])
                phace[val] = 1
                dist_to_tlss = abs(traci.vehicle.getDistance(id_venicle) - sorted_tlss[0][2])
                speed = traci.vehicle.getSpeed(id_venicle) * 3.6
                remaining_time = traci.trafficlight.getNextSwitch(sorted_tlss[0][0])

                result_vec = np.array([traci.vehicle.getDistance(id_venicle)]
                + [speed]
                + [dist_to_tlss]  + phace + [remaining_time] + [1.0],dtype=np.float64)
                
                self.train_model.set_obs_agent(self.train_model.to_id(id_venicle), result_vec)

        return super().step(t)
