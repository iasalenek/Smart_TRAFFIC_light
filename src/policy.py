from typing import List, Optional

import traci
from traci import StepListener

POLICY_MIN_SPEED = 30
POLICY_MAX_SPEED = 60


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
        extraEdges = set(self.edgeIDs) - set(traci.edge.getIDList())
        assert not extraEdges, f"Нет ребер {extraEdges}"
        # Проверяем наличие всех светофоров
        if trafficlightIDs is not None:
            extraTrafficlights = set(self.trafficlightIDs) - set(traci.trafficlight.getIDList())
            assert not extraTrafficlights, f"Нет светофоров {extraTrafficlights}"

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
