from typing import Optional, List
import traci
from traci import StepListener


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
        assert set(self.edgeIDs).issubset(traci.edge.getIDList())
        # Проверяем наличие всех светофоров
        if trafficlightIDs is not None:
            assert set(self.trafficlightIDs).issubset(traci.trafficlight.getIDList())

    def step(self, t=0):
        return super().step(t)


class MaxSpeedPolicy(BasePolicy):
    def __init__(
        self,
        edgeIDs: List[str],
        trafficlightIDs: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super(MaxSpeedPolicy, self).__init__(edgeIDs, trafficlightIDs, **kwargs)

    def step(self, t=0):
        # Пример политики, когда всем рекомендуется скорость 60 км/ч
        for edgeID in self.edgeIDs:
            for vehicleID in traci.edge.getLastStepVehicleIDs(edgeID):
                if traci.vehicle.getTypeID(vehicleID) == "connected":
                    traci.vehicle.setSpeed(vehicleID, speed=60 / 3.6)
        return super().step(t)
