import traci
from traci import StepListener

from src.utils import Distribution


class TrafficGenerator(StepListener):
    def __init__(
        self,
        routeID: str,
        typeID: str,
        deltaT: float,
        wavesFrequency: float,
        wavesAmplitudeDistribution: Distribution,
        randomCarsDistribution: Distribution,
    ) -> None:
        super().__init__()
        self.routeID = routeID
        self.typeID = typeID
        self.deltaT = deltaT

        self.wavesFrequency = wavesFrequency
        self.wavesAmplitudeDistribution = wavesAmplitudeDistribution
        self.randomCarsDistribution = randomCarsDistribution
        self.secondScinceLastWave = 0
        self.generatedCars = 0

    def step(self, t=0):
        self.secondScinceLastWave += self.deltaT
        carsToGenerate = self.randomCarsDistribution.sample()
        if self.secondScinceLastWave >= self.wavesFrequency:
            self.secondScinceLastWave = 0
            carsToGenerate += self.wavesAmplitudeDistribution.sample()
        for i in range(carsToGenerate):
            traci.vehicle.add(
                vehID=self.generatedCars + i,
                routeID=self.routeID,
                departLane="best",
                typeID=self.typeID,
            )
        self.generatedCars += carsToGenerate
        return super().step(t)

    def cleanUp(self):
        self.wavesAmplitudeDistribution.reset()
        self.randomCarsDistribution.reset()
        self.secondScinceLastWave = 0
        self.generatedCars = 0
        return super().cleanUp()
