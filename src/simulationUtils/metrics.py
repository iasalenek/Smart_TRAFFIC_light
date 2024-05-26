from typing import Iterable, Optional

import numpy as np
import traci
import traci.constants as tc
from traci import StepListener
from .policyTraffic import *


class EdgeMetric(StepListener):
    def __init__(
        self,
        edgeID: str,
        vehicletypeIDs: Optional[Iterable[str]] = None,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_model: trainTraffic = model
        self.metricName = None
        # Проверяем наличие ребра
        assert edgeID in traci.edge.getIDList(), f"Нет ребра с ID {edgeID}"
        self.edgeID = edgeID
        # Проверяем ниличие всех типов транспорта
        if vehicletypeIDs is not None:
            self.vehicletypeIDs = set(vehicletypeIDs)
            extraTypes = self.vehicletypeIDs - \
                set(traci.vehicletype.getIDList())
            assert not extraTypes, f"Нет транспорта с типами {extraTypes}"
        else:
            self.vehicletypeIDs = traci.vehicletype.getIDList()
        # Dict-ы с неаггрегированными и аггрегированными значениями по типам
        # автомобилей
        self._nonAggregatedValues = {vehicletype: []
                                     for vehicletype in vehicletypeIDs}
        self._nonAggregatedValues["All"] = []
        self.aggregatedValues = {
            vehicletype: None for vehicletype in self._nonAggregatedValues.keys()}

    def cleanUp(self):
        self.aggregatedValues = {
            vehicletype: np.mean(values)
            for vehicletype, values in self._nonAggregatedValues.items()
        }

    def __repr__(self) -> str:
        repr = f"{self.metricName}:\n"
        for vehicletype, value in self.aggregatedValues.items():
            repr += f"{vehicletype[:10]:<10} --- {value:>10.4f}\n"
        return repr


class MeanEdgeTime(EdgeMetric):
    def __init__(
        self,
        edgeID: str,
        vehicletypeIDs: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        super(MeanEdgeTime, self).__init__(edgeID, vehicletypeIDs, **kwargs)
        self.metricName = f"Mean Time for Edge {self.edgeID}"
        self._vehicleIdTimeDict = dict()
        self._vehicleIdTypeDict = dict()
        traci.edge.subscribe(
            self.edgeID, varIDs=[
                tc.LAST_STEP_VEHICLE_ID_LIST])

    def step(self, t=0):
        time = traci.simulation.getTime()
        # Множества всех машин на ребре и машин которые въехали/выехали с ребра
        # на этом шаге
        stepVehicleIDs = set(
            traci.edge.getSubscriptionResults(
                self.edgeID)[
                tc.LAST_STEP_VEHICLE_ID_LIST])
        departuredVehicleIDs = stepVehicleIDs - self._vehicleIdTimeDict.keys()
        arrivedVehicleIDs = self._vehicleIdTimeDict.keys() - stepVehicleIDs
        # Убираем уехавшие машины из рассмотрения и записываем их метрики
        for vehicleID in arrivedVehicleIDs:
            vehicletypeID = self._vehicleIdTypeDict.pop(vehicleID)
            edgeTime = time - self._vehicleIdTimeDict.pop(vehicleID)
            if vehicletypeID in self.vehicletypeIDs:
                self._nonAggregatedValues[vehicletypeID].append(edgeTime)
                self._nonAggregatedValues["All"].append(edgeTime)
        # Добавляем въехавшие автомобили и их типы
        for vehicleID in departuredVehicleIDs:
            self._vehicleIdTimeDict[vehicleID] = time
            self._vehicleIdTypeDict[vehicleID] = traci.vehicle.getTypeID(
                vehicleID)
        return super().step(t)


class MeanEdgeFuelConsumption(EdgeMetric):
    def __init__(
        self,
        edgeID: str,
        vehicletypeIDs: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        super(MeanEdgeFuelConsumption, self).__init__(edgeID, vehicletypeIDs, **kwargs)
        self.metricName = f"Mean Fuel Consumption for Edge {self.edgeID}"
        self._vehicleIdFuelDict = dict()
        self._vehicleIdTypeDict = dict()
        self.stepLength = traci.simulation.getDeltaT()
        traci.edge.subscribe(self.edgeID, varIDs=[tc.LAST_STEP_VEHICLE_ID_LIST])

    def step(self, t=0):
        # Множества всех машин на ребре и машин которые въехали/выехали с ребра на этом шаге
        stepVehicleIDs = set(
            traci.edge.getSubscriptionResults(self.edgeID)[tc.LAST_STEP_VEHICLE_ID_LIST]
        )
        departuredVehicleIDs = stepVehicleIDs - self._vehicleIdFuelDict.keys()
        arrivedVehicleIDs = self._vehicleIdFuelDict.keys() - stepVehicleIDs
        # Убираем уехавшие машины из рассмотрения и записываем их метрики
        for vehicleID in arrivedVehicleIDs:
            vehicletypeID = self._vehicleIdTypeDict.pop(vehicleID)
            fuelConsumption = self._vehicleIdFuelDict.pop(vehicleID)
            if vehicletypeID in self.vehicletypeIDs:
                self._nonAggregatedValues[vehicletypeID].append(fuelConsumption)
                self._nonAggregatedValues["All"].append(fuelConsumption)
        # Добавляем въехавшие автомобили, их типы и делаем новый subscription
        for vehicleID in departuredVehicleIDs:
            self._vehicleIdFuelDict[vehicleID] = 0
            self._vehicleIdTypeDict[vehicleID] = traci.vehicle.getTypeID(vehicleID)
            traci.vehicle.subscribe(vehicleID, varIDs=[tc.VAR_FUELCONSUMPTION])
        # Для всех автомобилей на ребре обновляем метрику
        for vehicleID in self._vehicleIdFuelDict:
            self._vehicleIdFuelDict[vehicleID] += (
                traci.vehicle.getSubscriptionResults(vehicleID)[tc.VAR_FUELCONSUMPTION]
                * self.stepLength
            )
        return super().step(t)
