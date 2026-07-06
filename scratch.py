from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class ModelOutputs:
    H_T: float
    H_P: float
    Q_T: float
    Q_P: float


@dataclass
class ModelInputs:
    y_T: float
    N_T: float
    N_P: float


@dataclass
class ModelState:
    t: float
    model_outputs: ModelOutputs
    model_inputs: ModelInputs


class OtherProtocol(Protocol):
    def update(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs: ...


class PlantProtocol(Protocol):
    def setp(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs: ...
    def reset(self) -> None: ...


class FMUPlant:
    def setp(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        return ModelOutputs(0, 0, 0, 0)

    def update(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        return ModelOutputs(0, 0, 0, 0)

    def reset(self) -> None:
        pass


def simulate(plant: PlantProtocol) -> None:
    plant.reset()
    pass


def other_fn(plant: OtherProtocol) -> None:
    plant.update(0, 1, ModelInputs(0, 0, 0))
    pass


if __name__ == "__main__":
    fmu_plant = FMUPlant()
    simulate(fmu_plant)
    other_fn(fmu_plant)
