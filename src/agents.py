# agent that propose some constant speed specified in the constructor
import stable_baselines3


class FixedPolicyAgent:
    def __init__(self, speed: int, min_speed: int = 0):
        self._speed = speed
        self._min_speed = min_speed
        return

    def act(self, state) -> int:
        return self._speed - self._min_speed


# agents that propose no speed (None)
# required for testing env without policy intervention or with in-built sumo strategies (GLOSA)
class DummyAgent:
    def __init__(self):
        return

    def act(self, state) -> int:
        return None


class StableBaselinesAgent:
    def __init__(self, model):
        self._model: stable_baselines3.PPO = model

    def act(self, state) -> int:
        return self._model.predict(state, deterministic=True)[0]
