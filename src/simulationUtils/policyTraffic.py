import queue
import numpy as np
import copy
from .simulation import *

NUMBER_OF_CARS = 1
NUMBER_OF_STATES_BY_CAR = 60
DEFAULT_STATE = (0,0,0,0,0,0,0,0,0,0)


class Buffer:

    def __init__(self, copyTuple):
        self.copyTuple = copyTuple
    def getValue(self):
        return self.copyTuple

class trainTraffic:

    def __init__(self, simultation):
        self.actual_action = None
        self.actual_state = None
        self.reward = None
        self.action = {}
        self.lastId = set()
        self.speed_dct = {}

        self.n_agent = NUMBER_OF_CARS
        self.use_ids = set()
        self.use_real_ids = set()

        
        self.buffer = None

        self.to_id_dict = {}
        self.from_id_dict = {}
        self.id_to_state = {}
        self.agent_obs = {}

        self.real_id_null_stateble = set()

        self.time = 0

        self.simulation_ = simultation
        self.speed_dct = {}

        for i in range(NUMBER_OF_STATES_BY_CAR - 1):
            self.speed_dct[i + 1] = 60 / (NUMBER_OF_STATES_BY_CAR - 2) * i + -30
        self.speed_dct[0] = 0
        # print(self.speed_dct[i + 1])

    def set_reward(self, reward):
        self.reward = reward

    def step_while(self):
        self.simulation_.step()

    def to_id(self, id):
        if id not in self.use_real_ids:
            return None
        return self.to_id_dict[id]

    def from_id(self, id):
        if id not in self.use_ids:
            return None
        return self.from_id_dict[id]

    def generate_id(self):
        for i in range(NUMBER_OF_CARS):
            if i not in self.use_ids:
                return i
        return None

    def get_buffer_previus(self):
        return self.buffer

    def set_and_clear_ids(self, connected):
        self.buffer = Buffer(
            (self.use_real_ids.copy(), 
            self.use_ids.copy(),
            copy.deepcopy(self.to_id_dict),
            copy.deepcopy(self.from_id_dict))
            )

        connected = set(connected)
        # чистит старые и устанавливает новые ids
        new_keys =  set(self.to_id_dict.keys()).intersection(connected)
        self.use_ids = {self.to_id_dict[i] for i in new_keys}
        self.use_real_ids = new_keys

        # добавляем пока можем
        for i in connected:
            if i in self.use_real_ids:
                continue
            new_id = self.generate_id()
            if new_id is None:
                break
            self.to_id_dict[i] = new_id
            self.from_id_dict[new_id] = i

            self.use_ids.add(new_id)
            self.use_real_ids.add(i)

    def get_real_id_null_stateble(self):
        return self.real_id_null_stateble

    def set_obs_agent(self, agent_id, state):
        if agent_id not in self.use_ids:
            return
        self.agent_obs[agent_id] = state

    def get_obs_agent(self, agent_id):
        if agent_id not in self.use_ids:
            return DEFAULT_STATE
        
        if self.buffer is None:
            return DEFAULT_STATE

        real_id_now = self.from_id(agent_id)

        if real_id_now not in self.buffer.getValue()[0]:
            return DEFAULT_STATE

        if self.buffer is not None:
            if agent_id in self.buffer.getValue()[1]:
                real_id_previos = self.buffer.getValue()[3][agent_id]
                if real_id_now != real_id_previos:
                    print("FFF")
                    return DEFAULT_STATE

        return self.agent_obs[agent_id]
    
    def is_good(self, speed):
        return speed >= 30 and speed <= 60

    def get_avail_agent_actions(self, agent_id):
        if agent_id not in self.use_ids:
            mask_action = [0 for _ in range(NUMBER_OF_STATES_BY_CAR)]
            mask_action[0] = 1
            # print(mask_action,0,self.speed_dct)
            return mask_action

        # state_ = self.get_obs_agent(agent_id)
        speed = traci.vehicle.getSpeed(self.from_id_dict[agent_id]) * 3.6
        # print(speed)
        mask_action = [self.is_good(speed + self.speed_dct[i]) for i in range(NUMBER_OF_STATES_BY_CAR)]
        # print(mask_action,speed,self.speed_dct)
        return mask_action

   

    def get_speed_diff(self, agent_id):
        if agent_id not in self.use_ids:
            return None
        return self.speed_dct[self.action[agent_id]]

    def step(self, ak):
        self.time += 1
        self.action = ak
        self.agent_obs.clear()
        self.step_while()
        if self.reward is None:
            raise RuntimeError
        ret_value = self.reward
        self.reward = None
        return -ret_value/1000000, (self.time > 1000)

    def close(self):
        self.time = 0
        self.simulation_.close()
    
    def reset(self):
        self.time = 0
        self.simulation_.reset()

    def init(self):
        self.time = 0
        self.simulation_.init()
