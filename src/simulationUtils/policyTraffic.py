import queue
import numpy as np

NUMBER_OF_CARS = 5
NUMBER_OF_STATES_BY_CAR = 5


class trainTraffic:

    @staticmethod
    def init_actions():
        speed_dct = {}
        ind = 0

        def action(vec, ln=0):
            if ln == NUMBER_OF_CARS:
                nonlocal ind
                speed_dct[ind] = vec
                ind += 1
                return
            for i in range(NUMBER_OF_STATES_BY_CAR):
                new_vec = vec + ((60 - 30) / NUMBER_OF_STATES_BY_CAR * i + 30,)
                action(new_vec, ln + 1)
        action(tuple())
        return speed_dct

    def __init__(self, simultation, speed_dct=None):
        self.actual_action = None
        self.actual_state = None
        self.reward = None
        self.action = 0
        self.lastId = set()
        self.speed_dct = {}

        self.simulation_ = simultation
        if speed_dct is None:
            self.init_actions()
        else:
            self.speed_dct = speed_dct

    def get_speed_dct(self):
        return self.speed_dct

    def get_speed(self):
        return self.speed_dct[self.action]

    def get_id(self):
        return self.lastId

    def set_id(self, id):
        self.lastId = id

    def set_state(self, state):
        self.actual_state = state

    def set_reward(self, reward):
        self.reward = reward

    def get_action(self):
        return self.action

    def step_while(self):
        self.simulation_.step()

    def step(self, ak):
        self.action = ak
        self.step_while()
        ret_value = (self.actual_state, self.reward)
        self.reward = None
        self.actual_state = None
        return ret_value

    def close(self):
        self.simulation_.close()
