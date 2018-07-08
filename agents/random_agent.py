import numpy as np
import random
import ipdb

class RandomAgent():
    def __init__(self, action_list=None):
        self.action_list = action_list  # 選択肢

    def act(self, q_values=None):
        action_id = random.randint(0, (len(self.action_list)-1))
        action = self.action_list[action_id]
        return action

    def get_reward(self, reward):
        pass
