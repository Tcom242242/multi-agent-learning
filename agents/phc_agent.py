from abc import ABCMeta, abstractmethod
import numpy as np
import ipdb

class Agent(metaclass=ABCMeta):
    """Abstract Agent Class"""

    def __init__(self, alpha=None, policy=None):
        """
        :param alpha:
        :param policy:
        """
        self.alpha = alpha
        self.policy = policy
        self.reward_history = []

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def get_reward(self, reward):
        pass


class PHCAgent(Agent):
    """
        Policy hill-climbing algorithm(PHC)
        http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
    """
    def __init__(self, delta=0.1,  action_list=None, **kwargs):
        """
        :param action_list:
        :param q_values:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.action_list = action_list  # 選択肢
        self.last_action_id = None
        self.q_values = self._init_q_values()   # 期待報酬値の初期化
        self.pi = [(1.0/len(action_list)) for idx in range(len(action_list))]
        self.delta = delta

    def _init_q_values(self):
        q_values = {}
        q_values = np.repeat(0.0, len(self.action_list))
        return q_values

    def act(self, q_values=None):
        action_id = self.policy.select_action(self.pi)    # 行動選択
        self.last_action_id = action_id
        action = self.action_list[action_id]
        return action

    def get_reward(self, reward):
        self.reward_history.append(reward)
        self.q_values[self.last_action_id] = self._update_q_value(reward)   # 期待報酬値の更新
        self.pi[self.last_action_id] = self._update_pi()

    def _update_q_value(self, reward):
        return ((1.0 - self.alpha) * self.q_values[self.last_action_id]) + (self.alpha * reward) # 通常の指数移動平均で更新

    def _update_pi(self):
       pi = self.pi[self.last_action_id]
       max_action_id = np.argmax(self.q_values)
       if self.last_action_id == max_action_id:
           update_amount = self.delta
       else:
           update_amount = ((-self.delta)/(len(self.action_list)-1))

       return update_amount
