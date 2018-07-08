import copy
import numpy as np
import ipdb
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, **kwargs):
        pass

class NormalPolicy(Policy):
    """
        与えられた確率分布に従って選択 
    """
    def __init__(self):
        super(NormalPolicy, self).__init__()

    ## @todo ボルツマン分布などで正規化
    def select_action(self, pi):
        new_pi = []
        sum_e = np.sum([np.exp(p) for p in pi ])
        for idx, p in enumerate(pi):
            new_pi.append((np.exp(p)/sum_e))

        new_pi.sort()
        print(new_pi)
        randm = np.random.rand()
        sum_p = 0.0
        for idx, p in enumerate(new_pi):
            sum_p += p
            if randm < sum_p:
                action = idx
                break
        return action

class EpsGreedyQPolicy(Policy):
    """
        ε-greedy選択 
    """
    def __init__(self, epsilon=.1, decay_rate=1):
        super(EpsGreedyQPolicy, self).__init__()
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        return action

    def decay_epsilon():    # 探索率を減少
        self.epsilon = self.epsilon*self.decay_rate

