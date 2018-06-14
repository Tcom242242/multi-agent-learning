from abc import ABCMeta, abstractmethod


class Game(metaclass=ABCMeta):

    @abstractmethod
    def _init_agents(self, conf_agents):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def game_log(self):
        pass

    def compute_global_rewards(self):
        pass

    def get_conf(self):
        pass
