from abc import ABCMeta, abstractmethod


class Env(metaclass=ABCMeta):

    @abstractmethod
    def step(self, **kwargs):
        pass

    @abstractmethod
    def step_log(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_config(self):
        pass
