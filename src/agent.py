import abc
import gym


class Agent(abc.ABC):

    def __init__(self, env: gym.Env):
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs, **kwargs) -> int:
        pass

    def __call__(self, obs, **kwargs):
        return self.select_action(obs, **kwargs)
