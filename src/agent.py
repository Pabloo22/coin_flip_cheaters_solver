import abc
import gym


class Agent(abc.ABC):

    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(self, obs):
        return self.act(obs)
