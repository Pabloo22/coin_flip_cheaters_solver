from .agent import Agent


class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def select_action(self, obs, **kwargs) -> int:
        return self.env.action_space.sample()

    def __str__(self):
        return "RandomAgent"


