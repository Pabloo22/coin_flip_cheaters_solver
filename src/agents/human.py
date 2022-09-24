from .agent import Agent


class Human(Agent):

    def __init__(self, env):
        super().__init__(env)

    def select_action(self, obs, **kwargs) -> int:
        while True:
            action = input("Enter action (0: flip coin, 1: label as cheater, 2: label as fair): ")
            if action in ["0", "1", "2"]:
                return int(action)

    def __str__(self):
        return "Human"
