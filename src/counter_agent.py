from agent import Agent

from coin_flip_env import CoinFlipEnv


class CounterAgent(Agent):

    def __init__(self, env: CoinFlipEnv, n_coin_flips=7):
        super().__init__(env)
        self.n_coin_flips = n_coin_flips
        self.percentage_threshold = (0.5 + env.cheater_head_prob) / 2
        self.head_threshold = round(self.percentage_threshold * n_coin_flips)
        self.tail_threshold = n_coin_flips - self.head_threshold

    def __call__(self, obs):
        percentage_heads, n_coin_flips = obs["heads"][0], obs["n_coin_flips"]
        n_heads = round(percentage_heads * n_coin_flips)
        n_tails = n_coin_flips - n_heads
        if n_heads >= self.head_threshold or n_tails >= self.tail_threshold:
            return 1 if percentage_heads > self.percentage_threshold else 2
        else:
            return 0

    def __str__(self):
        return f"CounterAgent(n_coin_flips={self.n_coin_flips}, " \
               f"head_threshold={self.head_threshold}, " \
               f"tail_threshold={self.tail_threshold})"
