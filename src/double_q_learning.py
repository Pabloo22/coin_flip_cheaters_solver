import numpy as np
import tqdm
from gym import spaces

from agent import Agent
from coin_flip_env import CoinFlipEnv


class DoubleQLearning(Agent):
    """
    Double Q-Learning is an algorithm that solves particular issues in Q-Learning, especially when Q-Learning can be
    tricked to take the bad action based on some positive rewards, while the expected reward of this action is
    guaranteed to be negative.
    It does that by maintaining two Q-Value lists each updating itself from the other. In short it finds the action
    that maximizes the Q-Value in one list, but instead of using this Q-Value, it uses the action to get a Q-Value
    from the other list.

    https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3
    """
    env: CoinFlipEnv
    max_coin_flips: int
    epsilon: float
    discount: float
    step_size: float
    random_state: np.random.RandomState
    init_method: str
    ...

    def __init__(self,
                 env: CoinFlipEnv,
                 max_coin_flips: int = 15,
                 epsilon: float = 0.1,
                 discount: float = 0.9,
                 step_size: float = 0.1,
                 init_method: str or int or float = "zeros",
                 seed: int = None,
                 normalize_reward: bool = False,
                 max_reward: float = 1.0,
                 min_reward: float = -1.0):

        super().__init__(env)
        self.epsilon = epsilon
        self.discount = discount
        self.step_size = step_size
        self.random_state = np.random.RandomState(seed)
        self.init_method = init_method
        self.q_a = None
        self.q_b = None
        self.initialized = False
        self.normalize = normalize_reward
        self.max_reward = max_reward
        self.min_reward = min_reward

        # In order to have a finite search space, we set the maximum number of coin flips to a fixed value.
        # Therefore, we can compute the total number of states in the game.
        # n_states = (max_coin_flips + 1)**2 * (max_coin_flips + 2) // 2
        # If the maximum number of coin flips is 50, the total number of states is:
        # (50 + 1)^2 * (50 + 2) // 2 = 67626
        # If the maximum number of coin flips is 30, the total number of states is:
        # (30 + 1)^2 * (30 + 2) // 2 = 15376

        # Why this formula?
        # If we compute the number of possible states for each amount of coins without
        # taking into account the number of coins left, we get:
        #     - if 0 coins have been flipped: 1 state
        #     - if 1 coin has been flipped: 2 states, 0 heads or 1 head
        #     - if 2 coins have been flipped: 3 states, 0 heads, 1 head or 2 heads
        #     - if 3 coins have been flipped: 4 states, 0 heads, 1 head, 2 heads or 3 heads
        #     - etc
        # We can see that the number of states is equal to the sum of the number of states for each amount of coins.
        # Now, if we want to take into account the number of coins left, we need to multiply the number of states
        # computed above by the sum of possible quantities of coins left. Thanks to the fact that the number of coins
        # flips is bounded, we can just consider any number of coins left between 0 and max_coin_flips. Therefore,
        # we have to multiply the aforementioned formula by (max_coin_flips + 1).

        # The total number of states if we play without taking into account the number of coins left is:
        self.n_states = (max_coin_flips + 1) * (max_coin_flips + 2) // 2 * (max_coin_flips + 1)

        self.n_actions = self.env.action_space.n

    def normalize_reward(self, reward: float):
        """Normalizes the reward to be between 0 and 1."""
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)

    def select_action(self, obs: spaces.Dict, training: bool = False, use_a: bool = True, use_b: bool = True) -> int:
        """Selects an action to perform in the current state."""
        state_num = self.env.obs_to_int(obs)
        if training and self.random_state.rand() < self.epsilon:
            action = self.env.action_space.sample()
        elif use_a and use_b:
            q = (self.q_a[state_num] + self.q_b[state_num]) / 2
            action = int(np.argmax(q))
        elif use_a:
            action = int(np.argmax(self.q_a[state_num]))
        elif use_b:
            action = int(np.argmax(self.q_b[state_num]))
        else:
            raise ValueError("Either use_a or use_b must be True.")

        return action

    def initialize_q_values(self):
        """Initializes the Q-Values."""
        if self.init_method == "zeros":
            self.q_a = np.zeros((self.n_states, self.n_actions))
            self.q_b = np.zeros((self.n_states, self.n_actions))
        elif self.init_method == "uniform":
            self.q_a = self.random_state.uniform(0, 1, (self.n_states, self.n_actions))
            self.q_b = self.random_state.uniform(0, 1, (self.n_states, self.n_actions))
        elif self.init_method == "random":
            self.q_a = self.random_state.normal(0, 1, (self.n_states, self.n_actions))
            self.q_b = self.random_state.normal(0, 1, (self.n_states, self.n_actions))
        elif isinstance(self.init_method, int) or isinstance(self.init_method, float):
            self.q_a = self.random_state.normal(self.init_method, 1, (self.n_states, self.n_actions))
            self.q_b = self.random_state.normal(self.init_method, 1, (self.n_states, self.n_actions))
        else:
            raise ValueError("init_method must be either 'zeros', 'uniform', 'random', or a number.")

        self.initialized = True

    def fit(self,
            n_episodes: int = 100_000,
            eval_interval: int = 10_000,
            verbose: bool = True,
            target_reward: float = 0,
            eval_episodes: int = 100,):
        """Trains the agent on the given environment."""

        if not self.initialized:
            self.initialize_q_values()

        target_reached = False
        for episode in tqdm.tqdm(range(n_episodes), desc="Training", disable=not verbose):
            obs = self.env.reset()
            state = self.env.obs_to_int(obs)
            terminated = False
            while not terminated:
                action = self.select_action(obs, training=True)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                reward = self.normalize_reward(reward) if self.normalize else reward
                next_state = self.env.obs_to_int(new_obs)

                update_a = self.random_state.rand() < 0.5
                if update_a:
                    a_star = self.select_action(new_obs, training=False, use_b=False)
                    target = reward + self.discount * self.q_b[next_state, a_star]
                    self.q_a[state, action] += self.step_size * (target - self.q_a[state, action])
                else:
                    b_star = self.select_action(new_obs, use_a=False)
                    target = reward + self.discount * self.q_a[next_state, b_star]
                    self.q_b[state, action] += self.step_size * (target - self.q_b[state, action])
                state = next_state
                obs = new_obs
            if episode % eval_interval == 0 and verbose:
                avg_reward = self.env.compute_avg_score(self, n_episodes=eval_episodes)
                print(f"Episode {episode}: {avg_reward}")
                if avg_reward >= target_reward:
                    n_episodes_array = [10000, 100000, 100000]
                    for n in n_episodes_array:
                        print(f"Target reward reached. Evaluation with {n} episodes:", end=" ")
                        avg_reward = self.env.compute_avg_score(self, n_episodes=n)
                        print(avg_reward)
                        if avg_reward >= target_reward:
                            target_reached = True
                        else:
                            target_reached = False
                            break

                    if target_reached:
                        print("Target reward reached.")
                        break

    def save_learning(self, path: str):
        """Saves the agent to the given path."""
        np.savez(path, q_a=self.q_a, q_b=self.q_b)

    def load(self, path: str):
        """Loads the agent from the given path."""
        data = np.load(path)
        self.q_a = data["q_a"]
        self.q_b = data["q_b"]
        self.initialized = True

    def __str__(self):
        return "Double Q-Learning"

    def __call__(self, obs: spaces.Dict, training: bool = False) -> int:
        return self.select_action(obs, training=training)
