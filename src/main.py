import tqdm

from environments import CoinFlipEnv
from agents import RandomAgent, CounterAgent, DoubleQLearning

ACTIONS = {0: "flip", 1: "label as cheater", 2: "label as not a cheater"}


def baselines(n_episodes=1_000_000):
    env = CoinFlipEnv()
    agent = RandomAgent(env)
    n_coin_flips = [9]
    counter_agents = [CounterAgent(env, n) for n in n_coin_flips]
    # print(f"Average score of {agent}: {env.compute_avg_score(agent, n_episodes=n_episodes)}")
    for counter_agent in counter_agents:
        print(f"Average score of {counter_agent}: {env.compute_avg_score(counter_agent, n_episodes=n_episodes)}")

    # Average scores (100_000 episodes | 1_000_000 episodes):
    # RandomAgent: -7.95119 | -7.983377
    # CounterAgent(n_coin_flips=0, head_threshold=0, tail_threshold=0): -7.5216
    # CounterAgent(n_coin_flips=1, head_threshold=1, tail_threshold=0): -7.437
    # CounterAgent(n_coin_flips=2, head_threshold=1, tail_threshold=1): -2.9956
    # CounterAgent(n_coin_flips=3, head_threshold=2, tail_threshold=1): -2.05441
    # CounterAgent(n_coin_flips=4, head_threshold=2, tail_threshold=2): -2.26785
    # CounterAgent(n_coin_flips=5, head_threshold=3, tail_threshold=2): -1.11272 | -1.154637
    # CounterAgent(n_coin_flips=6, head_threshold=4, tail_threshold=2): -1.32567 | -1.365762
    # CounterAgent(n_coin_flips=7, head_threshold=4, tail_threshold=3): -1.36954 | -1.418313
    # CounterAgent(n_coin_flips=8, head_threshold=5, tail_threshold=3): -1.11029 | -1.154039
    # CounterAgent(n_coin_flips=9, head_threshold=6, tail_threshold=3): -1.61561 | -1.656075
    # CounterAgent(n_coin_flips=10, head_threshold=6, tail_threshold=4): -1.68718


def train_double_q_learning():
    env = CoinFlipEnv()
    agent = DoubleQLearning(env)
    n_episodes = 1_000
    previous_record = -0.612844
    agent.fit(n_episodes=n_episodes,
              verbose=True,
              eval_interval=25_000,
              eval_episodes=1_000,
              target_reward=previous_record)
    avg_score = env.compute_avg_score(agent, n_episodes=100_000)
    print(f"Average score of {agent}: {avg_score}")
    if previous_record < avg_score:
        record = avg_score * (100_000 / 1_100_000) + \
                 env.compute_avg_score(agent, n_episodes=1_000_000) * (1_000_000 / 1_100_000)
        print(f"New record: {record}")
        agent.save_learning("double_q_learning")


def play_double_q_learning(verbose=False, n_episodes=1_000):
    env = CoinFlipEnv()
    agent = DoubleQLearning(env)
    agent.load("double_q_learning.npz")
    return play_real_game(agent, verbose=verbose, n_episodes=n_episodes)


def play_real_game(agent, verbose=True, n_episodes=1_000, init_coins=100):
    env = agent.env
    max_score = 0
    avg_score = 0
    for i in tqdm.tqdm(range(1, n_episodes + 1)):
        coins_left = init_coins
        score = 0
        while coins_left > 0:
            reward = env.play(agent, verbose=verbose)
            score += 1 if reward >= -30 else 0
            coins_left += reward
            if verbose:
                print(f"coins_left: {coins_left}, reward: {reward}")
                print(f"score: +{int(reward > -30)}")

        if score > max_score:
            max_score = score
        avg_score += (score - avg_score) / i

    return avg_score, max_score


def get_action_double_q_learning(n_heads, n_coin_flips):
    env = CoinFlipEnv()
    agent = DoubleQLearning(env)
    agent.load("double_q_learning.npz")
    obs = env.get_obs(n_heads, n_coin_flips)
    action = agent(obs)
    print(f"Action: {ACTIONS[action]}")


def get_scores_double_q_learning():
    avg_score, max_score = play_double_q_learning(verbose=False, n_episodes=100)
    print(f"Average score: {avg_score}")
    print(f"Max score: {max_score}")
    # Average score: 146.23719999999997
    # Max score: 8239 (world record: 4541)


def main():
    baselines()
    # train_double_q_learning()
    get_scores_double_q_learning()
    play_double_q_learning(verbose=True, n_episodes=1)
    get_action_double_q_learning(n_heads=0, n_coin_flips=0)


if __name__ == "__main__":
    main()
