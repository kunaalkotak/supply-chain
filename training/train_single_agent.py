"""
Single-agent baseline: only the inventory agent learns.

Supplier and shipping are fixed (Supplier A, Standard Class), so this is a
plain one-agent problem. It is used to check that the learning setup works
before adding the other two agents, and as the "1 agent" case when comparing
against the full multi-agent system.

Run from the project root:
    python training/train_single_agent.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
from environment.supply_chain_env import SupplyChainEnvV0
from environment.discretization import state_to_key
from agents.q_learning import QLearningAgent

CALIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration.json')
INVENTORY_ACTIONS = ['no_reorder', 'reorder_500', 'reorder_1000', 'reorder_1500']

FIXED_SUPPLIER = 'select_A'
FIXED_LOGISTICS = 'Standard Class'

N_TRAIN_EPISODES = 3000
EPISODE_LENGTH = 90


def run_episode(env, agent, learn=True, greedy=False):
    state = env.reset()
    state_key = state_to_key(state)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.choose_action(state_key, greedy=greedy)
        next_state, reward, done, info = env.step(action, FIXED_SUPPLIER, FIXED_LOGISTICS)
        next_key = state_to_key(next_state)
        if learn:
            agent.update(state_key, action, reward, next_key, done)
        state_key = next_key
        total_reward += reward
    return total_reward


def naive_threshold_policy_episode(env, threshold=400, reorder_action='reorder_1000'):
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = 'no_reorder' if state['inventory_level'] >= threshold else reorder_action
        state, reward, done, info = env.step(action, FIXED_SUPPLIER, FIXED_LOGISTICS)
        total_reward += reward
    return total_reward


def random_policy_episode(env, rng):
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = rng.choice(INVENTORY_ACTIONS)
        state, reward, done, info = env.step(action, FIXED_SUPPLIER, FIXED_LOGISTICS)
        total_reward += reward
    return total_reward


def main():
    env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=42)
    agent = QLearningAgent(INVENTORY_ACTIONS, alpha=0.1, gamma=0.9,
                            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998,
                            seed=42)

    episode_rewards = []
    for ep in range(N_TRAIN_EPISODES):
        total_reward = run_episode(env, agent, learn=True, greedy=False)
        episode_rewards.append(total_reward)
        agent.decay_epsilon()

    # evaluate the learned policy against two baselines on the same demand
    N_EVAL = 200
    eval_env_learned = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    learned_rewards = [run_episode(eval_env_learned, agent, learn=False, greedy=True)
                        for _ in range(N_EVAL)]

    eval_env_naive = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    naive_rewards = [naive_threshold_policy_episode(eval_env_naive) for _ in range(N_EVAL)]

    eval_env_random = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    rng = random.Random(7)
    random_rewards = [random_policy_episode(eval_env_random, rng) for _ in range(N_EVAL)]

    import statistics as stats
    print(f"Trained for {N_TRAIN_EPISODES} episodes, {EPISODE_LENGTH} steps each.")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table size (states visited): {len(agent.Q)}")
    print()
    print(f"{'Policy':<20}{'Mean total reward':>20}{'Std':>10}")
    print(f"{'Learned (greedy)':<20}{stats.mean(learned_rewards):>20.2f}{stats.stdev(learned_rewards):>10.2f}")
    print(f"{'Naive threshold':<20}{stats.mean(naive_rewards):>20.2f}{stats.stdev(naive_rewards):>10.2f}")
    print(f"{'Random':<20}{stats.mean(random_rewards):>20.2f}{stats.stdev(random_rewards):>10.2f}")

    return episode_rewards, learned_rewards, naive_rewards, random_rewards, agent


if __name__ == '__main__':
    main()
