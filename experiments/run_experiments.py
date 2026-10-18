"""
Controlled experiments. Every condition trains a fresh set of agents from
scratch, so any difference in the result comes from the one thing that was
changed and not from leftover learning.

    Experiment 1  exploration rate (how fast epsilon decays)
    Experiment 3  supplier reliability
    Experiment 4  demand volatility
    Experiment 7  learned policy vs. fixed rule vs. random

(Experiment 5, one agent vs. three, is the pair of scripts in training/.
Experiment 6, shared vs. hybrid reward, is run_experiment6.py.)

Each condition is one training run with one seed, so small differences
between conditions should not be over-read. It takes a couple of minutes.

Run from the project root:
    python experiments/run_experiments.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import statistics as stats
from environment.supply_chain_env import SupplyChainEnvV0
from agents.q_learning import QLearningAgent
from training.train_multi_agent import (
    run_episode, random_all_episode, fixed_baseline_episode,
    INVENTORY_ACTIONS, SUPPLIER_ACTIONS, LOGISTICS_ACTIONS, CALIB_PATH
)

EPISODE_LENGTH = 90
N_TRAIN = 3000
N_EVAL = 150


def train_and_eval(env_kwargs=None, epsilon_decay=0.998, seed=42, reward_weights=None):
    env_kwargs = env_kwargs or {}
    env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=seed,
                            reward_weights=reward_weights, **env_kwargs)
    inv_agent = QLearningAgent(INVENTORY_ACTIONS, epsilon_decay=epsilon_decay, seed=seed)
    sup_agent = QLearningAgent(SUPPLIER_ACTIONS, epsilon_decay=epsilon_decay, seed=seed+1)
    log_agent = QLearningAgent(LOGISTICS_ACTIONS, epsilon_decay=epsilon_decay, seed=seed+2)

    ep_rewards = []
    for _ in range(N_TRAIN):
        r = run_episode(env, inv_agent, sup_agent, log_agent, learn=True, greedy=False)
        ep_rewards.append(r)
        inv_agent.decay_epsilon(); sup_agent.decay_epsilon(); log_agent.decay_epsilon()

    eval_env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=seed+500,
                                 reward_weights=reward_weights, **env_kwargs)
    eval_rewards = [run_episode(eval_env, inv_agent, sup_agent, log_agent, learn=False, greedy=True)
                     for _ in range(N_EVAL)]
    return ep_rewards, eval_rewards, (inv_agent, sup_agent, log_agent)


def summarize(name, eval_rewards):
    print(f"{name:<45}{stats.mean(eval_rewards):>12.2f}{stats.stdev(eval_rewards):>10.2f}")


def header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Config':<45}{'Mean reward':>12}{'Std':>10}")


def main():
    header("EXPERIMENT 1: Exploration rate (epsilon decay)")
    for decay, label in [(0.99, 'Fast decay (0.99, ~150 ep to min)'),
                         (0.998, 'Default decay (0.998, ~1500 ep to min)'),
                         (0.9995, 'Slow decay (0.9995, ~6000 ep to min)')]:
        _, eval_r, _ = train_and_eval(epsilon_decay=decay)
        summarize(label, eval_r)

    print()
    header("EXPERIMENT 3: Supplier reliability conditions")
    configs = {
        'Baseline (A=0.90, B=0.70)': None,
        'Both suppliers unreliable (A=0.60, B=0.50)': {'A': {'reliability': 0.60}, 'B': {'reliability': 0.50}},
        'Both suppliers very reliable (A=0.98, B=0.95)': {'A': {'reliability': 0.98}, 'B': {'reliability': 0.95}},
    }
    for label, override in configs.items():
        _, eval_r, _ = train_and_eval(env_kwargs={'supplier_overrides': override} if override else {})
        summarize(label, eval_r)

    print()
    header("EXPERIMENT 4: Demand volatility")
    for mult, label in [(1.0, 'Real historical volatility (1.0x)'),
                        (1.5, 'Elevated volatility (1.5x spread)'),
                        (2.0, 'High volatility (2.0x spread)')]:
        _, eval_r, _ = train_and_eval(env_kwargs={'demand_volatility_multiplier': mult})
        summarize(label, eval_r)

    print()
    header("EXPERIMENT 7: Learned policy vs. fixed rule vs. random")
    _, learned_eval, _ = train_and_eval()
    env_b = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    baseline_eval = [fixed_baseline_episode(env_b) for _ in range(N_EVAL)]
    env_r = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    rng = random.Random(7)
    random_eval = [random_all_episode(env_r, rng) for _ in range(N_EVAL)]
    summarize('Independent Q-Learning (3 agents)', learned_eval)
    summarize('Fixed heuristic baseline', baseline_eval)
    summarize('Fully random', random_eval)


if __name__ == '__main__':
    main()
