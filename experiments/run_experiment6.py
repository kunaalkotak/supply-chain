"""
Experiment 6: does giving each agent a signal about its own mistakes help?

Two training setups are compared:
  shared  - every agent learns from the same shared reward (the default)
  hybrid  - each agent learns from the shared reward minus an extra penalty
            for its own outcome:
              inventory:  - units of unmet demand
              supplier:   - 5 if its supplier failed to deliver
              logistics:  - 5 for every late arrival

Both setups are scored on the same shared reward, so the comparison is fair.
Only the training signal differs.

Run from the project root:
    python experiments/run_experiment6.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import statistics as stats
from environment.supply_chain_env import SupplyChainEnvV0
from agents.q_learning import QLearningAgent
from environment.discretization import state_to_key, supplier_state_key, logistics_state_key
from training.train_multi_agent import INVENTORY_ACTIONS, SUPPLIER_ACTIONS, LOGISTICS_ACTIONS, CALIB_PATH

EPISODE_LENGTH = 90
N_TRAIN = 3000
N_EVAL = 150


def run_episode_hybrid(env, inv_agent, sup_agent, log_agent, use_hybrid, learn=True, greedy=False):
    state = env.reset()
    inv_key = state_to_key(state)
    sup_key = supplier_state_key(state)
    log_key = logistics_state_key(state)
    total_global_reward = 0.0
    done = False

    while not done:
        inv_action = inv_agent.choose_action(inv_key, greedy=greedy)
        sup_action = sup_agent.choose_action(sup_key, greedy=greedy)
        log_action = log_agent.choose_action(log_key, greedy=greedy)

        next_state, global_reward, done, info = env.step(inv_action, sup_action, log_action)

        if use_hybrid:
            inv_signal = global_reward - info['stockout_units']
            sup_signal = global_reward - (5.0 if info['supplier_failure'] else 0.0)
            log_signal = global_reward - (5.0 * info['late_arrivals'])
        else:
            inv_signal = sup_signal = log_signal = global_reward

        next_inv_key = state_to_key(next_state)
        next_sup_key = supplier_state_key(next_state)
        next_log_key = logistics_state_key(next_state)

        if learn:
            inv_agent.update(inv_key, inv_action, inv_signal, next_inv_key, done)
            sup_agent.update(sup_key, sup_action, sup_signal, next_sup_key, done)
            log_agent.update(log_key, log_action, log_signal, next_log_key, done)

        inv_key, sup_key, log_key = next_inv_key, next_sup_key, next_log_key
        total_global_reward += global_reward

    return total_global_reward


def train_and_eval(use_hybrid, seed=42):
    env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=seed)
    inv_agent = QLearningAgent(INVENTORY_ACTIONS, epsilon_decay=0.998, seed=seed)
    sup_agent = QLearningAgent(SUPPLIER_ACTIONS, epsilon_decay=0.998, seed=seed+1)
    log_agent = QLearningAgent(LOGISTICS_ACTIONS, epsilon_decay=0.998, seed=seed+2)

    for _ in range(N_TRAIN):
        run_episode_hybrid(env, inv_agent, sup_agent, log_agent, use_hybrid, learn=True, greedy=False)
        inv_agent.decay_epsilon(); sup_agent.decay_epsilon(); log_agent.decay_epsilon()

    eval_env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=seed+500)
    eval_rewards = [run_episode_hybrid(eval_env, inv_agent, sup_agent, log_agent, use_hybrid,
                                         learn=False, greedy=True) for _ in range(N_EVAL)]
    return eval_rewards


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT 6: Shared (global-only) reward vs. Hybrid (local + global) reward")
    print("=" * 80)
    print(f"{'Config':<45}{'Mean reward':>12}{'Std':>10}")

    shared_eval = train_and_eval(use_hybrid=False)
    hybrid_eval = train_and_eval(use_hybrid=True)

    print(f"{'Shared (global-only) reward':<45}{stats.mean(shared_eval):>12.2f}{stats.stdev(shared_eval):>10.2f}")
    print(f"{'Hybrid (local + global) reward':<45}{stats.mean(hybrid_eval):>12.2f}{stats.stdev(hybrid_eval):>10.2f}")
