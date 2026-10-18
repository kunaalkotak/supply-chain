"""
Main training script: all three agents (inventory, supplier, logistics) learn
at the same time with independent Q-learning.

Each agent has its own Q-table and never sees the others' tables. What ties
them together is that they all get the same reward after every step, and that
their actions feed into the same environment:
  - the inventory action sets the order size, which decides whether the
    supplier and logistics choices matter at all that day
  - the supplier action decides which reliability roll applies
  - the logistics action decides which real delay probability applies
  - the result decides tomorrow's stock, which is what inventory sees next

Run from the project root:
    python training/train_multi_agent.py                 # train and print results
    python training/train_multi_agent.py --save          # also save Q-tables for the dashboard
    python training/train_multi_agent.py --plots         # also save figures to results/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import random
import statistics as stats
from environment.supply_chain_env import SupplyChainEnvV0
from environment.discretization import state_to_key, supplier_state_key, logistics_state_key
from agents.q_learning import QLearningAgent

ROOT = os.path.join(os.path.dirname(__file__), '..')
CALIB_PATH = os.path.join(ROOT, 'data', 'calibration.json')
QTABLE_PATH = os.path.join(ROOT, 'dashboard', 'trained_qtables.json')
RESULTS_DIR = os.path.join(ROOT, 'results')

INVENTORY_ACTIONS = ['no_reorder', 'reorder_500', 'reorder_1000', 'reorder_1500']
SUPPLIER_ACTIONS = ['select_A', 'select_B', 'split_order']
LOGISTICS_ACTIONS = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

N_TRAIN_EPISODES = 4000
EPISODE_LENGTH = 90


def make_agents(seed=42):
    inv_agent = QLearningAgent(INVENTORY_ACTIONS, alpha=0.1, gamma=0.9,
                                epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998, seed=seed)
    sup_agent = QLearningAgent(SUPPLIER_ACTIONS, alpha=0.1, gamma=0.9,
                                epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998, seed=seed + 1)
    log_agent = QLearningAgent(LOGISTICS_ACTIONS, alpha=0.1, gamma=0.9,
                                epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998, seed=seed + 2)
    return inv_agent, sup_agent, log_agent


def run_episode(env, inv_agent, sup_agent, log_agent, learn=True, greedy=False):
    state = env.reset()
    inv_key = state_to_key(state)
    sup_key = supplier_state_key(state)
    log_key = logistics_state_key(state)
    total_reward = 0.0
    done = False

    while not done:
        inv_action = inv_agent.choose_action(inv_key, greedy=greedy)
        sup_action = sup_agent.choose_action(sup_key, greedy=greedy)
        log_action = log_agent.choose_action(log_key, greedy=greedy)

        next_state, reward, done, info = env.step(inv_action, sup_action, log_action)

        next_inv_key = state_to_key(next_state)
        next_sup_key = supplier_state_key(next_state)
        next_log_key = logistics_state_key(next_state)

        if learn:
            inv_agent.update(inv_key, inv_action, reward, next_inv_key, done)
            sup_agent.update(sup_key, sup_action, reward, next_sup_key, done)
            log_agent.update(log_key, log_action, reward, next_log_key, done)

        inv_key, sup_key, log_key = next_inv_key, next_sup_key, next_log_key
        total_reward += reward

    return total_reward


def random_all_episode(env, rng):
    """All three agents pick random actions every day."""
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        a1 = rng.choice(INVENTORY_ACTIONS)
        a2 = rng.choice(SUPPLIER_ACTIONS)
        a3 = rng.choice(LOGISTICS_ACTIONS)
        state, reward, done, info = env.step(a1, a2, a3)
        total_reward += reward
    return total_reward


def fixed_baseline_episode(env, threshold=400, reorder_action='reorder_1000'):
    """Simple rule of thumb: reorder 1,000 units whenever stock drops below
    400, always use Supplier A, always ship Standard Class."""
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        inv_action = 'no_reorder' if state['inventory_level'] >= threshold else reorder_action
        state, reward, done, info = env.step(inv_action, 'select_A', 'Standard Class')
        total_reward += reward
    return total_reward


def save_qtables(inv_agent, sup_agent, log_agent, path=QTABLE_PATH):
    """Write the three Q-tables to JSON so the dashboard can load them without
    retraining. Keys like ('low', 'none') are stored as 'low|none'."""
    def to_dict(agent):
        return {'|'.join(k): v for k, v in sorted(agent.Q.items())}
    with open(path, 'w') as f:
        json.dump({'inventory': to_dict(inv_agent),
                   'supplier': to_dict(sup_agent),
                   'logistics': to_dict(log_agent)}, f, indent=2)
    print(f"Saved Q-tables to {os.path.relpath(path)}")


def save_plots(episode_rewards, learned, baseline, random_r, out_dir=RESULTS_DIR):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # learning curve: raw episodes in grey, 100-episode moving average on top
    window = 100
    smooth = [sum(episode_rewards[max(0, i - window + 1):i + 1]) / len(episode_rewards[max(0, i - window + 1):i + 1])
              for i in range(len(episode_rewards))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(episode_rewards, color='#c8c8c8', linewidth=0.6, label='Single episode')
    ax.plot(smooth, color='#1f5fa8', linewidth=2, label=f'Average of last {window}')
    ax.set_xlabel('Training episode')
    ax.set_ylabel('Total reward in the episode')
    ax.set_title('Learning curve (three agents, 90-day episodes)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'learning_curve.png'), dpi=150)
    plt.close(fig)

    # bar chart: mean reward of each policy with one standard deviation
    names = ['Learned (3 agents)', 'Fixed rule', 'Random']
    means = [stats.mean(x) for x in (learned, baseline, random_r)]
    stds = [stats.stdev(x) for x in (learned, baseline, random_r)]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=['#1f5fa8', '#8a8a8a', '#b9b9b9'])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean total reward per episode')
    ax.set_title(f'Policy comparison ({len(learned)} test episodes each)')
    # put each value just beyond its error bar so nothing overlaps
    for bar, m, sd in zip(bars, means, stds):
        x = bar.get_x() + bar.get_width() / 2
        if m >= 0:
            ax.text(x, m + sd + 150, f'{m:,.0f}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(x, m - sd - 150, f'{m:,.0f}', ha='center', va='top', fontsize=10)
    ax.set_ylim(min(m - sd for m, sd in zip(means, stds)) * 1.6,
                max(m + sd for m, sd in zip(means, stds)) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'policy_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"Saved figures to {os.path.relpath(out_dir)}/")


def main(save=False, plots=False):
    env = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=42)
    inv_agent, sup_agent, log_agent = make_agents(seed=42)

    print(f"Training for {N_TRAIN_EPISODES} episodes...")
    episode_rewards = []
    for ep in range(N_TRAIN_EPISODES):
        r = run_episode(env, inv_agent, sup_agent, log_agent, learn=True, greedy=False)
        episode_rewards.append(r)
        inv_agent.decay_epsilon()
        sup_agent.decay_epsilon()
        log_agent.decay_epsilon()

    # test all three policies on the same demand sequence (same seed)
    N_EVAL = 200

    eval_env1 = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    learned_rewards = [run_episode(eval_env1, inv_agent, sup_agent, log_agent, learn=False, greedy=True)
                        for _ in range(N_EVAL)]

    eval_env2 = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    baseline_rewards = [fixed_baseline_episode(eval_env2) for _ in range(N_EVAL)]

    eval_env3 = SupplyChainEnvV0(CALIB_PATH, max_steps=EPISODE_LENGTH, seed=1000)
    rng = random.Random(7)
    random_rewards = [random_all_episode(eval_env3, rng) for _ in range(N_EVAL)]

    print(f"Trained {N_TRAIN_EPISODES} episodes x {EPISODE_LENGTH} steps.")
    print(f"Q-table sizes: inventory={len(inv_agent.Q)}, supplier={len(sup_agent.Q)}, logistics={len(log_agent.Q)}")
    print()
    print(f"{'Policy':<25}{'Mean total reward':>20}{'Std':>10}")
    print(f"{'Learned (all 3, greedy)':<25}{stats.mean(learned_rewards):>20.2f}{stats.stdev(learned_rewards):>10.2f}")
    print(f"{'Fixed heuristic baseline':<25}{stats.mean(baseline_rewards):>20.2f}{stats.stdev(baseline_rewards):>10.2f}")
    print(f"{'Fully random (all 3)':<25}{stats.mean(random_rewards):>20.2f}{stats.stdev(random_rewards):>10.2f}")

    if save:
        save_qtables(inv_agent, sup_agent, log_agent)
    if plots:
        save_plots(episode_rewards, learned_rewards, baseline_rewards, random_rewards)

    return episode_rewards, learned_rewards, baseline_rewards, random_rewards, inv_agent, sup_agent, log_agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the three-agent supply chain system.")
    parser.add_argument('--save', action='store_true',
                        help="write the trained Q-tables to dashboard/trained_qtables.json")
    parser.add_argument('--plots', action='store_true',
                        help="save a learning curve and a comparison chart to results/")
    args = parser.parse_args()
    main(save=args.save, plots=args.plots)
