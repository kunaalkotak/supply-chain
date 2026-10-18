"""
Tests for the Q-learning agent and the state discretization.
Run from the project root with:  python -m pytest tests -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from agents.q_learning import QLearningAgent
from environment.discretization import (
    discretize_inventory, discretize_pending, state_to_key,
)

ACTIONS = ['a', 'b', 'c']


def test_update_moves_q_towards_the_target():
    agent = QLearningAgent(ACTIONS, alpha=0.1, gamma=0.9)
    # Q starts at 0 everywhere, so target = 10 + 0.9 * 0 = 10 and the new
    # value is 0 + 0.1 * (10 - 0) = 1
    agent.update('s1', 'a', reward=10, next_state_key='s2', done=False)
    assert agent.Q['s1']['a'] == 1.0


def test_update_uses_best_next_value():
    agent = QLearningAgent(ACTIONS, alpha=0.5, gamma=0.9)
    agent.Q['s2']['b'] = 20.0
    agent.update('s1', 'a', reward=0, next_state_key='s2', done=False)
    # target = 0 + 0.9 * 20 = 18, new Q = 0 + 0.5 * 18
    assert agent.Q['s1']['a'] == 9.0


def test_terminal_update_ignores_next_state():
    agent = QLearningAgent(ACTIONS, alpha=0.5, gamma=0.9)
    agent.Q['s2']['b'] = 1000.0
    agent.update('s1', 'a', reward=4, next_state_key='s2', done=True)
    assert agent.Q['s1']['a'] == 2.0


def test_greedy_choice_is_the_highest_q_value():
    agent = QLearningAgent(ACTIONS, epsilon_start=1.0)
    agent.Q['s']['c'] = 5.0
    # even with epsilon = 1, greedy=True must ignore exploration
    assert all(agent.choose_action('s', greedy=True) == 'c' for _ in range(20))


def test_epsilon_decays_but_stops_at_the_minimum():
    agent = QLearningAgent(ACTIONS, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.5)
    for _ in range(20):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05


def test_inventory_bands():
    assert discretize_inventory(0) == 'stockout'
    assert discretize_inventory(100) == 'very_low'
    assert discretize_inventory(800) == 'low'
    assert discretize_inventory(1500) == 'medium'
    assert discretize_inventory(2200) == 'high'
    assert discretize_inventory(3000) == 'overstock'


def test_pending_bands():
    assert discretize_pending(0) == 'none'
    assert discretize_pending(500) == 'some'
    assert discretize_pending(1000) == 'a_lot'


def test_saved_qtables_cover_the_starting_state():
    """The dashboard starts every run at 800 units with nothing on order, so
    the saved tables need a row for that state."""
    start = {'inventory_level': 800, 'pending_orders_intransit': 0}
    key = '|'.join(state_to_key(start))
    path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'trained_qtables.json')
    with open(path) as f:
        tables = json.load(f)
    assert key in tables['inventory']
    assert key in tables['logistics']
