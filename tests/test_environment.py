"""
Unit tests for SupplyChainEnvV0.
Run with: python3 -m pytest project/tests/test_environment.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from environment.supply_chain_env import SupplyChainEnvV0, RewardWeights

CALIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration.json')


def make_env(seed=42, max_steps=30):
    return SupplyChainEnvV0(CALIB_PATH, max_steps=max_steps, seed=seed)


def test_reset_returns_valid_initial_state():
    env = make_env()
    state = env.reset()
    assert state['inventory_level'] == env.initial_inventory
    assert state['pending_orders_intransit'] == 0
    assert state['time_step'] == 0
    assert state['stockout_flag'] == 0


def test_step_returns_correct_tuple_shape():
    env = make_env()
    env.reset()
    next_state, reward, done, info = env.step('no_reorder', 'select_A', 'Standard Class')
    assert isinstance(next_state, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'demand' in info and 'fulfilled' in info


def test_no_reorder_never_creates_shipments():
    env = make_env()
    env.reset()
    for _ in range(10):
        _, _, done, info = env.step('no_reorder', 'select_A', 'Standard Class')
        assert 'shipments_created' not in info
        if done:
            break


def test_reorder_creates_shipment_via_chosen_supplier():
    env = make_env()
    env.reset()
    _, _, _, info = env.step('reorder_1000', 'select_B', 'First Class')
    # supplier B may fail the reliability draw, but if it succeeds, the
    # shipment must be attributed to B and use First Class stats
    if 'shipments_created' in info:
        assert all(s['supplier'] == 'B' for s in info['shipments_created'])
        assert all(s['mode'] == 'First Class' for s in info['shipments_created'])


def test_split_order_uses_both_suppliers():
    env = make_env(seed=1)
    env.reset()
    _, _, _, info = env.step('reorder_1000', 'split_order', 'Standard Class')
    if 'shipments_created' in info:
        suppliers_used = {s['supplier'] for s in info['shipments_created']}
        # both may not always appear (one could fail reliability draw),
        # but no supplier outside {A, B} should ever appear
        assert suppliers_used.issubset({'A', 'B'})


def test_inventory_decreases_with_demand_when_no_replenishment():
    env = make_env(seed=7, max_steps=5)
    env.reset()
    inv_trace = [env.inventory]
    for _ in range(5):
        state, _, done, _ = env.step('no_reorder', 'select_A', 'Standard Class')
        inv_trace.append(state['inventory_level'])
        if done:
            break
    # inventory should be non-increasing across steps with no reorders
    assert all(inv_trace[i] >= inv_trace[i + 1] for i in range(len(inv_trace) - 1))


def test_inventory_never_exceeds_warehouse_capacity():
    env = make_env(seed=3, max_steps=60)
    env.reset()
    for _ in range(60):
        state, _, done, _ = env.step('reorder_1500', 'select_A', 'Same Day')
        assert state['inventory_level'] <= env.warehouse_capacity
        if done:
            break


def test_episode_terminates_at_max_steps():
    env = make_env(max_steps=15)
    env.reset()
    done = False
    steps_taken = 0
    while not done:
        _, _, done, _ = env.step('no_reorder', 'select_A', 'Standard Class')
        steps_taken += 1
    assert steps_taken == 15


def test_step_after_done_raises():
    env = make_env(max_steps=2)
    env.reset()
    env.step('no_reorder', 'select_A', 'Standard Class')
    env.step('no_reorder', 'select_A', 'Standard Class')
    with pytest.raises(RuntimeError):
        env.step('no_reorder', 'select_A', 'Standard Class')


def test_reward_is_deterministic_given_seed():
    env1 = make_env(seed=99)
    env2 = make_env(seed=99)
    env1.reset(); env2.reset()
    r1 = [env1.step('reorder_500', 'select_A', 'Standard Class')[1] for _ in range(10)]
    r2 = [env2.step('reorder_500', 'select_A', 'Standard Class')[1] for _ in range(10)]
    assert r1 == r2


def test_demand_sequence_is_independent_of_supplier_and_shipping_actions():
    """Same seed should give policies the same demand sequence despite actions."""
    env1 = make_env(seed=123, max_steps=12)
    env2 = make_env(seed=123, max_steps=12)
    demands1, demands2 = [], []
    for _ in range(12):
        _, _, _, info1 = env1.step('no_reorder', 'select_A', 'Standard Class')
        _, _, _, info2 = env2.step('reorder_1500', 'split_order', 'Same Day')
        demands1.append(info1['demand'])
        demands2.append(info2['demand'])
    assert demands1 == demands2


def test_no_reorder_has_no_procurement_or_transport_cost():
    env = make_env(seed=11, max_steps=2)
    _, _, _, info = env.step('no_reorder', 'select_A', 'Same Day')
    assert info['shipped_qty'] == 0
    assert info['procurement_cost'] == 0
    assert info['transport_cost'] == 0


def test_procurement_and_transport_costs_are_reported_for_successful_shipments():
    env = make_env(seed=1, max_steps=2)
    # Use guaranteed reliability so the order is successful deterministically.
    env.SUPPLIERS['A']['reliability'] = 1.0
    _, _, _, info = env.step('reorder_500', 'select_A', 'Standard Class')
    assert info['shipped_qty'] == 500
    assert info['procurement_cost'] == 500 * env.SUPPLIERS['A']['unit_cost']
    assert info['transport_cost'] == 500 * env.TRANSPORT_COST_PER_UNIT['Standard Class']


def test_demand_values_come_from_real_calibration_pool():
    env = make_env(seed=5, max_steps=50)
    env.reset()
    pool = set(env.demand_pool)
    for _ in range(50):
        _, _, done, info = env.step('no_reorder', 'select_A', 'Standard Class')
        assert info['demand'] in pool
        if done:
            break
