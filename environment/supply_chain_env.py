"""
Simulated supply chain that the three agents are trained in.

This is NOT the DataCo dataset itself. It is a small simulator whose demand
and shipping delays are sampled from statistics worked out on the real DataCo
data (see data/calibration.json). Inventory, suppliers and costs are our own
assumptions, because DataCo has no inventory or supplier columns at all.

Comments marked [REAL] or [SIMULATED] show which parts are which.

The state is a plain dict:
    inventory_level            units currently in the warehouse
    pending_orders_intransit   units ordered but not arrived yet
    day_of_week, time_step     calendar position in the episode
    stockout_flag              1 if the warehouse is empty

Reward (one shared number for all three agents, weights are adjustable):
    R =   w_fulfill   * units_sold
        - w_stockout  * units_demanded_but_unavailable
        - w_holding   * holding_cost * units_above_70%_of_capacity
        - w_delay     * late_shipments_arriving_today
        - w_transport * transport_cost
        - w_procurement * procurement_cost
        - w_supplier  * 1 if a supplier failed to deliver this step
"""

import json
import random
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Weights for each term of the shared reward. They are plain parameters
    so they can be changed per experiment instead of being buried in step()."""
    w_fulfill: float = 1.0
    w_stockout: float = 2.0
    w_holding: float = 1.0
    w_delay: float = 1.5
    w_transport: float = 0.01
    # Procurement costs are currency-like units, scaled to keep reward terms
    # comparable to sales and the other cost terms.
    w_procurement: float = 0.01
    w_supplier: float = 1.0


class SupplyChainEnvV0:
    """
    One product, two suppliers, four shipping modes, one warehouse.

    Each call to step() takes one action from each agent:
        inventory_action: 'no_reorder' | 'reorder_500' | 'reorder_1000' | 'reorder_1500'
        supplier_action:  'select_A' | 'select_B' | 'split_order'
        logistics_action: 'Standard Class' | 'Second Class' | 'First Class' | 'Same Day'

    The agents depend on each other through the environment:
      - the inventory action decides how much is ordered (0 means the other
        two actions have no effect that day)
      - the supplier action decides who fills the order, which decides the
        chance that it fails
      - the logistics action decides the shipping mode, which decides the
        delay chance and duration (taken from the real data)
      - all three together decide when stock arrives, and so the reward
    """

    REORDER_QTY = {
        'no_reorder': 0,
        'reorder_500': 500,
        'reorder_1000': 1000,
        'reorder_1500': 1500,
    }

    # [SIMULATED] Supplier settings. DataCo has no supplier data.
    SUPPLIERS = {
        'A': {'reliability': 0.90, 'lead_time_days': 3, 'unit_cost': 12.0},
        'B': {'reliability': 0.70, 'lead_time_days': 6, 'unit_cost': 9.5},
    }

    # [SIMULATED] Cost per unit for each shipping mode. DataCo only records the
    # mode and how long it took, not what it cost.
    TRANSPORT_COST_PER_UNIT = {
        'Standard Class': 0.5,
        'Second Class': 0.8,
        'First Class': 1.5,
        'Same Day': 3.0,
    }

    def __init__(self, calibration_path, max_steps=180, warehouse_capacity=3000,
                 initial_inventory=800, holding_cost_per_unit=0.1,
                 reward_weights: RewardWeights = None, seed=None,
                 supplier_overrides=None, demand_volatility_multiplier=1.0):
        with open(calibration_path) as f:
            self.calibration = json.load(f)

        self.demand_pool = self.calibration['demand_pool']          # [REAL]
        self.mode_stats = self.calibration['shipping_mode_stats']    # [REAL]

        # Used by the demand-volatility experiment: stretch or shrink the real
        # demand values around their mean. 1.0 leaves the real pool unchanged.
        if demand_volatility_multiplier != 1.0:
            mean_d = sum(self.demand_pool) / len(self.demand_pool)
            self.demand_pool = [
                max(0, mean_d + (d - mean_d) * demand_volatility_multiplier)
                for d in self.demand_pool
            ]

        # Lets an experiment (or the dashboard sliders) change supplier settings
        # without editing the class defaults.
        self.SUPPLIERS = dict(SupplyChainEnvV0.SUPPLIERS)
        if supplier_overrides:
            for sid, params in supplier_overrides.items():
                self.SUPPLIERS[sid] = {**self.SUPPLIERS[sid], **params}

        self.max_steps = max_steps
        self.warehouse_capacity = warehouse_capacity                 # [SIMULATED]
        self.initial_inventory = initial_inventory                   # [SIMULATED]
        self.holding_cost_per_unit = holding_cost_per_unit           # [SIMULATED]
        self.weights = reward_weights or RewardWeights()
        # Independent random streams ensure policies using the same seed see
        # the same demand sequence, even if their actions use different numbers
        # of supplier/shipping random draws.
        seed_value = random.SystemRandom().randrange(0, 2**32) if seed is None else int(seed)
        self.demand_rng = random.Random(seed_value + 100_003)
        self.supplier_rng = random.Random(seed_value + 200_003)
        self.shipping_rng = random.Random(seed_value + 300_007)

        self.reset()

    def reset(self):
        self.t = 0
        self.inventory = self.initial_inventory
        # each shipment in transit is a dict: qty, arrival_day, supplier, is_late
        self.in_transit = []
        self.done = False
        return self._get_state()

    def _sample_demand(self):
        # [REAL] Pick a random day from the real history of daily quantities.
        # A fitted trend was tried earlier and predicted no better than the
        # plain average, so simple resampling is used instead.
        return self.demand_rng.choice(self.demand_pool)

    def _get_state(self):
        day_of_week = self.t % 7
        pending_qty = sum(s['qty'] for s in self.in_transit)
        return {
            'inventory_level': self.inventory,
            'pending_orders_intransit': pending_qty,
            'day_of_week': day_of_week,
            'time_step': self.t,
            'stockout_flag': int(self.inventory <= 0),
        }

    def step(self, inventory_action, supplier_action, logistics_action):
        if self.done:
            raise RuntimeError("step() called after episode ended -- call reset() first.")

        info = {}

        # 1. Inventory agent decides the order size
        order_qty = self.REORDER_QTY[inventory_action]

        # 2. Supplier agent decides who fills it
        supplier_failure = False
        if order_qty > 0:
            if supplier_action == 'split_order':
                half = order_qty // 2
                allocations = [('A', half), ('B', order_qty - half)]
            elif supplier_action == 'select_A':
                allocations = [('A', order_qty)]
            else:
                allocations = [('B', order_qty)]

            for supplier_id, qty in allocations:
                params = self.SUPPLIERS[supplier_id]
                fulfilled = self.supplier_rng.random() < params['reliability']  # [SIMULATED]
                if not fulfilled:
                    supplier_failure = True
                    continue  # this part of the order is lost

                # 3. Logistics agent decides how it is shipped
                mode_info = self.mode_stats[logistics_action]  # [REAL]
                is_late = self.shipping_rng.random() < mode_info['delay_probability']  # [REAL prob]
                base_duration = mode_info['avg_duration_days']  # [REAL]
                extra_delay = base_duration if is_late else 0
                lead_time = params['lead_time_days']  # [SIMULATED]
                arrival_day = self.t + lead_time + base_duration + extra_delay

                self.in_transit.append({
                    'qty': qty,
                    'arrival_day': arrival_day,
                    'supplier': supplier_id,
                    'is_late': is_late,
                })
                info.setdefault('shipments_created', []).append(
                    {'supplier': supplier_id, 'qty': qty, 'mode': logistics_action, 'late': is_late}
                )

        # 4. Today's demand [REAL]
        demand = self._sample_demand()
        fulfilled = min(self.inventory, demand)
        stockout_units = max(0, demand - self.inventory)
        self.inventory -= fulfilled

        # 5. Shipments that arrive today (or are overdue) refill the warehouse
        arrived_qty = 0
        still_in_transit = []
        late_arrivals = 0
        for shipment in self.in_transit:
            if shipment['arrival_day'] <= self.t:
                arrived_qty += shipment['qty']
                if shipment['is_late']:
                    late_arrivals += 1
            else:
                still_in_transit.append(shipment)
        self.in_transit = still_in_transit
        self.inventory = min(self.warehouse_capacity, self.inventory + arrived_qty)

        # 6. Shared reward
        excess_inventory = max(0, self.inventory - self.warehouse_capacity * 0.7)
        # Only successful allocations become shipments, so only those units
        # incur procurement and transportation costs in this model.
        created_shipments = info.get('shipments_created', [])
        shipped_qty = sum(item['qty'] for item in created_shipments)
        procurement_cost = sum(
            item['qty'] * self.SUPPLIERS[item['supplier']]['unit_cost']
            for item in created_shipments
        )
        transport_cost = self.TRANSPORT_COST_PER_UNIT[logistics_action] * shipped_qty
        w = self.weights
        reward = (
            w.w_fulfill * fulfilled
            - w.w_stockout * stockout_units
            - w.w_holding * self.holding_cost_per_unit * excess_inventory
            - w.w_delay * late_arrivals
            - w.w_transport * transport_cost
            - w.w_procurement * procurement_cost
            - w.w_supplier * (1 if supplier_failure else 0)
        )

        info.update({
            'demand': demand,
            'fulfilled': fulfilled,
            'stockout_units': stockout_units,
            'arrived_qty': arrived_qty,
            'late_arrivals': late_arrivals,
            'supplier_failure': supplier_failure,
            'shipped_qty': shipped_qty,
            'procurement_cost': procurement_cost,
            'transport_cost': transport_cost,
        })

        self.t += 1
        self.done = self.t >= self.max_steps

        return self._get_state(), reward, self.done, info
