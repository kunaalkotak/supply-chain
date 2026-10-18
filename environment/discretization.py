"""
Turns the raw state from the environment into a small label the Q-tables can
use as a key. Inventory can be anything from 0 to 3000, which is far too many
values for a table, so it is grouped into six bands. Pending orders are grouped
into three.

    inventory_level -> stockout, very_low, low, medium, high, overstock
    pending_orders  -> none, some, a_lot

Day of week and time step are left out on purpose. The demand data showed no
usable weekly pattern, so adding them would only make the table bigger.
"""


def discretize_inventory(level, capacity=3000):
    frac = level / capacity
    if level <= 0:
        return 'stockout'
    elif frac < 0.15:
        return 'very_low'
    elif frac < 0.35:
        return 'low'
    elif frac < 0.6:
        return 'medium'
    elif frac < 0.85:
        return 'high'
    else:
        return 'overstock'


def discretize_pending(pending_qty):
    if pending_qty == 0:
        return 'none'
    elif pending_qty < 1000:
        return 'some'
    else:
        return 'a_lot'


def state_to_key(state, capacity=3000):
    return (
        discretize_inventory(state['inventory_level'], capacity),
        discretize_pending(state['pending_orders_intransit']),
    )


def supplier_state_key(state, capacity=3000):
    """The supplier agent only looks at how urgent the stock situation is."""
    return (discretize_inventory(state['inventory_level'], capacity),)


def logistics_state_key(state, capacity=3000):
    """The logistics agent also looks at how much is already on the way, since
    that affects whether a new shipment needs to be fast."""
    return (
        discretize_inventory(state['inventory_level'], capacity),
        discretize_pending(state['pending_orders_intransit']),
    )
