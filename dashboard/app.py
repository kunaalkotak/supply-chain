"""
Streamlit dashboard for the supply chain RL project.

Start it from the project root:
    streamlit run dashboard/app.py

It loads the Q-tables saved by `python training/train_multi_agent.py --save`,
so nothing is trained when the page opens. The sliders change the environment
(supplier reliability, demand volatility) but the agents keep the tables they
already learned, which shows how the trained policy copes with conditions it
was not trained on.
"""
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import statistics as stats
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment.supply_chain_env import SupplyChainEnvV0
from environment.discretization import state_to_key, supplier_state_key, logistics_state_key
from agents.q_learning import QLearningAgent
from training.train_multi_agent import (
    INVENTORY_ACTIONS, SUPPLIER_ACTIONS, LOGISTICS_ACTIONS, CALIB_PATH, QTABLE_PATH
)

EPISODE_DAYS = 90

st.set_page_config(page_title="Supply Chain RL Dashboard", layout="wide")

ACTION_LABELS = {
    'no_reorder': 'Do not reorder',
    'reorder_500': 'Reorder 500 units',
    'reorder_1000': 'Reorder 1,000 units',
    'reorder_1500': 'Reorder 1,500 units',
    'select_A': 'Supplier A (reliable, pricier)',
    'select_B': 'Supplier B (cheaper, less reliable)',
    'split_order': 'Split the order between A and B',
    'Standard Class': 'Standard shipping',
    'Second Class': 'Second-class shipping',
    'First Class': 'First-class shipping',
    'Same Day': 'Same-day shipping',
}

INVENTORY_LABELS = {
    'stockout': 'out of stock',
    'very_low': 'very low stock',
    'low': 'low stock',
    'medium': 'medium stock',
    'high': 'high stock',
    'overstock': 'overstocked',
}
PENDING_LABELS = {
    'none': 'nothing on order',
    'some': 'a small amount on order',
    'a_lot': 'a lot on order',
}

POLICY_NAMES = {
    'learned': 'Learned agents (Q-learning)',
    'fixed': 'Fixed rule',
    'random': 'Random',
}


# ---------------------------------------------------------------- helpers

@st.cache_resource
def load_trained_agents():
    with open(QTABLE_PATH) as f:
        qtables = json.load(f)

    def build_agent(actions, serialized_q):
        agent = QLearningAgent(actions, epsilon_start=0.0, epsilon_min=0.0, seed=0)
        for k, v in serialized_q.items():
            agent.Q[tuple(k.split("|"))] = v
        return agent

    return (build_agent(INVENTORY_ACTIONS, qtables['inventory']),
            build_agent(SUPPLIER_ACTIONS, qtables['supplier']),
            build_agent(LOGISTICS_ACTIONS, qtables['logistics']))


def make_env(rel_a, rel_b, volatility, seed=None):
    return SupplyChainEnvV0(
        CALIB_PATH, max_steps=EPISODE_DAYS,
        supplier_overrides={'A': {'reliability': rel_a}, 'B': {'reliability': rel_b}},
        demand_volatility_multiplier=volatility,
        seed=seed,
    )


def fixed_rule_actions(state, threshold=400):
    """Reorder 1,000 units whenever stock is below the threshold. Always
    Supplier A, always standard shipping."""
    inv_action = 'no_reorder' if state['inventory_level'] >= threshold else 'reorder_1000'
    return inv_action, 'select_A', 'Standard Class'


def policy_actions(policy, state, agents, rng=None):
    if policy == 'learned':
        inv_agent, sup_agent, log_agent = agents
        return (
            inv_agent.choose_action(state_to_key(state), greedy=True),
            sup_agent.choose_action(supplier_state_key(state), greedy=True),
            log_agent.choose_action(logistics_state_key(state), greedy=True),
        )
    if policy == 'fixed':
        return fixed_rule_actions(state)
    return (rng.choice(INVENTORY_ACTIONS), rng.choice(SUPPLIER_ACTIONS), rng.choice(LOGISTICS_ACTIONS))


def describe_situation(state):
    inv, pending = state_to_key(state)
    return f"{INVENTORY_LABELS[inv]}, {PENDING_LABELS[pending]}"


def run_policy_episode(env, policy, agents, rng):
    """Play one full episode and return a few summary numbers."""
    state = env.reset()
    total, stockout_days, fulfilled, demand = 0.0, 0, 0, 0
    done = False
    while not done:
        actions = policy_actions(policy, state, agents, rng)
        state, reward, done, info = env.step(*actions)
        total += reward
        stockout_days += int(info['stockout_units'] > 0)
        fulfilled += info['fulfilled']
        demand += info['demand']
    return {'reward': total, 'stockout_days': stockout_days,
            'fill_rate': fulfilled / demand if demand else 1.0}


def compare_policies(n_episodes, rel_a, rel_b, volatility):
    agents = load_trained_agents()
    results = {}
    for policy in ('learned', 'fixed', 'random'):
        env = make_env(rel_a, rel_b, volatility, seed=1000)
        rng = random.Random(7)
        results[policy] = [run_policy_episode(env, policy, agents, rng) for _ in range(n_episodes)]
    return results


@st.cache_data
def time_in_each_situation(n_episodes=100):
    """How often the trained policy ends up in each situation under the
    default conditions. Used to flag table rows that are rarely or never seen."""
    agents = load_trained_agents()
    env = make_env(0.90, 0.70, 1.0, seed=1000)
    inv_counts, sup_counts, total = Counter(), Counter(), 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            inv_counts[state_to_key(state)] += 1
            sup_counts[supplier_state_key(state)] += 1
            total += 1
            state, _, done, _ = env.step(*policy_actions('learned', state, agents))
    return ({k: v / total for k, v in inv_counts.items()},
            {k: v / total for k, v in sup_counts.items()})


def preferred_action_table(agent, shares, label_fn):
    rows = []
    for key, q_values in agent.Q.items():
        best = max(q_values, key=q_values.get)
        rows.append({'Situation': label_fn(key),
                     'Preferred action': ACTION_LABELS[best],
                     'share': shares.get(key, 0.0)})
    df = pd.DataFrame(rows).sort_values('share', ascending=False)
    df['Time spent here'] = df['share'].map(lambda x: f"{x:.1%}")
    return df[['Situation', 'Preferred action', 'Time spent here']]


def q_value_table(agent):
    df = pd.DataFrame({'|'.join(k): v for k, v in agent.Q.items()}).T.round(1)
    df.index.name = 'State'
    return df


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Settings")

policy_label = st.sidebar.selectbox(
    "Who makes the decisions?",
    [POLICY_NAMES['learned'], POLICY_NAMES['fixed']],
    help="The learned agents are the three trained Q-learning agents. The fixed "
         "rule reorders 1,000 units whenever stock falls below 400, always "
         "uses Supplier A and always ships standard class.",
)
policy = 'learned' if policy_label == POLICY_NAMES['learned'] else 'fixed'

st.sidebar.subheader("Conditions")
rel_a = st.sidebar.slider(
    "Supplier A reliability", 0.3, 1.0, 0.90, 0.05,
    help="Chance that an order given to Supplier A is actually delivered. "
         "The agents were trained with 0.90.")
rel_b = st.sidebar.slider(
    "Supplier B reliability", 0.3, 1.0, 0.70, 0.05,
    help="Same for Supplier B. The agents were trained with 0.70.")
volatility = st.sidebar.slider(
    "Demand volatility", 0.5, 2.5, 1.0, 0.1,
    help="1.0 is the real day-to-day variation in the DataCo sales data. "
         "Higher values make demand swing more.")
st.sidebar.caption("Changing any setting above restarts the simulation.")

st.sidebar.subheader("Run")
n_days_to_run = st.sidebar.number_input("Days to advance", min_value=1, max_value=EPISODE_DAYS, value=1)
run_clicked = st.sidebar.button("Advance", type="primary")
run_full_clicked = st.sidebar.button(f"Run to day {EPISODE_DAYS}")
reset_clicked = st.sidebar.button("Restart")

# ---------------------------------------------------------------- session state

config_key = (policy, rel_a, rel_b, volatility)
if reset_clicked or st.session_state.get('config_key') != config_key:
    st.session_state.config_key = config_key
    st.session_state.env = make_env(rel_a, rel_b, volatility)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.history = []
    st.session_state.last_decision = None

agents = load_trained_agents()
env = st.session_state.env


def step_once():
    state = st.session_state.state
    if env.done:
        return
    actions = policy_actions(policy, state, agents)
    next_state, reward, done, info = env.step(*actions)
    st.session_state.last_decision = {
        'inventory': actions[0], 'supplier': actions[1], 'logistics': actions[2],
        'situation': describe_situation(state),
    }
    st.session_state.history.append({
        'day': state['time_step'], 'inventory': state['inventory_level'],
        'demand': info['demand'], 'fulfilled': info['fulfilled'],
        'stockout_units': info['stockout_units'], 'reward': reward,
        'ordered': env.REORDER_QTY[actions[0]],
    })
    st.session_state.state = next_state


if run_clicked:
    for _ in range(int(n_days_to_run)):
        step_once()
if run_full_clicked:
    while not env.done:
        step_once()

# ---------------------------------------------------------------- page header

st.title("Supply Chain Decision Dashboard")
st.write(
    "Three agents run the stock for one product: one decides how much to reorder, "
    "one picks the supplier, and one picks the shipping method. They learned by "
    "trial and error in a simulation that uses real demand and delivery data. "
    "Here you can watch them work, break the conditions, and compare them with "
    "simpler approaches."
)

with st.expander("How to use this page"):
    st.markdown(
        "1. Use the sidebar to pick who makes decisions. **Learned agents** are the trained "
        "ones; **Fixed rule** is a simple comparison.\n"
        "2. Press **Advance** to move forward day by day, or **Run to day 90** for a whole episode.\n"
        "3. Drag the sliders to make suppliers less reliable or demand more erratic, and see "
        "how each approach copes.\n"
        "4. The **Compare policies** tab tests all approaches over many episodes at once.\n"
        "5. The **What the agents learned** tab shows the rules they ended up with."
    )

with st.expander("What do the terms mean?"):
    st.markdown(
        "- **Inventory level**: units currently in the warehouse (capacity 3,000).\n"
        "- **Pending (in transit)**: units already ordered but not yet arrived.\n"
        "- **Stockout**: customers wanted more than the warehouse had, so some sales were lost.\n"
        "- **Reward**: the score the agents try to maximise. Units sold add points; stockouts, "
        "excess stock, late deliveries, shipping cost and supplier failures take points away.\n"
        "- **Fulfillment rate**: share of customer demand that was actually met."
    )

tab_live, tab_compare, tab_learned = st.tabs(
    ["Live simulation", "Compare policies", "What the agents learned"])

# ---------------------------------------------------------------- tab 1: live simulation

with tab_live:
    state = st.session_state.state

    if env.done:
        st.success(f"Day {EPISODE_DAYS} reached. Press Restart in the sidebar to run it again.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day", f"{state['time_step']} / {EPISODE_DAYS}")
    c2.metric("Units in warehouse", f"{state['inventory_level']:,}")
    c3.metric("Units on order", f"{state['pending_orders_intransit']:,}")
    c4.metric("Out of stock?", "Yes" if state['stockout_flag'] else "No")

    st.subheader("Latest decisions")
    if st.session_state.last_decision:
        d = st.session_state.last_decision
        a, b, c = st.columns(3)
        a.info(f"**Inventory agent**\n\n{ACTION_LABELS[d['inventory']]}")
        b.info(f"**Supplier agent**\n\n{ACTION_LABELS[d['supplier']]}")
        c.info(f"**Logistics agent**\n\n{ACTION_LABELS[d['logistics']]}")
        note = f"Situation when these were chosen: {d['situation']}."
        if d['inventory'] == 'no_reorder':
            note += " No order was placed, so the supplier and shipping choices had no effect that day."
        st.caption(note)
    else:
        st.write("Nothing has happened yet. Press Advance in the sidebar to start.")

    st.subheader("How it is going")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        total_reward = hist_df['reward'].sum()
        stockout_days = int((hist_df['stockout_units'] > 0).sum())
        unmet_units = int(hist_df['stockout_units'].sum())
        total_demand = hist_df['demand'].sum()
        fulfillment_rate = hist_df['fulfilled'].sum() / total_demand if total_demand > 0 else 1.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reward so far", f"{total_reward:,.0f}")
        m2.metric("Days with a stockout", stockout_days)
        m3.metric("Units of demand missed", f"{unmet_units:,}")
        m4.metric("Fulfillment rate", f"{fulfillment_rate * 100:.1f}%")

        left, right = st.columns(2)
        with left:
            st.caption("Units in the warehouse at the start of each day")
            st.line_chart(hist_df.set_index('day')[['inventory']])
        with right:
            st.caption("Reward earned each day")
            st.line_chart(hist_df.set_index('day')[['reward']])

        with st.expander("Day-by-day table"):
            table = hist_df.rename(columns={
                'day': 'Day', 'inventory': 'Stock at start', 'demand': 'Demand',
                'fulfilled': 'Sold', 'stockout_units': 'Missed', 'reward': 'Reward',
                'ordered': 'Units ordered'})
            st.dataframe(table.round(1), hide_index=True)
    else:
        st.write("Charts and totals will appear here once the simulation has run.")

# ---------------------------------------------------------------- tab 2: compare policies

with tab_compare:
    st.write(
        "Plays several full 90-day episodes with each approach under the conditions "
        "currently set in the sidebar. Each approach starts from the same random seed, "
        "so this is a fair side-by-side test."
    )
    n_episodes = st.slider("Episodes per approach", 10, 200, 50, 10)

    if st.button("Run comparison"):
        with st.spinner("Running episodes..."):
            st.session_state.comparison = {
                'results': compare_policies(n_episodes, rel_a, rel_b, volatility),
                'settings': f"Supplier A {rel_a:.2f}, Supplier B {rel_b:.2f}, "
                            f"demand volatility {volatility:.1f}, {n_episodes} episodes each",
            }

    if 'comparison' in st.session_state:
        results = st.session_state.comparison['results']
        st.caption("Results for: " + st.session_state.comparison['settings'])

        summary = []
        for pid in ('learned', 'fixed', 'random'):
            rewards = [r['reward'] for r in results[pid]]
            summary.append({
                'Approach': POLICY_NAMES[pid],
                'Average reward': stats.mean(rewards),
                'Spread (std)': stats.stdev(rewards),
                'Avg. stockout days': stats.mean(r['stockout_days'] for r in results[pid]),
                'Avg. fulfillment': stats.mean(r['fill_rate'] for r in results[pid]),
            })
        summary_df = pd.DataFrame(summary)

        fig, ax = plt.subplots(figsize=(6, 3.4))
        colors = ['#1f5fa8', '#8a8a8a', '#b9b9b9']
        ax.bar(summary_df['Approach'], summary_df['Average reward'],
               yerr=summary_df['Spread (std)'], capsize=5, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Reward per episode')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        shown = summary_df.copy()
        shown['Average reward'] = shown['Average reward'].map('{:,.0f}'.format)
        shown['Spread (std)'] = shown['Spread (std)'].map('{:,.0f}'.format)
        shown['Avg. stockout days'] = shown['Avg. stockout days'].map('{:.1f}'.format)
        shown['Avg. fulfillment'] = shown['Avg. fulfillment'].map('{:.1%}'.format)
        st.dataframe(shown, hide_index=True)
        st.caption(
            "Reward counts costs as well as sales, so an approach can meet almost all demand and "
            "still score badly. Random does this: it orders far too much, which costs a lot in "
            "storage and shipping."
        )
        st.caption(
            "Tip: the agents were trained with Supplier A at 0.90, Supplier B at 0.70 and normal "
            "demand. Try lowering both reliabilities in the sidebar and running the comparison again."
        )

# ---------------------------------------------------------------- tab 3: what the agents learned

with tab_learned:
    st.write(
        "Each agent keeps a table that scores every action in every situation it has met. "
        "Once trained, it simply picks the highest-scoring action. These tables show the "
        "favourite action for each situation, and how much of the time the trained agents "
        "actually spend in that situation."
    )
    st.caption(
        "Situations with a very small share of time were rarely visited during training, so "
        "their preferences are based on little experience and should not be over-read."
    )

    inv_agent, sup_agent, log_agent = agents
    inv_shares, sup_shares = time_in_each_situation()

    def inv_label(key):
        return f"{INVENTORY_LABELS[key[0]].capitalize()}, {PENDING_LABELS[key[1]]}"

    def sup_label(key):
        return INVENTORY_LABELS[key[0]].capitalize()

    st.subheader("Inventory agent")
    st.caption("Decides how much to reorder.")
    st.dataframe(preferred_action_table(inv_agent, inv_shares, inv_label),
                 hide_index=True)

    st.subheader("Supplier agent")
    st.caption("Decides which supplier gets the order. Only matters on days when something is ordered.")
    st.dataframe(preferred_action_table(sup_agent, sup_shares, sup_label),
                 hide_index=True)

    st.subheader("Logistics agent")
    st.caption("Decides the shipping method. Only matters on days when something is ordered.")
    st.dataframe(preferred_action_table(log_agent, inv_shares, inv_label),
                 hide_index=True)

    with st.expander("Raw Q-values"):
        st.write("Inventory agent")
        st.dataframe(q_value_table(inv_agent))
        st.write("Supplier agent")
        st.dataframe(q_value_table(sup_agent))
        st.write("Logistics agent")
        st.dataframe(q_value_table(log_agent))

# ---------------------------------------------------------------- footer

st.divider()
st.caption(
    "Demand and shipping-delay figures come from the real DataCo Smart Supply Chain dataset. "
    "Suppliers, inventory, warehouse size and costs are simulated assumptions, since DataCo "
    "has no data on them."
)
