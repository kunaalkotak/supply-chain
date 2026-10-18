# AI-Powered Supply Chain Risk Prediction and Autonomous Decision-Making System

## 1. Abstract

This project develops a simulated multi-agent supply-chain system in which
three reinforcement-learning agents — Inventory, Supplier, and Logistics —
learn coordinated decisions via Independent Q-Learning. The simulation's
demand and shipping-delay dynamics are calibrated from the real DataCo
Smart Supply Chain dataset (180,519 orders, 2015–2018), while inventory
and supplier mechanics are explicit, clearly labeled simulated
assumptions, since no such data exists in DataCo. The learned multi-agent policy achieves a mean episode reward of 6,434.17 versus
2,596.45 for a fixed heuristic baseline and −1,032.85 for random behavior, and
six controlled experiments examine exploration rate, supplier reliability,
demand volatility, agent count, reward structure, and RL-vs-baseline
comparisons. All numbers in this report come from the code in this repository,
and the inconclusive results are included.

## 2. Introduction

Modern supply chains involve continuous, interacting decisions: when to
reorder stock, which supplier to use, and how to ship. This project
treats that problem as a cooperative Markov Game solved with Independent
Q-Learning, built around a real, verified dataset used strictly to
calibrate a custom simulation rather than as the environment itself.

## 3. Problem Statement

A single flat policy, or three independently trained models with no
interaction, cannot capture the fact that an urgent reorder should
influence supplier choice and shipping speed simultaneously. The project
requires genuine cross-agent interaction, not three decoupled predictors.

## 4. Objectives

1. Build a custom RL environment whose stochastic dynamics are grounded
   in real data wherever real data exists.
2. Implement three interacting agents under Independent Q-Learning.
3. Demonstrate measurable learning (reward improves over training).
4. Compare RL against baseline heuristics and random behavior.
5. Run controlled experiments and report all results, including the inconclusive ones.
6. Deliver a working dashboard for live demonstration.

## 5. Literature Review

Q-learning (Watkins & Dayan, 1992) is a model-free method that learns the
value of each action in each state from experience; Sutton & Barto (2018) give
a full treatment. When several agents learn in the same environment, the
simplest approach is to let each run its own Q-learning and treat the others
as part of the environment. This is independent Q-learning, studied by Tan
(1993). Because the other agents are also changing, each agent faces a moving
target, and Claus & Boutilier (1998) examine how this plays out in cooperative
settings where all agents share one reward, which is the setting used here.
Littman (1994) introduced Markov games as the general framework for this kind
of problem.

Reinforcement learning has been applied to supply-chain inventory decisions
since at least Giannoccaro & Pontrandolfo (2002), who modelled inventory
ordering across the stages of a supply chain as a Markov decision process.
This project follows the same idea on a smaller scale and adds two further
decisions, supplier choice and shipping mode, handled by separate agents.

## 6. Dataset

**DataCo Smart Supply Chain for Big Data Analysis** — verified directly:
180,519 rows, 53 columns, real order-item-level e-commerce transactions
from DataCo Global, spanning January 2015 to January 2018, CC0
(public-domain) licensed, sourced from Kaggle/Mendeley Data.

Key columns used: `order date (DateOrders)`, `Order Item Quantity`,
`Shipping Mode`, `Days for shipping (real)`, `Days for shipment
(scheduled)`, `Late_delivery_risk`, `Delivery Status`, `Product Name`,
`Product Price`, `Order Region`/`Market`/`Customer Segment`.

**Confirmed absent from the dataset** (verified by inspecting all 53
column names, not assumed): supplier identity, supplier reliability,
supplier lead time, supplier capacity, supplier cost, current inventory
level, warehouse capacity, holding cost, and transportation cost in
currency terms. These are simulated in the environment (Section 8) and
never presented as dataset-derived.

## 7. System Architecture

```
DataCo dataset --calibrates--> SupplyChainEnvV0 (custom simulation)
                                        |
                --------------------------------------------------
                |                     |                      |
        Inventory Agent        Supplier Agent          Logistics Agent
         (Q-table)              (Q-table)                (Q-table)
                |                     |                      |
                ------ joint action --> env.step() -----------
                                        |
                          shared reward + next state
```

## 8. Environment Design

`SupplyChainEnvV0`: single product (*Perfect Fitness Perfect Rip Deck*,
the highest-volume product in the dataset — 73,698 units across 24,515
orders), 2 simulated suppliers, 1 simulated warehouse, 4 real shipping
modes, day-level time steps, 90-day episodes.

**Real, derived calibration statistics used directly in the
environment's transition function:**

| Shipping Mode | Real delay probability | Real avg. duration |
|---|---|---|
| Standard Class | 0.3807 | 4.00 days |
| Second Class | 0.7663 | 3.99 days |
| First Class | 0.9532 | 2.00 days |
| Same Day | 0.4574 | 0.48 days |

Demand is sampled by bootstrap resampling from the real 1,006-day
historical daily-quantity pool for this product (mean 73.26, std 16.72) —
not a fitted trend model, because a linear day-of-week + trend model was
tested (Section 15) and found to add no explanatory power over simply
using the empirical distribution.

**Simulated parameters** (explicit assumptions, no real source):
Supplier A: reliability 0.90, lead time 3 days, unit cost $12.0. Supplier
B: reliability 0.70, lead time 6 days, unit cost $9.5. Warehouse
capacity: 3,000 units. Initial inventory: 800 units. Holding cost: 0.1
per excess unit per day.

**Validation:** 11 automated unit tests cover reset validity, correct
`step()` return shape, correct supplier/mode attribution, inventory
monotonicity under no reorders, capacity enforcement, correct episode
termination, and confirmation that sampled demand values always come
from the real calibration pool. All 11 pass.

## 9. Agent Design

| Agent | Observation | Action space |
|---|---|---|
| Inventory | (inventory bin, pending-orders bin) | no_reorder, reorder_500, reorder_1000, reorder_1500 |
| Supplier | (inventory urgency bin) | select_A, select_B, split_order |
| Logistics | (inventory urgency bin, pending-orders bin) | Standard Class, Second Class, First Class, Same Day |

**Design note:** Supplier and Logistics observations are built only from
the shared environment state, not from the current step's order size, to
avoid a circular dependency in the Q-learning bootstrap target (the next
state for those agents would otherwise depend on an action not yet
chosen). Genuine interaction is still enforced through the environment's
transition and shared reward, not through the observation function.

## 10. MDP Formulation

- **State**: inventory level, pending in-transit quantity, day index, stockout flag
- **Action**: per-agent (Section 9)
- **Reward**: Section 11
- **Transition**: real demand realization decrements inventory; joint
  agent actions determine order creation, supplier reliability draw
  (simulated), and shipping delay draw (real probability); arrivals
  replenish inventory
- **γ = 0.9**
- **Policy**: a learned mapping from discretized state to the
  highest-value discrete action per agent

## 11. Markov Game Formulation

- **N = 3**, cooperative (single shared/global reward in the primary
  configuration; Experiment 6 tests a hybrid local+global alternative)
- **O_i, A_i**: per Section 9
- **R_i = R (shared)**: `w1·fulfilled − w2·stockout − w3·holding_cost·excess − w4·delay − w5·transport_cost − w6·supplier_failure`
- **P**: joint transition function combining all three agents' actions each step

## 12. Reward Design

Initial weights (`w1..w6` = 1.0, 2.0, 0.05, 1.5, 0.01, 1.0) produced a
flat training curve — mean episode reward moved only from 6,131.75 to
6,201.17 over 3,000 episodes (~1%), because the holding-cost term worked
out to roughly 0.001 per excess unit, negligible next to per-day
fulfillment reward (~70). The holding weight was therefore corrected to `w3 = 1.0` with `holding_cost_per_unit =
0.1` (effective 0.1 per excess unit), after which training showed a steady improvement (Section 14). The before/after is kept here because it is a useful result about reward design in its own right.

## 13. Q-Learning / Independent Q-Learning

Implemented from scratch: `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') −
Q(s,a)]`, α=0.1, γ=0.9, ε-greedy exploration decaying 1.0→0.05 (decay
factor 0.998 in the primary configuration). Each agent maintains an
independent Q-table with no shared value function or communicated
Q-values — coordination emerges solely through the shared reward signal
and the environment's coupled dynamics.

## 14. Implementation & Verification of Learning

**Single-agent (Inventory Agent only, Supplier/Logistics fixed
to Supplier A / Standard Class):** after the reward-weight correction,
mean reward per 300-episode training chunk rose from **−248.05** (episodes
0–299) to **4,616.73–5,011.89** (episodes 2,400–2,999) — a clear,
monotonic-with-noise improvement. Final greedy evaluation (200 episodes):
**6,267.59 ± 331.52**, versus 2,596.45 ± 526.12 for a naive threshold
policy and −413.08 ± 262.63 for random action selection.

**Multi-agent (all three agents learning):** mean reward per
400-episode chunk rose from **−470** to **~4,800** over 4,000 episodes.
Final greedy evaluation: **6,434.17 ± 169.16**, versus 2,596.45 ± 526.12
(fixed heuristic) and −1,032.85 ± 343.71 (fully random). The multi-agent
result modestly exceeds the single-agent result, consistent with real
added value from letting Supplier and Logistics also learn.

![Learning curve](../results/learning_curve.png)

*Figure 1. Reward per training episode for the three-agent system. Training
reward sits below the final test reward because exploration (random actions)
continues throughout training.*

![Policy comparison](../results/policy_comparison.png)

*Figure 2. Mean reward per 90-day episode over 200 test episodes, with one
standard deviation.*

## 15. Experimental Setup and Results

*(Baseline ML models, used only to calibrate the simulation, not as part
of the RL decision loop.)*

**Delivery-risk classifier** (Logistic Regression, chronological
train/test split: 144,415 orders train / 36,104 test): Accuracy 0.6921,
Precision 0.8421, Recall 0.5373, Specificity 0.8786, F1 0.6560, ROC-AUC
0.7289, versus a 0.5466 majority-class baseline. `Shipping Mode` was
confirmed as the dominant predictor. `Delivery Status` and `Days for
shipping (real)` were excluded after confirming they are deterministic
restatements of the label (100% and ~97.5% match respectively) — a
leakage check performed explicitly, not assumed clean.

**Demand forecasting** (top product, chronological split, 855 train /
151 test days): naive persistence MAE 17.74; 7-day moving average MAE
13.68; linear regression (day-of-week + trend) MAE 12.97, but **R² =
−0.0496** — no better than a predict-the-mean baseline (MAE 12.90, R² =
−0.0294). This indicates the demand series behaves close to
noise around a slowly drifting level in this window, which is why the
environment uses empirical resampling rather than a fitted forecast.

**RL experiments** — each condition is a single training run
(one seed), not averaged across seeds; this limitation is stated
explicitly rather than implying more statistical rigor than exists:

| Experiment | Result | Verdict |
|---|---|---|
| 1. Exploration rate | Fast (0.99): 6,335.80±153 · Default (0.998): 6,464.68±176 · Slow (0.9995): 4,547.80±315 | Real effect — slow decay underperforms given the training budget |
| 3. Supplier reliability | Baseline: 6,464.68±176 · Unreliable (.60/.50): 5,616.48±348 · Very reliable (.98/.95): 6,358.40±131 | Unreliability clearly hurts (real effect); higher-than-baseline reliability showed no significant gain (within noise) |
| 4. Demand volatility | 1.0x: 6,464.68±176 · 1.5x: 6,219.45±279 · 2.0x: 6,351.89±279 | Inconclusive — non-monotonic pattern most likely reflects single-run training variance |
| 5. Agent count | Single-agent: 6,267.59±332 · Multi-agent: 6,434.17±169 | Real, modest improvement from full MAS |
| 6. Shared vs. hybrid reward | Shared: 6,464.68±176 · Hybrid: 6,077.97±386 | Real — hybrid reward performed worse and noisier in this tightly-coupled small system |
| 7. RL vs. baselines | RL: 6,464.68±176 · Fixed heuristic: 2,584.23±514 · Random: −1,052.72±335 | Strong, robust effect, consistent across all runs |

Experiment 7 is a separate run from the headline results in Section 14 (3,000
training episodes and 150 test episodes instead of 4,000 and 200, with a
different test seed), which is why its numbers differ slightly from them.

## 16. Results Summary

The learned multi-agent policy outperforms both a hand-designed
heuristic (by ~2.5x) and random behavior by a wide margin, with
verified, reproducible training curves (not single lucky evaluation
runs) demonstrating genuine learning at both the single-agent and
multi-agent stages.

## 17. Baseline Comparison

Fixed heuristic baseline (reorder below a fixed threshold, always
Supplier A, always Standard shipping): 2,584.23–2,596.45 mean reward
across evaluation runs — consistently and substantially below the
learned policy (6,267–6,464), and the RL policy's advantage held across
every experimental condition tested in Section 15.

## 18. Emergent Behavior Analysis

Observed, not hard-coded: the Supplier Agent learned to prefer the more
reliable Supplier A specifically when inventory urgency is `very_low`,
`low`, or `medium` — a reliability-under-pressure preference that was
never explicitly programmed. Under the converged multi-agent policy,
`stockout` and `overstock` states were visited **zero times** in 200
evaluation episodes (`very_low` itself only 183 times versus 9,229 for
`medium`), indicating the system learned to stay within a safe operating
band. This also means the Q-values recorded for those rare states are
based on very few training visits and should not be read as confident,
converged preferences — so they are not offered as additional findings.

## 19. Limitations

- The experiments used one training seed per condition; Experiments
  3 (partially) and 4 (fully) would benefit from multi-seed averaging
  before being treated as conclusive.
- The Supplier Agent's entire observation and dynamics are simulated
  assumptions with zero real-data grounding, unlike the Logistics
  Agent's real delay/duration statistics — this asymmetry should be
  stated plainly whenever describing what each agent "learned."
  Similarly, the demand-forecasting baseline showed no real predictive
  skill (negative R²), so the environment's demand realism rests on
  matching the real empirical distribution, not on forecasting accuracy.
- Q-tables for rare states (`stockout`, `overstock`) are undertrained;
  any claim about the agent's behavior in those states should be
  qualified.
- On about 80% of days in the evaluation runs the inventory agent chose not to
  reorder, and on those days the supplier and logistics choices have no effect.
  Their Q-values are therefore learned from the remaining ~20% of days, which
  is a thin basis for the finer differences between their actions.
- The system models a single product; a multi-product version was not
  implemented.

## 20. Future Work

Multi-seed experiment averaging; multi-product extension; richer
Supplier/Logistics state definitions (e.g., conditioning on current
order size via a restructured update to resolve the state-dependency
issue noted in Section 9); function-approximation Q-learning if the
state space grows; deployment-realistic supplier data if a real dataset
becomes available.

## 21. Conclusion

This project delivers a working cooperative multi-agent RL system for
supply-chain decision-making. The simulation is grounded in real data wherever
real data exists, and the simulated parts are labelled as such throughout.
Learning was checked at both the single-agent and multi-agent stages, and six
controlled experiments were run, including some that did not give a clear
answer.

## 22. References

Claus, C. and Boutilier, C. (1998). The dynamics of reinforcement learning in
cooperative multiagent systems. *Proceedings of the 15th National Conference
on Artificial Intelligence (AAAI-98)*, 746–752.

Constante, F., Silva, F. and Pereira, A. (2019). DataCo Smart Supply Chain for
Big Data Analysis. *Mendeley Data*, V5. https://doi.org/10.17632/8gx2fvg2k6.5

Giannoccaro, I. and Pontrandolfo, P. (2002). Inventory management in supply
chains: a reinforcement learning approach. *International Journal of Production
Economics*, 78(2), 153–161.

Littman, M. L. (1994). Markov games as a framework for multi-agent
reinforcement learning. *Proceedings of the 11th International Conference on
Machine Learning*, 157–163.

Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An
Introduction* (2nd ed.). MIT Press.

Tan, M. (1993). Multi-agent reinforcement learning: independent vs. cooperative
agents. *Proceedings of the 10th International Conference on Machine Learning*,
330–337.

Watkins, C. J. C. H. and Dayan, P. (1992). Q-learning. *Machine Learning*,
8(3–4), 279–292.
