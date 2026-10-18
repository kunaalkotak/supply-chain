# Presentation Companion: Speaking Script, Demo Flow, and Q&A Prep

This accompanies `presentation.pptx` (17 slides, speaker notes embedded in
each slide). Use this document for deeper backup material and rehearsal.

## Speaking Script (expanded)

**Slide 1 — Title.** "Good [morning/afternoon]. We're presenting our AAI
mini-project: an AI-powered supply chain risk prediction and autonomous
decision-making system. This is a multi-agent reinforcement learning
project — three agents that learn to coordinate, not a standard
prediction model."

**Slide 2 — Problem.** "Real supply chains face three coupled decisions
every day: when to reorder, which supplier to use, and how to ship. These
aren't independent — an urgent reorder should pull toward a reliable
supplier and fast shipping. A flat policy, or three separate models,
can't represent that coupling."

**Slide 3 — Objectives.** Walk through the six objectives, pausing on "report all results, including inconclusive ones". Mention it early because it sets up the experiments slide.

**Slide 4 — Architecture.** "Real DataCo statistics calibrate our
simulation. Three agents observe it, act, and get one shared reward —
that shared reward is what makes this a cooperative Markov Game."

**Slide 5 — Agents.** Read each agent's observation/action/goal briefly;
note the Supplier and Logistics agents deliberately don't see the
current order size directly, to avoid a circular Q-learning dependency —
interaction still happens through the environment and shared reward.

**Slide 6 — Real vs. Simulated.** This is the most important slide for explaining what is real. "We verified column-by-column that DataCo has zero supplier data
— so any claim about learned supplier behavior is about a simulated
assumption, not something the dataset taught the model. Demand and
shipping delay, by contrast, are real derived statistics."

**Slide 7–8 — MDP / Markov Game.** Walk the tables top to bottom;
emphasize the Interaction row in the Markov Game table as the concrete
mechanism, not just terminology.

**Slide 9 — Q-Learning.** Read the equation left to right, defining each
symbol. State plainly: "we implemented this update rule ourselves — no
library performs the learning."

**Slide 10 — Learning curve.** "This is the direct answer to 'did it
learn.' Early training averages −470; by the end it's consistently
around 4,800. We checked this progression explicitly rather than trusting
a single final number."

**Slide 11 — Results bar chart.** "6,434 for our learned policy vs. 2,596
for a hand-designed heuristic vs. −1,033 for random. This gap held across
every experiment we ran."

**Slide 12 — Experiments table.** Read all six verdicts, including the
two non-clean ones (3 partial, 4 inconclusive) — say directly: "we're showing every result, including the ones that weren't clean."

**Slide 13 — Emergent behavior.** Explain the reliability-under-pressure
finding and the "never visited stockout/overstock" finding; note the
Q-value caveat for rare states.

**Slide 14 — Limitations.** Read these plainly. It is better to state the limits yourself than to have them found.

**Slide 15 — Live demo.** Transition to the actual dashboard. See Demo
Flow below.

**Slide 16 — Conclusion & Future Work.** Summarize; don't re-read every
bullet, hit the top 2–3 future-work items.

**Slide 17 — Thank you.** Open the floor for questions.

## Live Demo Flow (Dashboard)

1. Before the presentation, run `streamlit run dashboard/app.py` from the
   project root so it is already loaded when you switch over.
2. On the **Live simulation** tab, point out day 0, 800 units in the warehouse
   and nothing on order. Open "How to use this page" if you want to show the
   audience how it works.
3. Press **Advance** a few times and read the Latest decisions boxes. The line
   underneath tells you the situation the agents saw, for example "low stock,
   nothing on order". Note that on days with no reorder, the supplier and
   shipping choices have no effect.
4. Press **Run to day 90** and note the total reward and fulfillment rate.
5. Switch "Who makes the decisions?" to **Fixed rule**, run to day 90 again, and
   point out the lower reward and the stockout days. (The run restarts
   automatically when you change a setting, and demand is random each time, so
   the exact numbers differ from run to run.)
6. Open the **Compare policies** tab and press **Run comparison**. This runs
   many episodes of all three approaches from the same seed and shows the
   average reward for each.
7. Drag Supplier A reliability down to about 0.5 and run the comparison again.
   This shows how the trained agents cope with conditions they were not
   trained on.
8. If asked what the agents actually learned, open the **What the agents
   learned** tab. The "Time spent here" column shows which rows matter in
   practice.

## Likely Professor Questions & Prepared Answers

**Q: Why Q-learning and not a more advanced algorithm (DQN, PPO)?**
A: The course specifically covers Q-learning, SARSA, Expected SARSA, and
Monte Carlo methods, and requires the algorithm's mechanics to be
implemented and visible, not hidden behind a library. Tabular Independent
Q-Learning satisfies this directly and keeps the state space small enough
to inspect by hand — every Q-value in our tables can be read and
explained, which a function-approximation method would make harder to
audit for a course project.

**Q: How do you know the agents actually learned, and didn't just get
lucky on evaluation?**
A: We tracked reward across the full training run, not just the final
evaluation. The mean reward per 400-episode chunk rose from −470 to
~4,800 over 4,000 episodes — a steady trend, not a single lucky episode.

**Q: Isn't your Supplier Agent's "learning" fake, since there's no real
supplier data?**
A: It's learning within a simulated but well-defined MDP — the supplier
reliability parameters are explicit, declared assumptions, not fabricated
data claims. We flag this explicitly rather than implying DataCo taught
the model about real supplier behavior — that's why the Real vs.
Simulated slide exists.

**Q: Why did the hybrid reward perform worse than the shared reward?**
A: Our hypothesis is that in a small, tightly-coupled 3-agent system,
local penalties added credit-assignment noise without adding meaningfully
independent signal — each agent's "local" outcome was still highly
correlated with the others' actions. We report this as a finding, not
a confirmed mechanism; testing that hypothesis directly would be a good
follow-up experiment.

**Q: Why is Experiment 4 (demand volatility) inconclusive?**
A: We ran one training seed per volatility level. The result was
non-monotonic (6,464 → 6,219 → 6,352), which is more consistent with
seed-to-seed training variance than with a real volatility effect. We
reported it as it came out rather than inventing a story to explain noise.

**Q: How do you know there's no data leakage in your delivery-risk
model?**
A: We explicitly checked: `Delivery Status` maps 100% deterministically
onto the label, so it was excluded. `Days for shipping (real)` is the
value the label is computed from (97.5% match to a simple threshold), so
it was excluded too. We also checked `Order Status` and found
`CANCELED`/`SUSPECTED_FRAUD` deterministically imply zero delay risk —
confirmed it was never included as a feature.

**Q: What would you do differently with more time?**
A: Average the experiments over multiple random seeds, extend to
multiple products, and let Supplier/Logistics agents condition on the
current order size (would require restructuring the Q-learning update to
resolve a circular state-dependency we identified during design).

**Q: What's the discount factor and why 0.9?**
A: γ=0.9 was the course-recommended default; it means future rewards
matter but are discounted, so the agent doesn't act purely greedily on
immediate reward but also doesn't overweight distant, less certain
outcomes. We didn't tune γ separately in this project's experiments —
a reasonable extension.
