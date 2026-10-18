"""
Tabular Q-learning agent, written from scratch (no RL library).

Update rule:
    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

    alpha    learning rate. How far each new experience moves the old estimate.
    gamma    discount factor. How much future reward counts compared to
             immediate reward.
    epsilon  chance of trying a random action instead of the best known one,
             so the agent keeps exploring. It decays during training.
"""

import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 seed=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # Q[state][action] = estimated value. States are added the first time
        # they are seen.
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def choose_action(self, state_key, greedy=False):
        """Epsilon-greedy choice. With greedy=True it always takes the best
        known action, which is what evaluation and the dashboard use."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        q_values = self.Q[state_key]
        best_value = max(q_values.values())
        # if several actions tie, pick one of them at random
        best_actions = [a for a, v in q_values.items() if v == best_value]
        return self.rng.choice(best_actions)

    def update(self, state_key, action, reward, next_state_key, done):
        current_q = self.Q[state_key][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state_key].values())
        self.Q[state_key][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
