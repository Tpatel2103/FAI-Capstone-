import numpy as np


class QLearningAgent:

    NAME  = "Q-Learning"
    COLOR = "#2166ac"

    # __init__
    def __init__(self, n_rows=5, n_cols=5, n_actions=4,
                 alpha=0.1, gamma=0.9, epsilon=0.1,
                 epsilon_min=0.01, epsilon_decay=1.0, seed=42):
        self.n_rows        = n_rows
        self.n_cols        = n_cols
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)
        self.Q             = np.zeros((n_rows, n_cols, n_actions), dtype=np.float64)
        self.rewards_per_episode = []
        self.steps_per_episode   = []
        self.max_delta_q         = []
        self.td_errors           = []

    # select_action
    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state[0], state[1]]))

    # greedy_action
    def greedy_action(self, state):
        return int(np.argmax(self.Q[state[0], state[1]]))

    # update
    def update(self, state, action, reward, next_state, done):
        r, c   = state
        nr, nc = next_state
        target = reward if done else reward + self.gamma * np.max(self.Q[nr, nc])
        td_err = target - self.Q[r, c, action]
        self.Q[r, c, action] += self.alpha * td_err
        return td_err

    # end_episode
    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # V property
    @property
    def V(self):
        return np.max(self.Q, axis=2)

    # policy property
    @property
    def policy(self):
        return np.argmax(self.Q, axis=2)

    # reset
    def reset(self):
        self.Q = np.zeros((self.n_rows, self.n_cols, self.n_actions), dtype=np.float64)
        self.rewards_per_episode.clear()
        self.steps_per_episode.clear()
        self.max_delta_q.clear()
        self.td_errors.clear()

    # __repr__
    def __repr__(self):
        return f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, eps={self.epsilon:.4f})"
