import numpy as np


class DoubleQLearningAgent:

    NAME  = "Double Q-Learning"
    COLOR = "#d73027"

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
        self.QA            = np.zeros((n_rows, n_cols, n_actions), dtype=np.float64)
        self.QB            = np.zeros((n_rows, n_cols, n_actions), dtype=np.float64)
        self.rewards_per_episode = []
        self.steps_per_episode   = []
        self.max_delta_q         = []
        self.td_errors           = []
        self._update_A           = 0
        self._update_B           = 0

    # Q property — combined average of QA and QB
    @property
    def Q(self):
        return (self.QA + self.QB) / 2.0

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
        if self.rng.random() < 0.5:
            if done:
                target = reward
            else:
                best_a = int(np.argmax(self.QA[nr, nc]))
                target = reward + self.gamma * self.QB[nr, nc, best_a]
            td_err = target - self.QA[r, c, action]
            self.QA[r, c, action] += self.alpha * td_err
            self._update_A += 1
        else:
            if done:
                target = reward
            else:
                best_b = int(np.argmax(self.QB[nr, nc]))
                target = reward + self.gamma * self.QA[nr, nc, best_b]
            td_err = target - self.QB[r, c, action]
            self.QB[r, c, action] += self.alpha * td_err
            self._update_B += 1
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
        self.QA = np.zeros((self.n_rows, self.n_cols, self.n_actions), dtype=np.float64)
        self.QB = np.zeros((self.n_rows, self.n_cols, self.n_actions), dtype=np.float64)
        self.rewards_per_episode.clear()
        self.steps_per_episode.clear()
        self.max_delta_q.clear()
        self.td_errors.clear()
        self._update_A = 0
        self._update_B = 0

    # __repr__
    def __repr__(self):
        return f"DoubleQLearningAgent(alpha={self.alpha}, gamma={self.gamma}, eps={self.epsilon:.4f})"
