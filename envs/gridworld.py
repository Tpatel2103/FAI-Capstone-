import numpy as np


class GridWorld:

    ROWS = 5
    COLS = 5
    N_ACTIONS = 4
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES  = ["Up", "Down", "Left", "Right"]
    ACTION_ARROWS = ["↑", "↓", "←", "→"]

    START = (0, 0)
    GOAL  = (4, 4)
    TRAPS = frozenset({(1, 2), (2, 4), (3, 1)})
    WALLS = frozenset({(1, 1), (2, 2), (3, 3)})

    R_GOAL = 10.0
    R_TRAP = -5.0
    R_STEP = -0.1

    # __init__
    def __init__(self, stochastic=False, slip_prob=0.8, seed=42):
        self.stochastic = stochastic
        self.slip_prob  = slip_prob
        self.rng        = np.random.default_rng(seed)
        self._state     = self.START
        self._valid_states = [
            (r, c) for r in range(self.ROWS) for c in range(self.COLS)
            if (r, c) not in self.WALLS
        ]
        self.n_states  = len(self._valid_states)
        self.n_actions = self.N_ACTIONS

    # reset
    def reset(self):
        self._state = self.START
        return self._state

    # step
    def step(self, action):
        assert 0 <= action < self.N_ACTIONS, f"Invalid action {action}"
        if self.stochastic and self.rng.random() > self.slip_prob:
            actual_action = int(self.rng.integers(0, self.N_ACTIONS))
        else:
            actual_action = action
        dr, dc = self.ACTION_DELTAS[actual_action]
        r, c   = self._state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.ROWS and 0 <= nc < self.COLS) or (nr, nc) in self.WALLS:
            nr, nc = r, c
        self._state = (nr, nc)
        reward      = self._reward(self._state)
        done        = self._state == self.GOAL or self._state in self.TRAPS
        return self._state, reward, done, {"actual_action": actual_action}

    # _reward
    def _reward(self, state):
        if state == self.GOAL:  return self.R_GOAL
        if state in self.TRAPS: return self.R_TRAP
        return self.R_STEP

    # state property
    @property
    def state(self):
        return self._state

    # valid_states property
    @property
    def valid_states(self):
        return self._valid_states

    # cell_type
    def cell_type(self, r, c):
        s = (r, c)
        if s in self.WALLS:  return "wall"
        if s == self.GOAL:   return "goal"
        if s in self.TRAPS:  return "trap"
        if s == self.START:  return "start"
        return "empty"

    # render_ascii
    def render_ascii(self, Q=None):
        lines  = []
        border = "+" + ("----+" * self.COLS)
        lines.append(border)
        for r in range(self.ROWS):
            row = "|"
            for c in range(self.COLS):
                ct = self.cell_type(r, c)
                if   ct == "wall":  row += " ## |"
                elif ct == "goal":  row += "  G |"
                elif ct == "trap":  row += "  T |"
                elif ct == "start":
                    sym = self.ACTION_ARROWS[int(np.argmax(Q[r, c]))] if Q is not None else "S"
                    row += f"  {sym} |"
                else:
                    sym = self.ACTION_ARROWS[int(np.argmax(Q[r, c]))] if Q is not None else "."
                    row += f"  {sym} |"
            lines.append(row)
            lines.append(border)
        return "\n".join(lines)

    # __repr__
    def __repr__(self):
        mode = f"stochastic(p={self.slip_prob})" if self.stochastic else "deterministic"
        return f"GridWorld(5x5, {mode})"
