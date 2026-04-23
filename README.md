## A Comparative Empirical Study of Q-Learning, Sarsa and Double Q-Learning Algorithms


---

## Overview

This project reproduces the Q-learning algorithm introduced by Watkins & Dayan (1992) and compares it against two related temporal-difference control algorithms — **SARSA** and **Double Q-learning** — on a custom 5×5 grid-world environment.

The agent starts at the top-left corner and must navigate to the goal (+10 reward) while avoiding traps (−5) and walls. Experiments run in both a **deterministic** setting (actions always succeed) and a **stochastic** setting (actions succeed with probability 0.8).

The study validates Q-learning's convergence guarantee, measures overestimation bias, and compares how on-policy vs. off-policy learning behaves under noise.

---

## Algorithms Implemented

| Algorithm | Type | Description |
|---|---|---|
| **Q-Learning** | Off-policy | Learns optimal policy using greedy max target |
| **SARSA** | On-policy | Updates based on action actually taken; conservative near penalties |
| **Double Q-Learning** | Off-policy | Uses two Q-tables to eliminate overestimation bias |

---

## Project Structure

```
project/
├── envs/
│   └── gridworld.py           # 5×5 GridWorld (deterministic + stochastic)
├── agents/
│   ├── q_learning.py          # Q-Learning agent
│   ├── sarsa.py               # SARSA agent
│   └── double_q_learning.py   # Double Q-Learning agent
├── experiments/
│   ├── train.py               # Training loops + multi-seed evaluation
│   └── visualize.py           # Generates all result plots
├── results/                   # Output figures saved here (auto-created)
├── main.py                    # Entry point — runs everything
└── README.md
```

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

No GPU required. Runs on any standard laptop.

---

## How to Run

**1. Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Run with default settings:**

```bash
python main.py
```

**3. Reproduce report results exactly:**

```bash
python main.py --episodes 5000 --alpha 0.1 --gamma 0.9 --epsilon 0.1 --seed 42
```

**4. Quick run — skip hyperparameter sweep:**

```bash
python main.py --episodes 500 --skip-sweep
```

All output figures are saved automatically to the `results/` folder.

---

## CLI Options

| Argument | Default | Description |
|---|---|---|
| `--episodes` | `2000` | Training episodes per algorithm |
| `--alpha` | `0.1` | Learning rate α |
| `--gamma` | `0.9` | Discount factor γ |
| `--epsilon` | `0.1` | Exploration rate ε |
| `--seed` | `42` | Random seed for reproducibility |
| `--results-dir` | `results` | Folder to save output figures |
| `--n-seeds` | `5` | Seeds for multi-seed robustness test |
| `--skip-sweep` | `False` | Skip hyperparameter sensitivity sweep |
| `--sweep-episodes` | `1000` | Episodes per config during sweep |

---

## Output Figures

Running the script generates the following plots in `results/`:

| File | Description |
|---|---|
| `01_gridworld_layout.png` | Environment layout — both variants |
| `02_qlearning_training.png` | Q-Learning baseline: reward, convergence, steps |
| `03_learning_curves_det.png` | Reward per episode — deterministic |
| `04_learning_curves_sto.png` | Reward per episode — stochastic |
| `05_convergence_det.png` | Max \|ΔQ\| — deterministic |
| `06_convergence_sto.png` | Max \|ΔQ\| — stochastic |
| `07_policies_det.png` | Learned greedy policies — deterministic |
| `08_policies_sto.png` | Learned greedy policies — stochastic |
| `09_value_heatmaps_det.png` | State-value heatmaps — deterministic |
| `10_value_heatmaps_sto.png` | State-value heatmaps — stochastic |
| `11_overestimation_bias.png` | Q-learning vs Double Q-learning bias |
| `12_steps_per_episode.png` | Steps per episode comparison |
| `13_multiseed_comparison.png` | Mean ± std over 5 seeds |
| `16_summary_metrics_table.png` | Full evaluation metrics table |

---

## Key Results

| Algorithm | Environment | Avg Reward | Conv. Episode | Q-Bias |
|---|---|---|---|---|
| Q-Learning | Deterministic | 8.06 | 549 | ref |
| SARSA | Deterministic | 8.33 | N/C | n/a |
| Double Q-Learning | Deterministic | **8.75** | **388** | +0.0394 |
| Q-Learning | Stochastic | **6.66** | N/C | ref |
| SARSA | Stochastic | 5.32 | N/C | n/a |
| Double Q-Learning | Stochastic | 5.83 | N/C | +0.2422 |

> **N/C** = Not Converged within training budget.

---

## References

- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems. University of Cambridge.
- Van Hasselt, H. (2010). Double Q-learning. *NeurIPS*, 23.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

