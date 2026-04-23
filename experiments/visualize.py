"""
experiments/visualize.py
=========================
All visualisation functions for the capstone project.

Produces the following plots (saved to results/):
  01_gridworld_layout.png         — environment overview
  02_qlearning_training.png       — Q-learning baseline (Milestone 1)
  03_learning_curves_det.png      — reward curves, deterministic
  04_learning_curves_sto.png      — reward curves, stochastic
  05_convergence_det.png          — MaxDQ curves, deterministic
  06_convergence_sto.png          — MaxDQ curves, stochastic
  07_policies_det.png             — greedy policies, deterministic
  08_policies_sto.png             — greedy policies, stochastic
  09_value_heatmaps_det.png       — V(s) heatmaps, deterministic
  10_value_heatmaps_sto.png       — V(s) heatmaps, stochastic
  11_overestimation_bias.png      — Q-learning vs Double Q overestimation
  12_steps_per_episode.png        — steps per episode comparison
  13_multiseed_comparison.png     — mean±std over 5 seeds
  14_hyperparam_alpha.png         — alpha sensitivity
  15_hyperparam_epsilon.png       — epsilon sensitivity
  16_summary_metrics_table.png    — all 4 Milestone-2 metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

COLORS = {
    "Q-Learning":        "#2166ac",
    "SARSA":             "#1a9641",
    "Double Q-Learning": "#d73027",
}
LINESTYLES = {
    "Q-Learning":        "-",
    "SARSA":             "--",
    "Double Q-Learning": "-.",
}
ARROWS = ["↑", "↓", "←", "→"]


def _smooth(values, w=50):
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    result = np.convolve(arr, np.ones(w) / w, mode="same")
    for i in range(min(w, len(result))):
        result[i] = np.mean(arr[: i + 1])
    return result


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ── 01: Grid-world layout ─────────────────────────────────────────────────────

def plot_gridworld(env, out_dir):
    from envs.gridworld import GridWorld
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("5×5 Grid-World Environments (Milestone 1 & 2)", fontsize=12, fontweight="bold")

    for ax, (title, stoch) in zip(axes, [("Deterministic (p=1.0)", False),
                                          ("Stochastic   (p=0.8)", True)]):
        _draw_empty_grid(ax, env, title)

    _save(fig, os.path.join(out_dir, "01_gridworld_layout.png"))


def _draw_empty_grid(ax, env, title):
    CELL = {
        "wall":  ("#555555", "W", "white"),
        "goal":  ("#1a9641", "G\n+10", "white"),
        "trap":  ("#d73027", "T\n−5", "white"),
        "start": ("#2166ac", "S", "white"),
        "empty": ("#f5f5f5", "", "#888"),
    }
    ax.set_xlim(-0.5, env.COLS - 0.5)
    ax.set_ylim(-0.5, env.ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.COLS))
    ax.set_yticks(range(env.ROWS))
    ax.set_xticklabels([f"c{i}" for i in range(env.COLS)], fontsize=8)
    ax.set_yticklabels([f"r{env.ROWS-1-i}" for i in range(env.ROWS)], fontsize=8)
    ax.set_title(title)

    for r in range(env.ROWS):
        for c in range(env.COLS):
            ct = env.cell_type(r, c)
            color, text, tc = CELL[ct]
            dr = env.ROWS - r - 1
            ax.add_patch(plt.Rectangle((c - 0.5, dr - 0.5), 1, 1, color=color, zorder=2))
            ax.text(c, dr, text, ha="center", va="center", color=tc, fontsize=9,
                    fontweight="bold" if ct in ("goal", "trap") else "normal", zorder=3)

    for i in range(env.ROWS + 1):
        ax.axhline(i - 0.5, color="gray", lw=0.5, zorder=1)
    for j in range(env.COLS + 1):
        ax.axvline(j - 0.5, color="gray", lw=0.5, zorder=1)

    # Legend
    patches = [
        mpatches.Patch(color="#2166ac", label="S = Start"),
        mpatches.Patch(color="#1a9641", label="G = Goal (+10)"),
        mpatches.Patch(color="#d73027", label="T = Trap  (−5)"),
        mpatches.Patch(color="#555555", label="W = Wall"),
        mpatches.Patch(color="#f5f5f5", label=". = Empty (−0.1)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9)


# ── 02: Q-learning baseline (Milestone 1) ─────────────────────────────────────

def plot_qlearning_baseline(agent, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Q-Learning Baseline — Deterministic Grid World (Milestone 1)", fontsize=12, fontweight="bold")

    ep = np.arange(len(agent.rewards_per_episode))
    c  = COLORS["Q-Learning"]

    # Reward
    axes[0].fill_between(ep, agent.rewards_per_episode, alpha=0.12, color=c)
    axes[0].plot(ep, _smooth(agent.rewards_per_episode), color=c, lw=2)
    axes[0].set_title("Cumulative reward per episode")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color="gray", lw=0.5, ls=":")
    axes[0].grid(alpha=0.25)

    # Convergence
    axes[1].semilogy(ep, np.maximum(_smooth(agent.max_delta_q, w=30), 1e-7), color=c, lw=2)
    axes[1].axhline(0.01, color="red", lw=1, ls="--", alpha=0.7, label="0.01 threshold")
    axes[1].set_title("Convergence: Max |ΔQ| (log)")
    axes[1].set_xlabel("Episode"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # Steps
    axes[2].fill_between(ep, agent.steps_per_episode, alpha=0.12, color=c)
    axes[2].plot(ep, _smooth(agent.steps_per_episode), color=c, lw=2)
    axes[2].set_title("Steps per episode")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Steps")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "02_qlearning_training.png"))


# ── 03-04: Learning curves ─────────────────────────────────────────────────────

def plot_learning_curves(results_det, results_sto, out_dir):
    for suffix, results, env_label in [
        ("det", results_det, "Deterministic"),
        ("sto", results_sto, "Stochastic (p=0.8)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, data in results.items():
            r  = data["agent"].rewards_per_episode
            sm = _smooth(r)
            ep = np.arange(len(r))
            ax.fill_between(ep, r, alpha=0.07, color=COLORS[name])
            ax.plot(ep, sm, color=COLORS[name], lw=2,
                    linestyle=LINESTYLES[name], label=name)

        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"Cumulative Reward per Episode — {env_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative reward (smoothed, w=50)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
        _save(fig, os.path.join(out_dir, f"0{'3' if suffix=='det' else '4'}_learning_curves_{suffix}.png"))


# ── 05-06: Convergence curves ──────────────────────────────────────────────────

def plot_convergence(results_det, results_sto, out_dir):
    for idx, (results, env_label) in enumerate([(results_det, "Deterministic"),
                                                 (results_sto, "Stochastic (p=0.8)")]):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, data in results.items():
            d  = data["agent"].max_delta_q
            sm = np.maximum(_smooth(d, w=30), 1e-7)
            ep = np.arange(len(d))
            ax.semilogy(ep, sm, color=COLORS[name], lw=2,
                        linestyle=LINESTYLES[name], label=name)

        ax.axhline(0.01, color="black", lw=0.8, ls=":", alpha=0.6, label="0.01 convergence threshold")
        ax.set_title(f"Convergence: Max |ΔQ| — {env_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Max |ΔQ| (log scale)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
        _save(fig, os.path.join(out_dir, f"0{'5' if idx==0 else '6'}_convergence_{'det' if idx==0 else 'sto'}.png"))


# ── 07-08: Greedy policy grids ────────────────────────────────────────────────

def plot_policies(results_det, results_sto, env, out_dir):
    for idx, (results, env_label, suffix) in enumerate([
        (results_det, "Deterministic", "det"),
        (results_sto, "Stochastic",   "sto"),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(f"Learned Greedy Policies — {env_label}", fontsize=12, fontweight="bold")
        for ax, (name, data) in zip(axes, results.items()):
            _draw_policy_grid(ax, env, data["agent"].Q, name)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"0{'7' if idx==0 else '8'}_policies_{suffix}.png"))


def _draw_policy_grid(ax, env, Q, title):
    ax.set_xlim(-0.5, env.COLS - 0.5)
    ax.set_ylim(-0.5, env.ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    for r in range(env.ROWS):
        for c in range(env.COLS):
            ct = env.cell_type(r, c)
            dr = env.ROWS - r - 1
            if ct == "wall":
                ax.add_patch(plt.Rectangle((c-0.5, dr-0.5), 1, 1, color="#444", zorder=2))
                ax.text(c, dr, "W", ha="center", va="center", color="white", fontsize=9, zorder=3)
            elif ct == "goal":
                ax.add_patch(plt.Rectangle((c-0.5, dr-0.5), 1, 1, color="#1a9641", zorder=2))
                ax.text(c, dr, "G", ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=3)
            elif ct == "trap":
                ax.add_patch(plt.Rectangle((c-0.5, dr-0.5), 1, 1, color="#d73027", zorder=2))
                ax.text(c, dr, "T", ha="center", va="center", color="white", fontsize=11, zorder=3)
            else:
                v    = np.max(Q[r, c])
                norm = min(max((v + 2) / 12.0, 0.0), 1.0)
                bg   = plt.cm.Blues(0.15 + norm * 0.75) if ct != "start" else "#2166ac"
                ax.add_patch(plt.Rectangle((c-0.5, dr-0.5), 1, 1, color=bg, zorder=2))
                best = int(np.argmax(Q[r, c]))
                prefix = "S\n" if ct == "start" else ""
                ax.text(c, dr, prefix + ARROWS[best],
                        ha="center", va="center", color="white" if ct == "start" else "#111",
                        fontsize=14, zorder=3)

    for i in range(env.ROWS + 1): ax.axhline(i - 0.5, color="gray", lw=0.4, zorder=1)
    for j in range(env.COLS + 1): ax.axvline(j - 0.5, color="gray", lw=0.4, zorder=1)


# ── 09-10: V(s) heatmaps ─────────────────────────────────────────────────────

def plot_value_heatmaps(results_det, results_sto, env, out_dir):
    for idx, (results, env_label, suffix) in enumerate([
        (results_det, "Deterministic", "det"),
        (results_sto, "Stochastic",   "sto"),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(f"State-Value Function V(s) = max_a Q(s,a) — {env_label}",
                     fontsize=12, fontweight="bold")
        for ax, (name, data) in zip(axes, results.items()):
            V  = data["agent"].V.copy()
            mask = np.zeros_like(V, dtype=bool)
            for wr, wc in env.WALLS: mask[wr, wc] = True
            Vm = np.ma.array(V, mask=mask)
            im = ax.imshow(Vm, cmap="RdYlGn", vmin=-5, vmax=10, aspect="equal")
            plt.colorbar(im, ax=ax, shrink=0.75, label="V(s)")
            for r in range(env.ROWS):
                for c in range(env.COLS):
                    ct = env.cell_type(r, c)
                    lbl = {"wall":"W","goal":"G","trap":"T"}.get(ct, f"{V[r,c]:.1f}")
                    col = "white" if ct in ("wall","goal","trap") else "black"
                    ax.text(c, r, lbl, ha="center", va="center", fontsize=8, color=col)
            ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"0{'9' if idx==0 else '10'}_value_heatmaps_{suffix}.png"))


# ── 11: Overestimation bias ────────────────────────────────────────────────────

def plot_overestimation_bias(results_det, results_sto, env, out_dir):
    """
    Milestone-2 metric 4: overestimation bias.
    Q-learning typically overestimates max Q-values vs Double Q-learning.
    """
    nw = [(r, c) for r in range(env.ROWS) for c in range(env.COLS)
          if env.cell_type(r, c) != "wall"]
    labels = [f"({r},{c})" for r, c in nw]
    x = np.arange(len(nw))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Q-Value Overestimation: Q-learning vs Double Q-learning (Van Hasselt, 2010)",
                 fontsize=12, fontweight="bold")

    for ax, (results, env_label) in zip(axes, [
        (results_det, "Deterministic"),
        (results_sto, "Stochastic (p=0.8)"),
    ]):
        q_vals  = [np.max(results["Q-Learning"]["agent"].Q[r, c])        for r, c in nw]
        dq_vals = [np.max(results["Double Q-Learning"]["agent"].Q[r, c]) for r, c in nw]

        ax.bar(x - w/2, q_vals,  w, label="Q-learning",
               color=COLORS["Q-Learning"], alpha=0.85)
        ax.bar(x + w/2, dq_vals, w, label="Double Q-learning",
               color=COLORS["Double Q-Learning"], alpha=0.85)

        bias = np.mean(np.array(q_vals) - np.array(dq_vals))
        ax.set_title(f"{env_label}\nMean overestimation = {bias:+.4f}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel("max Q(s,·)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
        ax.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "11_overestimation_bias.png"))


# ── 12: Steps per episode ──────────────────────────────────────────────────────

def plot_steps(results_det, results_sto, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Steps per Episode (lower = more efficient)", fontsize=12, fontweight="bold")

    for ax, (results, env_label) in zip(axes, [
        (results_det, "Deterministic"),
        (results_sto, "Stochastic (p=0.8)"),
    ]):
        for name, data in results.items():
            s  = data["agent"].steps_per_episode
            sm = _smooth(s)
            ep = np.arange(len(s))
            ax.plot(ep, sm, color=COLORS[name], lw=2,
                    linestyle=LINESTYLES[name], label=name)
        ax.set_title(env_label)
        ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "12_steps_per_episode.png"))


# ── 13: Multi-seed mean±std ────────────────────────────────────────────────────

def plot_multiseed(multiseed_results: dict, out_dir):
    """
    multiseed_results = {algo_name: {rewards_mean, rewards_std, ...}}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in multiseed_results.items():
        ep   = np.arange(len(res["rewards_mean"]))
        mean = _smooth(res["rewards_mean"])
        std  = res["rewards_std"]
        ax.plot(ep, mean, color=COLORS[name], lw=2,
                linestyle=LINESTYLES[name], label=name)
        ax.fill_between(ep, mean - std, mean + std, alpha=0.15, color=COLORS[name])

    n = list(multiseed_results.values())[0]["n_seeds"]
    ax.set_title(f"Multi-Seed Comparison — Mean ± Std ({n} seeds) — Deterministic",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative reward")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    _save(fig, os.path.join(out_dir, "13_multiseed_comparison.png"))


# ── 14-15: Hyperparameter sensitivity ─────────────────────────────────────────

def plot_hyperparam_sensitivity(param_name, param_vals, scores_dict, out_dir, idx):
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"Q-Learning": "o", "SARSA": "s", "Double Q-Learning": "^"}
    for name, vals in scores_dict.items():
        ax.plot(param_vals, vals, color=COLORS[name],
                marker=markers[name], lw=2, ms=7, label=name)

    ax.set_xlabel(f"{param_name}", fontsize=11)
    ax.set_ylabel("Avg reward (last 100 ep)", fontsize=11)
    ax.set_title(f"Sensitivity to {param_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    fname = f"{'14' if idx == 0 else '15'}_hyperparam_{param_name.split()[0].lower()}.png"
    _save(fig, os.path.join(out_dir, fname))


# ── 16: Milestone-2 metrics summary table ─────────────────────────────────────

def plot_metrics_table(metrics_data: list, out_dir):
    """
    metrics_data: list of dicts with keys:
      Algorithm, Environment, AvgReward, AvgSteps, ConvEp, PolicyStab, QBias
    """
    fig, ax = plt.subplots(figsize=(14, len(metrics_data) * 0.55 + 1.5))
    ax.axis("off")

    cols = ["Algorithm", "Environment",
            "Avg Reward\n(last 100 ep)",
            "Avg Steps\n(last 100 ep)",
            "Convergence\nEpisode",
            "Policy\nStability",
            "Q Overest.\nBias"]

    rows = [
        [d["Algorithm"], d["Environment"],
         f'{d["AvgReward"]:.4f}', f'{d["AvgSteps"]:.1f}',
         str(d["ConvEp"]) if d["ConvEp"] >= 0 else "N/C",
         f'{d["PolicyStab"]:.2f}',
         d["QBias"]]
        for d in metrics_data
    ]

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)

    # Header styling
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2166ac")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row striping
    for i in range(len(rows)):
        bg = "#eef3f9" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)

    ax.set_title("Milestone 2 — All Evaluation Metrics", fontsize=13,
                 fontweight="bold", pad=15)
    _save(fig, os.path.join(out_dir, "16_summary_metrics_table.png"))
