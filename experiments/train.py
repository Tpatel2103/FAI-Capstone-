import numpy as np
from agents.sarsa import SARSAAgent


# train_offpolicy
def train_offpolicy(agent, env, n_episodes=2000, max_steps=200, verbose=False):
    rewards_all, steps_all, deltas_all, tderr_all = [], [], [], []

    for ep in range(n_episodes):
        state     = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        ep_tderrs = []
        Q_before  = agent.Q.copy()

        for _ in range(max_steps):
            action                      = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            td_err                      = agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            ep_steps  += 1
            ep_tderrs.append(abs(td_err))
            state = next_state
            if done:
                break

        agent.end_episode()
        delta = float(np.max(np.abs(agent.Q - Q_before)))
        rewards_all.append(ep_reward)
        steps_all.append(ep_steps)
        deltas_all.append(delta)
        tderr_all.append(float(np.mean(ep_tderrs)) if ep_tderrs else 0.0)

        if verbose and (ep + 1) % 500 == 0:
            avg_r = np.mean(rewards_all[-100:])
            print(f"    Ep {ep+1:4d}/{n_episodes} | AvgR(100)={avg_r:7.3f} | eps={agent.epsilon:.3f} | MaxDQ={delta:.5f}")

    agent.rewards_per_episode = rewards_all
    agent.steps_per_episode   = steps_all
    agent.max_delta_q         = deltas_all
    agent.td_errors           = tderr_all
    return {"rewards": rewards_all, "steps": steps_all,
            "max_delta_q": deltas_all, "td_errors": tderr_all}


# train_sarsa
def train_sarsa(agent, env, n_episodes=2000, max_steps=200, verbose=False):
    rewards_all, steps_all, deltas_all, tderr_all = [], [], [], []

    for ep in range(n_episodes):
        state     = env.reset()
        action    = agent.select_action(state)
        ep_reward = 0.0
        ep_steps  = 0
        ep_tderrs = []
        Q_before  = agent.Q.copy()

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action                 = agent.select_action(next_state)
            td_err = agent.update(state, action, reward, next_state, next_action, done)
            ep_reward += reward
            ep_steps  += 1
            ep_tderrs.append(abs(td_err))
            state  = next_state
            action = next_action
            if done:
                break

        agent.end_episode()
        delta = float(np.max(np.abs(agent.Q - Q_before)))
        rewards_all.append(ep_reward)
        steps_all.append(ep_steps)
        deltas_all.append(delta)
        tderr_all.append(float(np.mean(ep_tderrs)) if ep_tderrs else 0.0)

        if verbose and (ep + 1) % 500 == 0:
            avg_r = np.mean(rewards_all[-100:])
            print(f"    Ep {ep+1:4d}/{n_episodes} | AvgR(100)={avg_r:7.3f} | eps={agent.epsilon:.3f} | MaxDQ={delta:.5f}")

    agent.rewards_per_episode = rewards_all
    agent.steps_per_episode   = steps_all
    agent.max_delta_q         = deltas_all
    agent.td_errors           = tderr_all
    return {"rewards": rewards_all, "steps": steps_all,
            "max_delta_q": deltas_all, "td_errors": tderr_all}


# train
def train(agent, env, n_episodes=2000, max_steps=200, verbose=False):
    if isinstance(agent, SARSAAgent):
        return train_sarsa(agent, env, n_episodes, max_steps, verbose)
    return train_offpolicy(agent, env, n_episodes, max_steps, verbose)


# run_multiple_seeds
def run_multiple_seeds(AgentClass, agent_kwargs, env_kwargs,
                       n_episodes=2000, n_seeds=5, max_steps=200):
    from envs.gridworld import GridWorld
    all_rewards, all_deltas = [], []
    for seed in range(n_seeds):
        agent = AgentClass(**{**agent_kwargs, "seed": seed})
        env   = GridWorld(**{**env_kwargs,   "seed": seed})
        m     = train(agent, env, n_episodes=n_episodes, max_steps=max_steps)
        all_rewards.append(m["rewards"])
        all_deltas.append(m["max_delta_q"])
    r = np.array(all_rewards)
    d = np.array(all_deltas)
    return {"rewards_mean": r.mean(0), "rewards_std": r.std(0),
            "deltas_mean":  d.mean(0), "deltas_std":  d.std(0),
            "n_seeds": n_seeds, "n_episodes": n_episodes}


# convergence_episode
def convergence_episode(max_delta_q, threshold=0.01, consecutive=10):
    for i in range(consecutive - 1, len(max_delta_q)):
        window = max_delta_q[i - consecutive + 1: i + 1]
        if all(d < threshold for d in window):
            return i - consecutive + 1
    return -1


# policy_stability
def policy_stability(max_delta_q, window=100, threshold=0.01):
    if not max_delta_q:
        return 0.0
    recent = max_delta_q[-window:]
    return round(sum(1 for d in recent if d < threshold) / len(recent), 4)
