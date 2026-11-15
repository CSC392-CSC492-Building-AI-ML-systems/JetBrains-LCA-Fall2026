```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import (
    UCB,
    UCBV,
    ETC,
    MOSS,
    IMED,
    NPTS,
    EXP3,
)
from rlberry.manager import ExperimentManager
from rlberry.wrappers import WriterWrapper

# Set random seed for reproducibility
np.random.seed(42)

# Problem parameters
n_arms = 5
arm_means = [0.1, 0.5, 0.3, 0.8, 0.6]
horizon = 5000
n_fit = 1  # Number of independent runs per agent

# Environment constructor
def env_ctor():
    return BernoulliBandit(arm_means)

# Agent constructors
agent_classes = {
    "UCB": UCB,
    "UCBV": UCBV,
    "ETC": ETC,
    "MOSS": MOSS,
    "IMED": IMED,
    "NPTS": NPTS,
    "EXP3": EXP3,
}

agent_kwargs = {
    "UCB": dict(),
    "UCBV": dict(),
    "ETC": dict(m=10),
    "MOSS": dict(),
    "IMED": dict(),
    "NPTS": dict(),
    "EXP3": dict(gamma=0.07),
}

# Wrap agent with WriterWrapper to track actions and rewards
def agent_ctor(agent_class, **kwargs):
    def ctor(env, **_):
        agent = agent_class(env, **kwargs)
        return WriterWrapper(agent)
    return ctor

# Create ExperimentManagers for each agent
managers = []
for name, agent_class in agent_classes.items():
    ctor = agent_ctor(agent_class, **agent_kwargs[name])
    manager = ExperimentManager(
        ctor,
        (env_ctor,),
        fit_budget=horizon,
        n_fit=n_fit,
        agent_name=name,
        output_dir=None,
        parallelization="none",
        seed=42,
    )
    managers.append(manager)

# Fit all agents
for manager in managers:
    manager.fit()

# Compute cumulative regret and pseudo-regret
def compute_regrets(manager, arm_means, horizon):
    logs = manager.get_writer_data()
    actions = np.array(logs["action"])
    rewards = np.array(logs["reward"])
    optimal_mean = np.max(arm_means)
    optimal_reward = optimal_mean * np.ones_like(rewards)
    pseudo_regret = np.cumsum(optimal_mean - np.array([arm_means[a] for a in actions]))
    regret = np.cumsum(optimal_reward - rewards)
    return pseudo_regret, regret, actions

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(18, 5))

# Line styles for each agent
line_styles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (1, 10)),
]
colors = sns.color_palette("tab10", n_colors=len(agent_classes))

# Plot cumulative pseudo-regret
plt.subplot(1, 3, 1)
for idx, manager in enumerate(managers):
    pseudo_regret, _, _ = compute_regrets(manager, arm_means, horizon)
    plt.plot(
        np.arange(1, len(pseudo_regret) + 1),
        pseudo_regret,
        label=manager.agent_name,
        linestyle=line_styles[idx % len(line_styles)],
        color=colors[idx],
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Pseudo-Regret")
plt.title("Cumulative Pseudo-Regret")
plt.legend()

# Plot cumulative regret
plt.subplot(1, 3, 2)
for idx, manager in enumerate(managers):
    _, regret, _ = compute_regrets(manager, arm_means, horizon)
    plt.plot(
        np.arange(1, len(regret) + 1),
        regret,
        label=manager.agent_name,
        linestyle=line_styles[idx % len(line_styles)],
        color=colors[idx],
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret")
plt.legend()

# Plot number of times each arm was selected
plt.subplot(1, 3, 3)
width = 0.1
x = np.arange(n_arms)
for idx, manager in enumerate(managers):
    _, _, actions = compute_regrets(manager, arm_means, horizon)
    counts = np.bincount(actions, minlength=n_arms)
    plt.bar(
        x + idx * width,
        counts,
        width=width,
        label=manager.agent_name,
        color=colors[idx],
        alpha=0.8,
    )
plt.xlabel("Arm")
plt.ylabel("Number of selections")
plt.title("Arm Selection Counts")
plt.xticks(x + width * (len(managers) - 1) / 2, [f"Arm {i}" for i in range(n_arms)])
plt.legend()

plt.tight_layout()
plt.show()
```