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
    EXP3
)
from rlberry.wrappers import WriterWrapper
from rlberry.manager import ExperimentManager

# Problem parameters
n_arms = 5
arm_means = [0.1, 0.5, 0.3, 0.8, 0.6]
horizon = 5000
n_fit = 1  # Number of independent runs per agent

# Environment constructor
def env_ctor():
    return BernoulliBandit(arm_means)

# Agent constructors and names
agent_classes = [
    (UCB, "UCB"),
    (UCBV, "UCBV"),
    (ETC, "ETC"),
    (MOSS, "MOSS"),
    (IMED, "IMED"),
    (NPTS, "NPTS"),
    (EXP3, "EXP3")
]

agent_kwargs = {
    "UCB": dict(),
    "UCBV": dict(),
    "ETC": dict(m=10),
    "MOSS": dict(),
    "IMED": dict(),
    "NPTS": dict(),
    "EXP3": dict(gamma=0.07)
}

# Create ExperimentManagers for each agent
managers = []
for agent_class, agent_name in agent_classes:
    manager = ExperimentManager(
        agent_class,
        (env_ctor, {}),
        agent_kwargs[agent_name],
        fit_budget=horizon,
        n_fit=n_fit,
        agent_name=agent_name,
        wrappers=[(WriterWrapper, {})]
    )
    managers.append(manager)

# Train agents
for manager in managers:
    manager.fit()

# Gather data
regret_data = {}
pseudo_regret_data = {}
arm_counts_data = {}

optimal_mean = max(arm_means)

for manager in managers:
    agent_name = manager.agent_name
    writer_data = manager.get_writer_data()
    actions = writer_data['action'][0]
    rewards = writer_data['reward'][0]
    n_steps = len(actions)
    # Cumulative regret
    optimal_rewards = np.full(n_steps, optimal_mean)
    cumulative_regret = np.cumsum(optimal_rewards - rewards)
    regret_data[agent_name] = cumulative_regret
    # Cumulative pseudo-regret
    chosen_means = np.array([arm_means[a] for a in actions])
    cumulative_pseudo_regret = np.cumsum(optimal_mean - chosen_means)
    pseudo_regret_data[agent_name] = cumulative_pseudo_regret
    # Arm counts
    arm_counts = np.zeros(n_arms)
    for a in actions:
        arm_counts[a] += 1
    arm_counts_data[agent_name] = arm_counts

# Plotting styles
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
colors = sns.color_palette("tab10", n_colors=len(agent_classes))

# Plot cumulative pseudo-regret
plt.figure(figsize=(10, 6))
for idx, (agent_class, agent_name) in enumerate(agent_classes):
    plt.plot(
        pseudo_regret_data[agent_name],
        label=agent_name,
        linestyle=linestyles[idx % len(linestyles)],
        color=colors[idx]
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Pseudo-Regret")
plt.title("Cumulative Pseudo-Regret of Bandit Agents")
plt.legend()
plt.tight_layout()
plt.show()

# Plot cumulative regret
plt.figure(figsize=(10, 6))
for idx, (agent_class, agent_name) in enumerate(agent_classes):
    plt.plot(
        regret_data[agent_name],
        label=agent_name,
        linestyle=linestyles[idx % len(linestyles)],
        color=colors[idx]
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret of Bandit Agents")
plt.legend()
plt.tight_layout()
plt.show()

# Plot number of times each arm was selected
plt.figure(figsize=(10, 6))
bar_width = 0.1
indices = np.arange(n_arms)
for idx, (agent_class, agent_name) in enumerate(agent_classes):
    plt.bar(
        indices + idx * bar_width,
        arm_counts_data[agent_name],
        width=bar_width,
        label=agent_name,
        color=colors[idx]
    )
plt.xlabel("Arm")
plt.ylabel("Number of selections")
plt.title("Number of times each arm was selected")
plt.xticks(indices + bar_width * (len(agent_classes) - 1) / 2, [f"Arm {i}" for i in range(n_arms)])
plt.legend()
plt.tight_layout()
plt.show()
```