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

# Set random seed for reproducibility
np.random.seed(42)

# Problem parameters
n_arms = 5
means = [0.1, 0.5, 0.3, 0.8, 0.6]
T = 1000  # Horizon

# Create the bandit environment
env_ctor = BernoulliBandit
env_kwargs = dict(p=means)

# Define agent classes and their parameters
agent_classes = [
    ("UCB", UCB, {}),
    ("UCBV", UCBV, {}),
    ("ETC", ETC, {"horizon": T}),
    ("MOSS", MOSS, {}),
    ("IMED", IMED, {}),
    ("NPTS", NPTS, {}),
    ("EXP3", EXP3, {"gamma": 0.07}),
]

# Create ExperimentManagers for each agent
managers = []
for name, agent_cls, agent_kwargs in agent_classes:
    manager = ExperimentManager(
        agent_cls,
        (env_ctor, env_kwargs),
        fit_budget=T,
        n_fit=1,
        agent_name=name,
        agent_kwargs=agent_kwargs,
        wrapper_class=WriterWrapper,
    )
    managers.append(manager)

# Train all agents
for manager in managers:
    manager.fit()

# Gather data for plotting
regret_data = {}
pseudo_regret_data = {}
arm_counts_data = {}

optimal_mean = max(means)

for manager in managers:
    writer_data = manager.get_writer_data()
    rewards = np.array(writer_data["reward"])
    actions = np.array(writer_data["action"])
    cum_rewards = np.cumsum(rewards)
    cum_optimal = np.cumsum([optimal_mean] * len(rewards))
    cum_regret = cum_optimal - cum_rewards
    pseudo_regret = np.cumsum([optimal_mean - means[a] for a in actions])
    arm_counts = np.array([np.sum(actions == i) for i in range(n_arms)])
    regret_data[manager.agent_name] = cum_regret
    pseudo_regret_data[manager.agent_name] = pseudo_regret
    arm_counts_data[manager.agent_name] = arm_counts

# Plotting styles
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
colors = sns.color_palette("tab10", n_colors=len(agent_classes))

# Plot cumulative pseudo-regret
plt.figure(figsize=(10, 6))
for idx, (name, _, _) in enumerate(agent_classes):
    plt.plot(
        pseudo_regret_data[name],
        label=name,
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
for idx, (name, _, _) in enumerate(agent_classes):
    plt.plot(
        regret_data[name],
        label=name,
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
plt.figure(figsize=(12, 6))
bar_width = 0.1
indices = np.arange(n_arms)
for idx, (name, _, _) in enumerate(agent_classes):
    plt.bar(
        indices + idx * bar_width,
        arm_counts_data[name],
        width=bar_width,
        label=name,
        color=colors[idx]
    )
plt.xlabel("Arm")
plt.ylabel("Number of Selections")
plt.title("Number of Times Each Arm Was Selected")
plt.xticks(indices + bar_width * (len(agent_classes) - 1) / 2, [f"Arm {i}" for i in range(n_arms)])
plt.legend()
plt.tight_layout()
plt.show()
```