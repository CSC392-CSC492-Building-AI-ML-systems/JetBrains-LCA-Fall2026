```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import (
    UCBAgent,
    UCBVAgent,
    ETCAgent,
    MOSSAgent,
    IMEDAgent,
    NPTSAgent,
    EXP3Agent,
)
from rlberry.wrappers import WriterWrapper
from rlberry.manager import ExperimentManager

# Problem parameters
n_arms = 5
arm_means = [0.1, 0.5, 0.3, 0.8, 0.6]
horizon = 1000
n_fit = 1  # Number of independent runs per agent

# Environment
env_ctor = (BernoulliBandit, dict(p=arm_means))

# Agent definitions
agent_classes = [
    ("UCB", UCBAgent, {}),
    ("UCBV", UCBVAgent, {}),
    ("ETC", ETCAgent, {"horizon": horizon}),
    ("MOSS", MOSSAgent, {}),
    ("IMED", IMEDAgent, {}),
    ("NPTS", NPTSAgent, {}),
    ("EXP3", EXP3Agent, {"gamma": 0.07}),
]

# Experiment managers
managers = []
for name, agent_class, agent_kwargs in agent_classes:
    agent_ctor = (agent_class, agent_kwargs)
    manager = ExperimentManager(
        agent_ctor,
        env_ctor,
        fit_budget=horizon,
        n_fit=n_fit,
        agent_name=name,
        wrappers=[(WriterWrapper, {})],
        parallelization="none",
        seed=123,
    )
    managers.append(manager)

# Train agents
for manager in managers:
    manager.fit()

# Compute regrets and arm pulls
optimal_mean = max(arm_means)
results = {}

for manager in managers:
    agent_name = manager.agent_name
    writer_data = manager.get_writer_data()
    actions = np.array(writer_data["action"])
    rewards = np.array(writer_data["reward"])
    n_steps = len(actions)
    # Cumulative regret
    optimal_rewards = np.full(n_steps, optimal_mean)
    regret = np.cumsum(optimal_rewards - rewards)
    # Cumulative pseudo-regret
    pseudo_regret = np.zeros(n_steps)
    arm_counts = np.zeros(n_arms)
    for t in range(n_steps):
        arm_counts[actions[t]] += 1
        pseudo_regret[t] = np.sum((optimal_mean - np.array(arm_means)) * arm_counts)
    results[agent_name] = {
        "regret": regret,
        "pseudo_regret": pseudo_regret,
        "actions": actions,
        "arm_counts": arm_counts,
    }

# Plotting
sns.set(style="whitegrid")
colors = sns.color_palette("tab10", n_colors=len(agent_classes))
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

plt.figure(figsize=(10, 6))
for idx, (agent_name, _) in enumerate(results.items()):
    plt.plot(
        results[agent_name]["pseudo_regret"],
        label=agent_name,
        color=colors[idx],
        linestyle=linestyles[idx % len(linestyles)],
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Pseudo-Regret")
plt.title("Cumulative Pseudo-Regret vs Time")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for idx, (agent_name, _) in enumerate(results.items()):
    plt.plot(
        results[agent_name]["regret"],
        label=agent_name,
        color=colors[idx],
        linestyle=linestyles[idx % len(linestyles)],
    )
plt.xlabel("Time step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret vs Time")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
bar_width = 0.1
indices = np.arange(n_arms)
for idx, (agent_name, _) in enumerate(results.items()):
    plt.bar(
        indices + idx * bar_width,
        results[agent_name]["arm_counts"],
        width=bar_width,
        label=agent_name,
        color=colors[idx],
        alpha=0.8,
    )
plt.xlabel("Arm")
plt.ylabel("Number of times selected")
plt.title("Number of times each arm was selected")
plt.xticks(indices + bar_width * (len(agent_classes) - 1) / 2, [str(i) for i in range(n_arms)])
plt.legend()
plt.tight_layout()
plt.show()
```