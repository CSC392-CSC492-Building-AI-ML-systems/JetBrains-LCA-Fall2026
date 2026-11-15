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
means = [0.1, 0.5, 0.3, 0.7, 0.6]
T = 1000  # Horizon

# Create bandit environment
env_ctor = BernoulliBandit
env_kwargs = dict(p=means)

# Agent parameters
agent_classes = [
    UCB,
    UCBV,
    ETC,
    MOSS,
    IMED,
    NPTS,
    EXP3
]
agent_names = [
    "UCB",
    "UCBV",
    "ETC",
    "MOSS",
    "IMED",
    "NPTS",
    "EXP3"
]
agent_kwargs = [
    dict(bandit_env_kwargs=env_kwargs),
    dict(bandit_env_kwargs=env_kwargs),
    dict(bandit_env_kwargs=env_kwargs, horizon=T),
    dict(bandit_env_kwargs=env_kwargs),
    dict(bandit_env_kwargs=env_kwargs),
    dict(bandit_env_kwargs=env_kwargs),
    dict(bandit_env_kwargs=env_kwargs)
]

# Wrap agents with WriterWrapper to track actions and rewards
def make_wrapped_agent(agent_class, agent_kwargs):
    def ctor(env, **kwargs):
        agent = agent_class(env, **agent_kwargs)
        return WriterWrapper(agent)
    return ctor

wrapped_agent_classes = [
    make_wrapped_agent(agent_class, agent_kwargs[i])
    for i, agent_class in enumerate(agent_classes)
]

# Construct ExperimentManager for each agent
managers = []
for i, agent_ctor in enumerate(wrapped_agent_classes):
    manager = ExperimentManager(
        agent_ctor,
        env_ctor,
        fit_budget=T,
        n_fit=1,
        agent_name=agent_names[i],
        env_kwargs=env_kwargs,
        parallelization="none",
        output_dir=None,
        seed=42
    )
    managers.append(manager)

# Train agents
for manager in managers:
    manager.fit()

# Compute cumulative regret and pseudo-regret
optimal_mean = np.max(means)
results = {}
for i, manager in enumerate(managers):
    agent = manager.agent_instance
    actions = np.array(agent.writer_data["action"])
    rewards = np.array(agent.writer_data["reward"])
    chosen_means = np.array([means[a] for a in actions])
    pseudo_regret = np.cumsum(optimal_mean - chosen_means)
    regret = np.cumsum(optimal_mean - rewards)
    arm_counts = np.bincount(actions, minlength=n_arms)
    results[agent_names[i]] = {
        "pseudo_regret": pseudo_regret,
        "regret": regret,
        "actions": actions,
        "arm_counts": arm_counts
    }

# Plotting
sns.set(style="whitegrid")
colors = sns.color_palette("tab10", n_colors=len(agent_names))
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

plt.figure(figsize=(10, 6))
for i, name in enumerate(agent_names):
    plt.plot(results[name]["pseudo_regret"], label=name, color=colors[i], linestyle=linestyles[i])
plt.xlabel("Time step")
plt.ylabel("Cumulative Pseudo-Regret")
plt.title("Cumulative Pseudo-Regret of Bandit Agents")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, name in enumerate(agent_names):
    plt.plot(results[name]["regret"], label=name, color=colors[i], linestyle=linestyles[i])
plt.xlabel("Time step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret of Bandit Agents")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
bar_width = 0.1
indices = np.arange(n_arms)
for i, name in enumerate(agent_names):
    plt.bar(indices + i * bar_width, results[name]["arm_counts"], width=bar_width, label=name, color=colors[i])
plt.xlabel("Arm")
plt.ylabel("Number of times selected")
plt.title("Number of times each arm was selected")
plt.xticks(indices + bar_width * (len(agent_names) - 1) / 2, [str(i) for i in range(n_arms)])
plt.legend()
plt.tight_layout()
plt.show()
```