```python
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import cql
from acme.jax import experiments
from acme.jax import utils as acme_utils
from acme.utils import loggers
import dm_env
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import random

# User-specified parameters
ENV_NAME = 'HalfCheetah-v2'
DATASET_NAME = 'd4rl_halfcheetah_medium-v0'
BATCH_SIZE = 256
EVAL_PERIOD = 10000
NUM_DEMO_EPISODES = 10
SEED = 42
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
CQL_ALPHA = 1.0
CQL_TEMP = 1.0
CQL_NUM_ACTION_SAMPLES = 10
NUM_STEPS = 1_000_000

# Set random seeds
np.random.seed(SEED)
random.seed(SEED)
jax_rng = jax.random.PRNGKey(SEED)

# Create environment
def make_environment(seed=SEED):
    env = gym.make(ENV_NAME)
    env.seed(seed)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

environment = make_environment()
environment_spec = specs.make_environment_spec(environment)

# Load demonstrations dataset (assuming D4RL and Acme's dataset API)
import d4rl  # noqa: F401
import tensorflow as tf

def load_demonstrations(dataset_name):
    env = gym.make(dataset_name)
    dataset = env.get_dataset()
    # Convert to Acme's Transition format
    transitions = []
    for i in range(len(dataset['observations']) - 1):
        transition = acme.types.Transition(
            observation=dataset['observations'][i],
            action=dataset['actions'][i],
            reward=dataset['rewards'][i],
            discount=dataset['terminals'][i] ^ 1,
            next_observation=dataset['observations'][i+1],
        )
        transitions.append(transition)
    return transitions

demonstrations = load_demonstrations(DATASET_NAME)

# Define networks
def make_networks(env_spec):
    policy_network = cql.make_policy_network(env_spec)
    critic_network = cql.make_critic_network(env_spec)
    return cql.CQLNetworks(
        policy_network=policy_network,
        critic_network=critic_network,
    )

networks = make_networks(environment_spec)

# Optimizers
policy_optimizer = optax.adam(ACTOR_LR)
critic_optimizer = optax.adam(CRITIC_LR)
alpha_optimizer = optax.adam(ALPHA_LR)

# Logger
logger = loggers.TerminalLogger(label='learner', time_delta=10.0)

# CQL config
cql_config = cql.CQLConfig(
    batch_size=BATCH_SIZE,
    min_replay_size=BATCH_SIZE,
    max_replay_size=1_000_000,
    samples_per_insert=256,
    n_step=1,
    discount=0.99,
    target_update_period=1,
    policy_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    alpha_lr=ALPHA_LR,
    cql_alpha=CQL_ALPHA,
    cql_temp=CQL_TEMP,
    cql_num_action_samples=CQL_NUM_ACTION_SAMPLES,
    seed=SEED,
)

# Evaluator network
def evaluator_factory(networks):
    return networks.policy_network

# Build the agent
agent = cql.CQLBuilder(cql_config).make_agent(
    random_key=jax_rng,
    environment_spec=environment_spec,
    networks=networks,
    policy_optimizer=policy_optimizer,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=alpha_optimizer,
    demonstration_data=demonstrations,
)

# Environment loop with evaluation
eval_env = make_environment(seed=SEED+1)
eval_policy = acme_utils.apply_policy_and_sample(evaluator_factory(networks))

train_loop = acme.EnvironmentLoop(
    environment=environment,
    actor=agent,
    logger=logger,
)

eval_loop = acme.EnvironmentLoop(
    environment=eval_env,
    actor=acme.agents.jax.cql.CQLPolicy(
        policy_network=eval_policy,
        random_key=jax_rng,
        evaluation=True,
    ),
    logger=loggers.TerminalLogger(label='evaluator', time_delta=10.0),
)

# Training and evaluation loop
steps = 0
while steps < NUM_STEPS:
    train_loop.run(num_steps=EVAL_PERIOD)
    eval_loop.run(num_episodes=NUM_DEMO_EPISODES)
    steps += EVAL_PERIOD
```