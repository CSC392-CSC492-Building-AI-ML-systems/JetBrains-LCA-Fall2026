```python
import acme
from acme import specs
from acme.agents.jax import cql
from acme.jax import experiments
from acme.jax import utils as jax_utils
from acme.jax.networks import make_default_networks
from acme.utils import loggers
import jax
import numpy as np

# User-specified parameters
ENV_NAME = 'Pendulum-v1'
DATASET_NAME = 'pendulum_demonstrations'
BATCH_SIZE = 256
EVAL_PERIOD = 10000
NUM_DEMO_EPISODES = 10
SEED = 42
LEARNING_RATE = 3e-4
CQL_ALPHA = 1.0
CQL_TEMP = 1.0
CQL_NUM_ACTION_SAMPLES = 10
CQL_MIN_Q_WEIGHT = 5.0
NUM_EVAL_EPISODES = 5
NUM_STEPS = 100_000

# Create environment and specs
environment = acme.environments.ContinuousEnvironment(ENV_NAME)
environment_spec = specs.make_environment_spec(environment)

# Obtain demonstration dataset
demonstrations = acme.datasets.demonstration_dataset_factory(
    dataset_name=DATASET_NAME,
    environment_spec=environment_spec,
    num_episodes=NUM_DEMO_EPISODES,
    seed=SEED,
)

# Create agent networks
networks = cql.CQLNetworks(environment_spec)

# Optimizer
import optax
optimizer = optax.adam(LEARNING_RATE)

# CQL config
cql_config = cql.CQLConfig(
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    min_q_weight=CQL_MIN_Q_WEIGHT,
    num_action_samples=CQL_NUM_ACTION_SAMPLES,
    temperature=CQL_TEMP,
    cql_alpha=CQL_ALPHA,
    seed=SEED,
)

# CQL builder
builder = cql.CQLBuilder(config=cql_config)

# Logger
logger = loggers.TerminalLogger(label='learner', time_delta=10.0)

# Create the learner
learner = cql.CQLLearner(
    networks=networks,
    random_key=jax.random.PRNGKey(SEED),
    dataset=demonstrations,
    optimizer=optimizer,
    logger=logger,
    config=cql_config,
)

# Define evaluator network (copy of policy network)
evaluator_network = networks.policy_network

# Evaluation logger
eval_logger = loggers.TerminalLogger(label='evaluator', time_delta=10.0)

# Evaluation loop
eval_env = acme.environments.ContinuousEnvironment(ENV_NAME)
eval_loop = acme.EnvironmentLoop(
    environment=eval_env,
    actor=acme.jax.actors.FeedForwardActor(
        policy_fn=lambda params, obs, key: evaluator_network(params, obs),
        params=learner.get_policy_params(),
        random_key=jax.random.PRNGKey(SEED + 1),
    ),
    logger=eval_logger,
)

# Training loop
env_loop = acme.EnvironmentLoop(
    environment=environment,
    actor=acme.jax.actors.FeedForwardActor(
        policy_fn=lambda params, obs, key: networks.policy_network(params, obs),
        params=learner.get_policy_params(),
        random_key=jax.random.PRNGKey(SEED),
    ),
    logger=logger,
)

# Main training and evaluation loop
for step in range(0, NUM_STEPS, EVAL_PERIOD):
    # Train for EVAL_PERIOD steps
    for _ in range(EVAL_PERIOD):
        learner.step()
    # Evaluate
    eval_loop.run(num_episodes=NUM_EVAL_EPISODES)
```