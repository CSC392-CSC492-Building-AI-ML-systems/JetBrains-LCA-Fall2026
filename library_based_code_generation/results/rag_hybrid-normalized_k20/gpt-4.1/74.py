```python
import acme
from acme import specs
from acme.agents.jax import cql
from acme.jax import experiments
from acme.jax import utils as jax_utils
from acme.jax.networks import make_default_networks
from acme.jax.environment_loop import EnvironmentLoop
from acme.utils import loggers
import jax
import numpy as np

# User-specified parameters
ENV_NAME = 'HalfCheetah-v2'
DATASET_NAME = 'halfcheetah-medium-v2'
BATCH_SIZE = 256
EVAL_PERIOD = 10000
NUM_DEMO_EPISODES = 10
SEED = 42
LEARNING_RATE = 3e-4
CQL_ALPHA = 1.0
CQL_TEMP = 1.0
CQL_NUM_ACTION_SAMPLES = 10
CQL_MIN_Q_WEIGHT = 5.0
CQL_TARGET_UPDATE_PERIOD = 1000
NUM_EVAL_EPISODES = 5
NUM_LEARN_STEPS = 100000

# Create environment and environment spec
environment = acme.wrappers.GymWrapper(gym.make(ENV_NAME))
environment_spec = specs.make_environment_spec(environment)

# Create demonstration dataset
demonstrations = cql.demonstration_dataset_factory(DATASET_NAME, environment_spec)

# Create agent networks
networks = cql.CQLNetworks(environment_spec)

# Optimizer
optimizer = optax.adam(LEARNING_RATE)

# CQL config
cql_config = cql.CQLConfig(
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    min_q_weight=CQL_MIN_Q_WEIGHT,
    num_action_samples=CQL_NUM_ACTION_SAMPLES,
    temperature=CQL_TEMP,
    cql_alpha=CQL_ALPHA,
    target_update_period=CQL_TARGET_UPDATE_PERIOD,
    seed=SEED,
)

# Build the agent
builder = cql.CQLBuilder(config=cql_config)

# Create the learner
learner = builder.make_learner(
    random_key=jax.random.PRNGKey(SEED),
    networks=networks,
    dataset=demonstrations,
    logger=loggers.TerminalLogger('learner', time_delta=10.0),
)

# Create the actor
actor = builder.make_actor(
    random_key=jax.random.PRNGKey(SEED + 1),
    policy_network=networks.policy_network,
    variable_source=learner,
)

# Evaluator network (copy of policy network)
evaluator_network = networks.policy_network

# Evaluation environment
eval_env = acme.wrappers.GymWrapper(gym.make(ENV_NAME))

# Evaluation actor
eval_actor = builder.make_actor(
    random_key=jax.random.PRNGKey(SEED + 2),
    policy_network=evaluator_network,
    variable_source=learner,
    evaluation=True,
)

# Environment loop for training
train_logger = loggers.TerminalLogger('train_loop', time_delta=10.0)
train_loop = EnvironmentLoop(
    environment=environment,
    actor=actor,
    logger=train_logger,
)

# Environment loop for evaluation
eval_logger = loggers.TerminalLogger('eval_loop', time_delta=10.0)
eval_loop = EnvironmentLoop(
    environment=eval_env,
    actor=eval_actor,
    logger=eval_logger,
)

# Training and evaluation loop
for step in range(0, NUM_LEARN_STEPS, EVAL_PERIOD):
    # Train
    train_loop.run(num_steps=EVAL_PERIOD)
    # Evaluate
    eval_loop.run(num_episodes=NUM_EVAL_EPISODES)
```