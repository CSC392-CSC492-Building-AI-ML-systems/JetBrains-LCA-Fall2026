```python
import acme
from acme import specs
from acme.agents.jax import cql
from acme.jax import experiments
from acme.jax import utils as jax_utils
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax

# User-specified parameters
ENV_NAME = 'HalfCheetah-v2'
DATASET_NAME = 'd4rl_halfcheetah_medium'
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
NUM_LEARN_STEPS = 100000

# Create environment and environment spec
environment = acme.wrappers.GymWrapper(gym.make(ENV_NAME))
environment_spec = specs.make_environment_spec(environment)

# Obtain demonstrations dataset
def get_demonstrations(dataset_name, num_episodes):
    # Placeholder for demonstration loading logic
    # Replace with actual dataset loading as per your setup
    # For example, using D4RL:
    import d4rl
    env = gym.make(ENV_NAME)
    dataset = d4rl.qlearning_dataset(env)
    # Optionally subsample episodes
    return dataset

demonstrations = get_demonstrations(DATASET_NAME, NUM_DEMO_EPISODES)

# Build CQL networks
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
    seed=SEED,
    max_steps=NUM_LEARN_STEPS,
)

# Logger
logger = loggers.TerminalLogger(label='learner', time_delta=10.0)

# Create the learner
learner = cql.CQLLearner(
    networks=networks,
    random_key=jax.random.PRNGKey(SEED),
    dataset=demonstrations,
    environment_spec=environment_spec,
    config=cql_config,
    logger=logger,
    optimizer=optimizer,
)

# Create the actor
actor = cql.CQLPolicy(
    networks=networks,
    random_key=jax.random.PRNGKey(SEED + 1),
    environment_spec=environment_spec,
    evaluation=True,
)

# Evaluator network (copy of policy network)
evaluator_network = networks.policy_network

# Environment loop for training and evaluation
def run_loop():
    env = acme.wrappers.GymWrapper(gym.make(ENV_NAME))
    eval_env = acme.wrappers.GymWrapper(gym.make(ENV_NAME))
    for step in range(0, NUM_LEARN_STEPS, EVAL_PERIOD):
        # Training
        for _ in range(EVAL_PERIOD):
            learner.step()
        # Evaluation
        returns = []
        for _ in range(NUM_EVAL_EPISODES):
            timestep = eval_env.reset()
            done = False
            total_reward = 0.0
            while not timestep.last():
                action = actor.select_action(timestep.observation)
                timestep = eval_env.step(action)
                total_reward += timestep.reward
            returns.append(total_reward)
        print(f"Step {step+EVAL_PERIOD}: Eval returns: {returns}")

run_loop()
```