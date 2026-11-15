```python
import acme
from acme import specs
from acme.agents.jax import cql
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax

# User-specified parameters
ENV_NAME = 'Pendulum-v1'
DATASET_NAME = 'pendulum_demonstrations'
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
NUM_EVAL_EPISODES = 5
NUM_STEPS = 200_000

# Create environment and specs
environment = acme.environments.ContinuousEnvironment(ENV_NAME)
environment_spec = specs.make_environment_spec(environment)

# Load demonstrations dataset
def load_demonstrations(dataset_name, num_episodes):
    # Placeholder: Replace with actual dataset loading logic
    # Should return a list of (obs, action, reward, next_obs, done) tuples
    # For demonstration, generate random transitions
    demos = []
    for _ in range(num_episodes):
        obs = environment.reset()
        done = False
        while not done:
            action = environment.action_spec().generate_value()
            next_obs, reward, done, _ = environment.step(action)
            demos.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return demos

demonstrations = load_demonstrations(DATASET_NAME, NUM_DEMO_EPISODES)

# Network definitions
def make_networks(env_spec):
    policy_network = networks_lib.FeedForwardNetwork(
        init=lambda key: {},
        apply=lambda params, obs: jnp.tanh(jax.random.normal(key=key, shape=env_spec.actions.shape))
    )
    critic_network = networks_lib.FeedForwardNetwork(
        init=lambda key: {},
        apply=lambda params, obs, action: jnp.sum(obs) + jnp.sum(action)
    )
    return cql.CQLNetworks(
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=networks_lib.identity,
        sample=policy_network.apply,
    )

networks = make_networks(environment_spec)

# Optimizers
policy_optimizer = optax.adam(ACTOR_LR)
critic_optimizer = optax.adam(CRITIC_LR)
alpha_optimizer = optax.adam(ALPHA_LR)

# Logger
logger = loggers.TerminalLogger(label='cql_agent', time_delta=10.0)

# Agent configuration
agent = cql.CQL(
    environment_spec=environment_spec,
    policy_network=networks.policy_network,
    critic_network=networks.critic_network,
    observation_network=networks.observation_network,
    policy_optimizer=policy_optimizer,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=alpha_optimizer,
    demonstrations=demonstrations,
    batch_size=BATCH_SIZE,
    cql_alpha=CQL_ALPHA,
    cql_temperature=CQL_TEMP,
    cql_num_action_samples=CQL_NUM_ACTION_SAMPLES,
    seed=SEED,
    logger=logger,
)

# Evaluator network (copy of policy network)
def evaluator_network(params, obs):
    return networks.policy_network.apply(params, obs)

# Evaluation environment
eval_env = acme.environments.ContinuousEnvironment(ENV_NAME)

# Environment loop
loop = acme.EnvironmentLoop(
    environment=environment,
    actor=agent,
    logger=logger,
    should_update=True,
    should_evaluate=True,
    evaluation_environment=eval_env,
    evaluation_actor=agent,
    evaluation_episodes=NUM_EVAL_EPISODES,
    evaluation_period=EVAL_PERIOD,
)

# Run training loop
loop.run(num_steps=NUM_STEPS)
```