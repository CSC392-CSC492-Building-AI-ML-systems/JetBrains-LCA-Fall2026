```python
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import crr
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.utils import loggers
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import functools
from typing import Any, Callable, Dict, Optional

def add_next_action_extras(transitions):
    """Adds 'next_action' to the extras of each transition."""
    next_actions = np.roll(transitions['action'], shift=-1, axis=0)
    # The last transition's next_action is undefined; set to zeros.
    next_actions[-1] = 0
    if 'extras' not in transitions:
        transitions['extras'] = {}
    transitions['extras']['next_action'] = next_actions
    return transitions

def make_environment(env_name: str, seed: int) -> dm_env.Environment:
    """Creates and wraps the environment."""
    env = acme.make_environment(env_name)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.GymWrapper(env)
    env = wrappers.EpisodeStatisticsWrapper(env)
    env = wrappers.RewardNormalizationWrapper(env)
    env = wrappers.ObservationActionRewardWrapper(env)
    env.seed(seed)
    return env

def get_demonstrations(dataset_name: str, num_episodes: int, env_spec: specs.EnvironmentSpec):
    """Loads demonstration dataset."""
    # Placeholder: Replace with actual dataset loading logic.
    # For example, use acme.datasets.load_demonstrations or similar.
    # Here, we create dummy transitions for demonstration.
    transitions = {
        'observation': np.zeros((num_episodes, *env_spec.observations.shape), dtype=env_spec.observations.dtype),
        'action': np.zeros((num_episodes, *env_spec.actions.shape), dtype=env_spec.actions.dtype),
        'reward': np.zeros((num_episodes,), dtype=np.float32),
        'discount': np.ones((num_episodes,), dtype=np.float32),
        'next_observation': np.zeros((num_episodes, *env_spec.observations.shape), dtype=env_spec.observations.dtype),
        'extras': {}
    }
    transitions = add_next_action_extras(transitions)
    return transitions

def create_networks(env_spec: specs.EnvironmentSpec, policy_lr: float, critic_lr: float):
    """Creates the policy and critic networks."""
    # Example: simple MLPs for policy and critic.
    def policy_network(obs):
        mlp = networks_lib.MLP([256, 256, env_spec.actions.num_values])
        logits = mlp(obs)
        return logits

    def critic_network(obs, action):
        mlp = networks_lib.MLP([256, 256, 1])
        x = jnp.concatenate([obs, action], axis=-1)
        q = mlp(x)
        return q

    networks = crr.make_networks(
        environment_spec=env_spec,
        policy_layer_sizes=(256, 256),
        critic_layer_sizes=(256, 256),
    )
    return networks

def main(
    env_name: str = 'CartPole-v1',
    dataset_name: str = 'cartpole_demo',
    batch_size: int = 256,
    evaluation_period: int = 1000,
    num_demonstration_episodes: int = 100,
    seed: int = 0,
    policy_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    discount: float = 0.99,
    target_update_period: int = 100,
    use_sarsa_target: bool = False,
):
    # Create environment and spec.
    environment = make_environment(env_name, seed)
    environment_spec = specs.make_environment_spec(environment)

    # Get demonstration dataset.
    demonstrations = get_demonstrations(dataset_name, num_demonstration_episodes, environment_spec)

    # Create networks.
    networks = create_networks(environment_spec, policy_learning_rate, critic_learning_rate)

    # Create the learner.
    learner = crr.CrrLearner(
        networks=networks,
        random_key=jax.random.PRNGKey(seed),
        demonstration_iterator=iter([demonstrations]),
        batch_size=batch_size,
        policy_learning_rate=policy_learning_rate,
        critic_learning_rate=critic_learning_rate,
        discount=discount,
        target_update_period=target_update_period,
        use_sarsa_target=use_sarsa_target,
        logger=loggers.TerminalLogger('learner', time_delta=10.0),
    )

    # Define evaluator network.
    evaluator_network = networks['policy']

    # Create actor.
    actor = crr.CrrActor(
        policy_network=evaluator_network,
        random_key=jax.random.PRNGKey(seed + 1),
        variable_source=learner,
        adder=None,
    )

    # Create environment loop.
    env_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=actor,
        logger=loggers.TerminalLogger('environment_loop', time_delta=10.0),
    )

    # Run the environment loop.
    env_loop.run(num_episodes=None)

if __name__ == '__main__':
    main()
```