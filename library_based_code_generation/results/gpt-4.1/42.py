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
import optax
from typing import Dict, Any, Callable

def add_next_action_extras(transitions: Dict[str, Any]) -> Dict[str, Any]:
    """Adds 'next_action' to the extras dict in transitions."""
    next_actions = np.roll(transitions['action'], shift=-1, axis=0)
    # For the last transition, next_action is zeros
    next_actions[-1] = 0
    if 'extras' not in transitions:
        transitions['extras'] = {}
    transitions['extras']['next_action'] = next_actions
    return transitions

def make_environment(env_name: str) -> dm_env.Environment:
    import gym
    env = gym.make(env_name)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

def get_demonstrations(dataset_name: str, num_episodes: int):
    # Placeholder: Replace with actual dataset loading logic
    # Should return a list of episodes, each as a list of transitions
    # Each transition is a dict with keys: observation, action, reward, discount, next_observation, etc.
    # For demonstration, we return an empty list
    return []

def make_networks(env_spec: specs.EnvironmentSpec, policy_lr: float, critic_lr: float) -> crr.CRRNetworks:
    def policy_network(obs):
        mlp = networks_lib.MLP([256, 256, env_spec.actions.num_values])
        return mlp(obs)
    def critic_network(obs, action):
        mlp = networks_lib.MLP([256, 256, 1])
        x = jnp.concatenate([obs, action], axis=-1)
        return mlp(x)
    return crr.CRRNetworks(
        policy_network=policy_network,
        critic_network=critic_network,
        policy_optimizer=optax.adam(policy_lr),
        critic_optimizer=optax.adam(critic_lr),
    )

def main(
    env_name: str = 'CartPole-v1',
    dataset_name: str = 'cartpole_demo',
    batch_size: int = 256,
    evaluation_period: int = 1000,
    num_demonstration_episodes: int = 10,
    random_seed: int = 42,
    policy_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    discount: float = 0.99,
    target_update_period: int = 100,
    use_sarsa_target: bool = False,
):
    # Set random seeds
    np.random.seed(random_seed)
    key = jax.random.PRNGKey(random_seed)

    # Create environment and spec
    environment = make_environment(env_name)
    environment_spec = specs.make_environment_spec(environment)

    # Get demonstrations
    demonstrations = get_demonstrations(dataset_name, num_demonstration_episodes)

    # Create networks
    networks = make_networks(
        environment_spec,
        policy_lr=policy_learning_rate,
        critic_lr=critic_learning_rate,
    )

    # Create the CRR learner
    learner = crr.CRRLearner(
        networks=networks,
        random_key=key,
        dataset=demonstrations,
        batch_size=batch_size,
        discount=discount,
        target_update_period=target_update_period,
        use_sarsa_target=use_sarsa_target,
        add_next_action_extras_fn=add_next_action_extras,
        logger=loggers.TerminalLogger('learner', time_delta=10.0),
    )

    # Evaluator network (policy)
    def evaluator_network(params, obs):
        return networks.policy_network.apply(params, obs)

    # Create actor
    actor = crr.CRRActor(
        policy_network=networks.policy_network,
        variable_source=learner,
        random_key=key,
    )

    # Create environment loop
    env_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=actor,
        logger=loggers.TerminalLogger('environment_loop', time_delta=10.0),
    )

    # Run the environment loop
    for _ in range(evaluation_period):
        env_loop.run_episode()

if __name__ == '__main__':
    main()
```