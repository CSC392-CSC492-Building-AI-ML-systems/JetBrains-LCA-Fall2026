```python
import acme
from acme import specs
from acme.agents.jax import crr
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.jax import variable_utils
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import functools

def add_next_action_extras(transitions):
    """Adds next_action to extras in transitions."""
    next_actions = np.roll(transitions['action'], shift=-1, axis=0)
    transitions['extras'] = transitions.get('extras', {})
    transitions['extras']['next_action'] = next_actions
    return transitions

def main(
    env_name='CartPole-v1',
    dataset_name='cartpole_demonstrations',
    batch_size=256,
    evaluation_period=1000,
    num_demonstration_episodes=10,
    random_seed=42,
    policy_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    discount=0.99,
    target_update_period=100,
    use_sarsa_target=False,
):
    # Create environment and spec
    environment = acme.make_environment(env_name)
    environment_spec = specs.make_environment_spec(environment)

    # Get demonstration dataset
    demonstration_dataset = crr.demonstration_dataset_factory(
        environment_spec=environment_spec,
        dataset_name=dataset_name,
        num_demonstration_episodes=num_demonstration_episodes,
        add_extras_fn=add_next_action_extras,
    )

    # Create networks
    def network_factory(spec):
        return crr.CRRNetworks(
            environment_spec=spec,
            policy_layer_sizes=(256, 256),
            critic_layer_sizes=(256, 256),
        )
    networks = network_factory(environment_spec)

    # Create logger
    logger = loggers.TerminalLogger(label='learner', time_delta=10.0)

    # Create CRR learner
    learner = crr.CRRLearner(
        networks=networks,
        random_key=jax.random.PRNGKey(random_seed),
        dataset=demonstration_dataset,
        batch_size=batch_size,
        policy_learning_rate=policy_learning_rate,
        critic_learning_rate=critic_learning_rate,
        discount=discount,
        target_update_period=target_update_period,
        use_sarsa_target=use_sarsa_target,
        logger=logger,
    )

    # Evaluator network (policy)
    evaluator_network = networks.policy_network

    # Create actor
    variable_client = variable_utils.VariableClient(learner, 'policy')
    actor = crr._LearningActor(
        policy_network=evaluator_network,
        variable_client=variable_client,
        random_key=jax.random.PRNGKey(random_seed + 1),
    )

    # Create environment loop
    env_loop = acme.EnvironmentLoop(
        environment=environment,
        actor=actor,
        logger=loggers.TerminalLogger(label='environment_loop', time_delta=10.0),
    )

    # Run environment loop
    env_loop.run(num_episodes=evaluation_period)

if __name__ == '__main__':
    main()
```