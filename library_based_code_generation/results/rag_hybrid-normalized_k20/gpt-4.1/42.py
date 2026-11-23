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
import optax
import functools

def add_next_action_extras(transitions):
    """Adds 'next_action' to extras in transitions."""
    next_actions = np.roll(transitions['action'], shift=-1, axis=0)
    # Last transition's next_action is dummy (e.g., zeros)
    next_actions[-1] = 0
    if 'extras' not in transitions:
        transitions['extras'] = {}
    transitions['extras']['next_action'] = next_actions
    return transitions

def main(
    environment_name='CartPole-v1',
    dataset_name='cartpole_demonstrations',
    batch_size=256,
    evaluation_period=10,
    num_demonstration_episodes=10,
    random_seed=0,
    policy_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    discount=0.99,
    target_update_period=100,
    use_sarsa_target=False,
):

    # Environment factory
    def environment_factory(seed):
        env = acme.make_environment(environment_name, seed=seed)
        return env

    # Environment spec
    environment = environment_factory(random_seed)
    environment_spec = specs.make_environment_spec(environment)

    # Demonstration dataset
    dataset = acme.demonstration_dataset_factory(
        environment_name=environment_name,
        dataset_name=dataset_name,
        num_episodes=num_demonstration_episodes,
        add_extras_fn=add_next_action_extras,
    )

    # Networks
    def network_factory(spec: specs.EnvironmentSpec):
        policy_network = networks_lib.make_policy_network(spec)
        critic_network = networks_lib.make_critic_network(spec)
        return crr.CRRNetworks(
            policy_network=policy_network,
            critic_network=critic_network,
        )

    networks = network_factory(environment_spec)

    # Learner
    learner = crr.CRRLearner(
        networks=networks,
        random_key=jax.random.PRNGKey(random_seed),
        dataset=dataset,
        batch_size=batch_size,
        policy_optimizer=optax.adam(policy_learning_rate),
        critic_optimizer=optax.adam(critic_learning_rate),
        discount=discount,
        target_update_period=target_update_period,
        use_sarsa_target=use_sarsa_target,
        logger=loggers.TerminalLogger('learner', time_delta=1.0),
    )

    # Evaluator network (policy)
    evaluator_network = networks.policy_network

    # Actor
    variable_source = variable_utils.VariableSourceFromLearner(learner)
    actor = crr.CRRActor(
        policy_network=evaluator_network,
        variable_source=variable_source,
        random_key=jax.random.PRNGKey(random_seed + 1),
        environment_spec=environment_spec,
    )

    # Environment loop
    loop = acme.EnvironmentLoop(
        environment=environment_factory(random_seed + 2),
        actor=actor,
        logger=loggers.TerminalLogger('environment_loop', time_delta=1.0),
    )

    # Run the environment loop
    loop.run(num_episodes=evaluation_period)

if __name__ == '__main__':
    main()
```