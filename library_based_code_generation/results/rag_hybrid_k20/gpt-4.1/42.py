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
import random

def add_next_action_extras(transitions):
    """Adds 'next_action' to the extras of each transition."""
    new_transitions = []
    for i in range(len(transitions) - 1):
        transition = transitions[i]
        next_transition = transitions[i + 1]
        extras = dict(transition.extras)
        extras['next_action'] = next_transition.action
        new_transition = transition._replace(extras=extras)
        new_transitions.append(new_transition)
    # For the last transition, next_action is None or zeros
    last_transition = transitions[-1]
    extras = dict(last_transition.extras)
    extras['next_action'] = np.zeros_like(last_transition.action)
    new_transition = last_transition._replace(extras=extras)
    new_transitions.append(new_transition)
    return new_transitions

def main(
    environment_name='CartPole-v1',
    dataset_name='cartpole_demonstrations',
    batch_size=256,
    evaluation_period=1000,
    num_demonstration_episodes=10,
    random_seed=42,
    policy_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    discount=0.99,
    target_update_period=100,
    use_sarsa_target=False,
):

    # Set random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    key = jax.random.PRNGKey(random_seed)

    # Environment factory
    def environment_factory(seed=random_seed):
        env = acme.make_environment(environment_name, seed=seed)
        return env

    # Environment spec
    environment = environment_factory()
    environment_spec = specs.make_environment_spec(environment)

    # Demonstration dataset factory
    def demonstration_dataset_factory():
        # This function should return a dataset of transitions
        # For demonstration, we use a placeholder function
        # Replace with actual dataset loading as needed
        dataset = []
        for _ in range(num_demonstration_episodes):
            episode = []
            timestep = environment.reset()
            done = False
            while not timestep.last():
                action = environment.action_spec().generate_value()
                next_timestep = environment.step(action)
                transition = acme.types.Transition(
                    observation=timestep.observation,
                    action=action,
                    reward=next_timestep.reward,
                    discount=next_timestep.discount,
                    next_observation=next_timestep.observation,
                    extras={}
                )
                episode.append(transition)
                timestep = next_timestep
            episode = add_next_action_extras(episode)
            dataset.extend(episode)
        return dataset

    # Networks
    def network_factory(spec: specs.EnvironmentSpec):
        # Simple MLPs for policy and critic
        policy_network = networks_lib.FeedForwardNetwork(
            init=lambda key: {},
            apply=lambda params, obs: jax.nn.softmax(jnp.ones(spec.actions.num_values))
        )
        critic_network = networks_lib.FeedForwardNetwork(
            init=lambda key: {},
            apply=lambda params, obs, action: jnp.array(0.0)
        )
        return crr.CRRNetworks(
            policy_network=policy_network,
            critic_network=critic_network,
        )

    networks = network_factory(environment_spec)

    # Logger
    logger = loggers.TerminalLogger(label='learner', time_delta=10.0)

    # Learner
    learner = crr.CRRLearner(
        networks=networks,
        random_key=key,
        demonstration_dataset=demonstration_dataset_factory(),
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

    # Variable client for actor
    variable_client = variable_utils.VariableClient(learner, 'policy')

    # Actor
    actor = crr.CRRActor(
        policy_network=evaluator_network,
        variable_client=variable_client,
        random_key=key,
        environment_spec=environment_spec,
    )

    # Environment loop
    loop = acme.EnvironmentLoop(
        environment=environment_factory(),
        actor=actor,
        logger=loggers.TerminalLogger(label='environment_loop', time_delta=10.0),
    )

    # Run the environment loop
    loop.run(num_episodes=evaluation_period)

if __name__ == '__main__':
    main()
```