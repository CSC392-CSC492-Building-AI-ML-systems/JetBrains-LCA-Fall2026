```python
import functools
from absl import app, flags
import acme
from acme import specs
from acme.agents.jax import behavioral_cloning
from acme.jax import experiments
from acme.jax import utils as acme_utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax
import dm_env
import gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstration episodes.')
flags.DEFINE_integer('learning_steps', 100_000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for MLP.')
flags.DEFINE_list('mlp_layers', ['256', '256'], 'Sizes of MLP hidden layers.')
flags.DEFINE_bool('distributed', False, 'Whether to run distributed experiment.')

def make_environment_factory(env_name):
    def environment_factory(seed: int = 0) -> dm_env.Environment:
        env = gym.make(env_name)
        env.seed(seed)
        return acme.wrappers.GymWrapper(env)
    return environment_factory

def make_demonstration_dataset_factory(env_factory, num_demonstrations):
    def demonstration_dataset_factory():
        env = env_factory()
        dataset = []
        for _ in range(num_demonstrations):
            timestep = env.reset()
            episode = []
            while not timestep.last():
                action = env.action_spec().generate_value()
                next_timestep = env.step(action)
                episode.append((timestep, action, next_timestep))
                timestep = next_timestep
            dataset.extend(episode)
        # Convert to Acme's demonstration dataset format
        observations = np.array([t[0].observation for t in dataset])
        actions = np.array([t[1] for t in dataset])
        return {'observations': observations, 'actions': actions}
    return demonstration_dataset_factory

def make_network_factory(mlp_layers, dropout_rate, action_spec):
    def network_fn(obs):
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP([int(size) for size in mlp_layers], activation=jax.nn.relu),
            hk.Dropout(rate=dropout_rate),
            hk.Linear(action_spec.num_values if hasattr(action_spec, 'num_values') else action_spec.shape[0])
        ])
        return mlp(obs)
    def network_factory(spec: specs.EnvironmentSpec):
        return hk.without_apply_rng(hk.transform(network_fn))
    return network_factory

def build_experiment_config():
    env_factory = make_environment_factory(FLAGS.env_name)
    dummy_env = env_factory()
    env_spec = specs.make_environment_spec(dummy_env)
    demonstration_dataset_factory = make_demonstration_dataset_factory(env_factory, FLAGS.num_demonstrations)
    network_factory = make_network_factory(FLAGS.mlp_layers, FLAGS.dropout_rate, env_spec.actions)
    bc_builder = behavioral_cloning.BCBuilder(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.learning_steps,
    )
    logger_factory = experiments.create_experiment_logger_factory()
    return experiments.ExperimentConfig(
        builder=bc_builder,
        environment_factory=env_factory,
        network_factory=network_factory,
        demonstration_dataset_factory=demonstration_dataset_factory,
        logger_factory=logger_factory,
        max_num_actor_steps=FLAGS.learning_steps,
        seed=0,
    )

def main(_):
    experiment_config = build_experiment_config()
    if FLAGS.distributed:
        experiment = experiments.make_distributed_offline_experiment(experiment_config)
        experiment.launch()
    else:
        experiments.run_offline_experiment(experiment_config)

if __name__ == '__main__':
    app.run(main)
```