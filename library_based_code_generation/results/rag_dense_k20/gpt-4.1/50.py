```python
import acme
from acme import specs
from acme.agents.jax import bc
from acme.jax import experiments
from acme.jax.networks import MLP
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstration episodes.')
flags.DEFINE_integer('learning_steps', 100_000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for MLP.')
flags.DEFINE_list('mlp_layers', ['256', '256'], 'Sizes of MLP hidden layers.')
flags.DEFINE_bool('distributed', False, 'Whether to run distributed experiment.')

def environment_factory(seed: int = 0) -> dm_env.Environment:
    import gym
    env = gym.make(FLAGS.env_name)
    env.seed(seed)
    return dm_env.from_gym(env)

def demonstration_dataset_factory(environment_spec: specs.EnvironmentSpec):
    def _generator():
        env = environment_factory()
        for _ in range(FLAGS.num_demonstrations):
            timestep = env.reset()
            episode = []
            while not timestep.last():
                action = env.action_spec().generate_value()
                episode.append((timestep.observation, action))
                timestep = env.step(action)
            for obs, act in episode:
                yield {'observation': obs, 'action': act}
    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature={
            'observation': tf.TensorSpec(environment_spec.observations.shape, environment_spec.observations.dtype),
            'action': tf.TensorSpec(environment_spec.actions.shape, environment_spec.actions.dtype),
        }
    )
    ds = ds.shuffle(1000).batch(FLAGS.batch_size).repeat()
    return ds

def network_factory(environment_spec: specs.EnvironmentSpec, is_training: bool = True):
    num_actions = environment_spec.actions.num_values if hasattr(environment_spec.actions, 'num_values') else environment_spec.actions.shape[0]
    mlp_layers = [int(x) for x in FLAGS.mlp_layers]
    dropout_rate = FLAGS.dropout_rate if is_training else 0.0

    def net_fn(obs):
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP(mlp_layers, activation=jax.nn.relu, activate_final=True),
            hk.Dropout(rate=dropout_rate),
            hk.Linear(num_actions),
        ])
        return mlp(obs)
    return hk.without_apply_rng(hk.transform(net_fn))

def build_experiment_config():
    env_factory = environment_factory
    spec = specs.make_environment_spec(env_factory())
    net_factory = lambda: network_factory(spec)
    dataset_factory = lambda: demonstration_dataset_factory(spec)
    bc_config = bc.BCConfig(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.learning_steps,
    )
    builder = bc.BCBuilder(bc_config)
    return experiments.OfflineExperimentConfig(
        builder=builder,
        environment_factory=env_factory,
        network_factory=net_factory,
        demonstration_dataset_factory=dataset_factory,
        logger_factory=loggers.TerminalLogger,
        seed=0,
        max_num_actor_steps=None,
    )

def main(_):
    config = build_experiment_config()
    if FLAGS.distributed:
        experiment = experiments.make_distributed_offline_experiment(config)
        experiment.launch()
    else:
        experiments.run_offline_experiment(config)

if __name__ == '__main__':
    app.run(main)
```