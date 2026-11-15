```python
import functools
from absl import app, flags
import acme
from acme import specs
from acme.agents.jax import behavioral_cloning
from acme.jax import experiments
from acme.jax import utils as acme_utils
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import dm_env
import gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstration episodes.')
flags.DEFINE_integer('learning_steps', 100_000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for MLP.')
flags.DEFINE_list('mlp_layers', ['256', '256'], 'Sizes of MLP hidden layers.')
flags.DEFINE_bool('distributed', False, 'Run experiment in distributed mode.')

def make_environment_factory(env_name):
    def environment_factory(seed: int) -> dm_env.Environment:
        env = gym.make(env_name)
        env.seed(seed)
        return acme.wrappers.GymWrapper(env)
    return environment_factory

def make_demonstrations_factory(env_factory, num_demonstrations):
    def demonstrations_factory():
        env = env_factory(seed=0)
        spec = specs.make_environment_spec(env)
        demonstrations = []
        for _ in range(num_demonstrations):
            timestep = env.reset()
            episode = []
            while not timestep.last():
                action = env.action_spec().generate_value()
                episode.append((timestep.observation, action))
                timestep = env.step(action)
            demonstrations.extend(episode)
        observations = np.stack([obs for obs, _ in demonstrations])
        actions = np.stack([act for _, act in demonstrations])
        dataset = {'observations': observations, 'actions': actions}
        return dataset
    return demonstrations_factory

def make_network_factory(mlp_layers, dropout_rate, action_spec):
    def network_fn(obs, is_training=True):
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP([int(size) for size in mlp_layers], activation=jax.nn.relu),
            hk.dropout(hk.next_rng_key(), dropout_rate, jnp.ones([1])) if is_training else lambda x: x,
            hk.Linear(action_spec.num_values if hasattr(action_spec, 'num_values') else action_spec.shape[0])
        ])
        return mlp(obs)
    def network_factory(spec):
        return hk.without_apply_rng(hk.transform(network_fn))
    return network_factory

def build_experiment_config():
    env_factory = make_environment_factory(FLAGS.env_name)
    dummy_env = env_factory(seed=0)
    env_spec = specs.make_environment_spec(dummy_env)
    demonstrations_factory = make_demonstrations_factory(env_factory, FLAGS.num_demonstrations)
    network_factory = make_network_factory(FLAGS.mlp_layers, FLAGS.dropout_rate, env_spec.actions)
    bc_builder = behavioral_cloning.BCBuilder(
        demonstrations=demonstrations_factory,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        num_steps=FLAGS.learning_steps,
    )
    return experiments.ExperimentConfig(
        builder=bc_builder,
        environment_factory=env_factory,
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=None,
        environment_spec=env_spec,
    )

def main(_):
    config = build_experiment_config()
    if FLAGS.distributed:
        program = experiments.make_distributed_experiment(config)
        acme.launch_program(program)
    else:
        experiments.run_experiment(config)

if __name__ == '__main__':
    app.run(main)
```