```python
import functools
from absl import app, flags
import acme
from acme import specs
from acme.agents.jax.bc import builder as bc_builder
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstration episodes.')
flags.DEFINE_integer('learning_steps', 100_000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for MLP.')
flags.DEFINE_list('mlp_layers', ['256', '256'], 'Sizes of MLP hidden layers.')
flags.DEFINE_bool('distributed', False, 'Whether to run distributed experiment.')

def environment_factory(seed: int):
    import gym
    env = gym.make(FLAGS.env_name)
    env.seed(seed)
    return env

def demonstration_dataset_factory(environment_factory, num_demonstrations):
    def dataset_fn():
        env = environment_factory(0)
        dataset = []
        for _ in range(num_demonstrations):
            obs = env.reset()
            done = False
            episode = []
            while not done:
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                episode.append((obs, action, reward, next_obs, done))
                obs = next_obs
            dataset.extend(episode)
        return dataset
    return dataset_fn

def mlp_with_dropout(obs_spec, action_spec):
    num_actions = action_spec.num_values if hasattr(action_spec, 'num_values') else action_spec.shape[0]
    mlp_layers = [int(x) for x in FLAGS.mlp_layers]
    dropout_rate = FLAGS.dropout_rate

    def net_fn(inputs, is_training=True):
        x = hk.Flatten()(inputs)
        for width in mlp_layers:
            x = hk.Linear(width)(x)
            x = jax.nn.relu(x)
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x) if is_training else x
        logits = hk.Linear(num_actions)(x)
        return logits

    return hk.without_apply_rng(hk.transform(net_fn))

def network_factory(spec: specs.EnvironmentSpec):
    net = mlp_with_dropout(spec.observations, spec.actions)
    def apply_fn(params, obs, is_training=True):
        return net.apply(params, obs, is_training)
    dummy_obs = jax_utils.zeros_like(spec.observations)
    params = net.init(jax.random.PRNGKey(0), dummy_obs)
    return networks_lib.FeedForwardNetwork(init=lambda rng: params, apply=apply_fn)

def build_experiment_config():
    env_factory = functools.partial(environment_factory)
    demo_factory = demonstration_dataset_factory(env_factory, FLAGS.num_demonstrations)
    spec = specs.make_environment_spec(env_factory(0))
    net_factory = functools.partial(network_factory, spec)
    bc = bc_builder.BCBuilder(
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        demonstration_dataset_factory=demo_factory,
    )
    return experiments.ExperimentConfig(
        builder=bc,
        environment_factory=env_factory,
        network_factory=net_factory,
        seed=0,
        max_num_actor_steps=FLAGS.learning_steps,
        logger_factory=experiments.create_experiment_logger_factory('bc_experiment'),
    )

def main(_):
    config = build_experiment_config()
    if FLAGS.distributed:
        exp = experiments.make_distributed_experiment(config)
        exp.launch()
    else:
        experiments.run_experiment(config)

if __name__ == '__main__':
    app.run(main)
```