```python
import functools
from absl import app, flags
import acme
from acme import specs
from acme.agents.jax.bc import builder as bc_builder
from acme.jax import experiments
from acme.jax import networks as acme_networks
from acme.jax import utils as acme_utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstration episodes.')
flags.DEFINE_integer('learning_steps', 10000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for MLP.')
flags.DEFINE_list('mlp_layers', ['256', '256'], 'Sizes of MLP hidden layers.')
flags.DEFINE_bool('distributed', False, 'Whether to run distributed experiment.')

def make_environment_factory(env_name):
    def environment_factory(seed: int):
        import gym
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return environment_factory

def make_demonstration_dataset_factory(env_factory, num_demonstrations):
    def demonstration_dataset_factory(batch_size: int):
        env = env_factory(seed=0)
        obs_spec = specs.make_environment_spec(env).observations
        action_spec = specs.make_environment_spec(env).actions
        demonstrations = []
        for _ in range(num_demonstrations):
            obs = env.reset()
            done = False
            episode = []
            while not done:
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                episode.append({'observation': obs, 'action': action})
                obs = next_obs
            demonstrations.extend(episode)
        # Convert to arrays
        observations = np.array([d['observation'] for d in demonstrations])
        actions = np.array([d['action'] for d in demonstrations])
        def generator():
            idxs = np.arange(len(observations))
            while True:
                np.random.shuffle(idxs)
                for i in range(0, len(idxs), batch_size):
                    batch_idxs = idxs[i:i+batch_size]
                    yield {
                        'observations': observations[batch_idxs],
                        'actions': actions[batch_idxs]
                    }
        return generator()
    return demonstration_dataset_factory

def make_network_factory(mlp_layers, dropout_rate):
    def network_factory(spec: specs.EnvironmentSpec):
        num_actions = spec.actions.num_values if hasattr(spec.actions, 'num_values') else spec.actions.shape[0]
        def net_fn(obs, is_training=True):
            mlp = hk.Sequential([
                hk.Flatten(),
                hk.nets.MLP([int(x) for x in mlp_layers], activation=jax.nn.relu, activate_final=True),
                hk.dropout(hk.next_rng_key(), dropout_rate, jnp.ones([int(mlp_layers[-1])])),
                hk.Linear(num_actions)
            ])
            return mlp(obs)
        return acme_networks.FeedForwardNetwork(
            init=hk.transform(net_fn).init,
            apply=hk.transform(net_fn).apply
        )
    return network_factory

def build_experiment_config():
    env_factory = make_environment_factory(FLAGS.env_name)
    demo_factory = make_demonstration_dataset_factory(env_factory, FLAGS.num_demonstrations)
    network_factory = make_network_factory(FLAGS.mlp_layers, FLAGS.dropout_rate)
    bc = bc_builder.BCBuilder(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.learning_steps
    )
    return experiments.OfflineExperimentConfig(
        builder=bc,
        environment_factory=env_factory,
        network_factory=network_factory,
        demonstration_dataset_factory=demo_factory,
        logger_factory=loggers.make_default_logger,
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