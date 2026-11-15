```python
import argparse
import torch
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.generators import online_scenario
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Replay
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.evaluation.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='CUDA device to use, -1 for CPU')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if args.cuda >= 0 and torch.cuda.is_available() else 'cpu')

# Transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
mnist_train = MNIST(root='./data', train=True, download=True, transform=train_transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=test_transform)

# Online scenario: 100 mini-batches of size 256
online_benchmark = online_scenario(
    train_dataset=mnist_train,
    test_dataset=mnist_test,
    n_experiences=100,
    experience_size=256,
    shuffle=True,
    seed=1234,
    drop_last=False
)

# Model
model = SimpleMLP(num_classes=10)

# Metrics and logger
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# Replay strategy with ReservoirSamplingBuffer
strategy = Replay(
    model,
    torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    torch.nn.CrossEntropyLoss(),
    train_mb_size=32,
    train_epochs=1,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin,
    mem_size=500,
    storage_policy=ReservoirSamplingBuffer(max_size=500)
)

# Training loop
results = []
for experience in online_benchmark.train_stream:
    strategy.train(experience)
    eval_results = strategy.eval(online_benchmark.test_stream)
    results.append(eval_results)
```