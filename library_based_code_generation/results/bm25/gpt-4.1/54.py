```python
import argparse
import torch
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.scenarios import OnlineCLScenario
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Replay
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.evaluation import EvaluationPlugin
from torchvision import transforms

def main():
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

    # Online Continual Learning Scenario
    scenario = OnlineCLScenario(
        train_dataset=mnist_train,
        test_dataset=mnist_test,
        task_labels=False,
        batch_size=32,
        experience_size=1,  # online: 1 sample per experience
        shuffle=True
    )

    # Model
    model = SimpleMLP(num_classes=10)
    model.to(device)

    # Metrics and Logger
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[interactive_logger]
    )

    # Replay Strategy with ReservoirSamplingBuffer
    strategy = Replay(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        torch.nn.CrossEntropyLoss(),
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
        mem_size=200,
        storage_policy=ReservoirSamplingBuffer(max_size=200)
    )

    # Training loop
    results = []
    for experience in scenario.train_stream:
        strategy.train(experience)
        eval_results = strategy.eval(scenario.test_stream)
        results.append(eval_results)

if __name__ == "__main__":
    main()
```