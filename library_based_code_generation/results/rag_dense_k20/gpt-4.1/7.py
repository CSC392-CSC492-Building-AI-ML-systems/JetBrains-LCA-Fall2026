```python
import argparse
import torch
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.logging import TextLogger, InteractiveLogger, CsvLogger, TensorboardLogger
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, class_accuracy_metrics, amca_metrics,
    forgetting_metrics, bwt_metrics, fwt_metrics, cpu_usage_metrics,
    timing_metrics, ram_usage_metrics, gpu_usage_metrics, disk_usage_metrics,
    mac_metrics, labels_repartition_metrics
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computations (e.g., "cuda:0" or "cpu")')
    args = parser.parse_args()
    device = torch.device(args.device)

    # 2. Define transformations for training and testing data.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3. Create a benchmark using the MNIST dataset with the defined transformations.
    benchmark = SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        train_transform=train_transform,
        eval_transform=test_transform
    )

    # 4. Create a simple MLP model with the number of classes equal to the number of classes in the benchmark.
    n_classes = benchmark.n_classes
    model = SimpleMLP(num_classes=n_classes)

    # 5. Define various loggers.
    text_logger = TextLogger(open('log.txt', 'w'))
    interactive_logger = InteractiveLogger()
    csv_logger = CsvLogger('metrics.csv')
    tensorboard_logger = TensorboardLogger('tb_data')

    loggers = [text_logger, interactive_logger, csv_logger, tensorboard_logger]

    # 6. Define an evaluation plugin with a wide range of metrics.
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        class_accuracy_metrics(epoch=True, experience=True, stream=True),
        amca_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        fwt_metrics(experience=True, stream=True),
        cpu_usage_metrics(epoch=True, experience=True),
        timing_metrics(epoch=True, experience=True),
        ram_usage_metrics(epoch=True, experience=True),
        gpu_usage_metrics(epoch=True, experience=True),
        disk_usage_metrics(epoch=True, experience=True),
        mac_metrics(epoch=True, experience=True),
        labels_repartition_metrics(epoch=True, experience=True),
        loggers=loggers
    )

    # 7. Create a Naive continual learning strategy.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin
    )

    # 8. Train and evaluate on the benchmark's streams.
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience)
        print(f"End training on experience {experience.current_experience}")

        print("Computing accuracy on the test set")
        results = strategy.eval(benchmark.test_stream)
        print(f"Results after experience {experience.current_experience}: {results}")

    # 9. Print all metrics stored by the evaluation plugin.
    print("All metrics:")
    for k, v in eval_plugin.get_all_metrics().items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
```