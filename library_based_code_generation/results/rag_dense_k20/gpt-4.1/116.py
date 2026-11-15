```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from avalanche.benchmarks.classic import CLEAR
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics
from avalanche.training.strategies import Naive
import os
import random
import numpy as np

# Hyperparameters
hyperparams = {
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 64,
    "num_epochs": 10,
    "seed": 1234,
    "protocol": "ni",  # or "nc"
    "log_dir": "./logs",
    "model_save_dir": "./models"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def main():
    os.makedirs(hyperparams["log_dir"], exist_ok=True)
    os.makedirs(hyperparams["model_save_dir"], exist_ok=True)

    set_seed(hyperparams["seed"])
    device = get_device()

    # Model
    model = resnet18(pretrained=False, num_classes=10)
    model = model.to(device)

    # Transforms
    train_transform, test_transform = get_transforms()

    # Benchmark
    if hyperparams["protocol"] == "ni":
        protocol = "ni"
        seed = 1234
    else:
        protocol = "nc"
        seed = 5678
    set_seed(seed)
    benchmark = CLEAR(
        data_root="./data",
        protocol=protocol,
        train_transform=train_transform,
        eval_transform=test_transform
    )

    # Loggers
    tb_logger = TensorboardLogger(hyperparams["log_dir"])
    text_logger = TextLogger(open(os.path.join(hyperparams["log_dir"], "log.txt"), "a"))
    interactive_logger = InteractiveLogger()

    # Evaluation Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(save_image=False, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        cpu_usage_metrics(epoch=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[tb_logger, text_logger, interactive_logger]
    )

    # Optimizer and Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparams["lr"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Strategy
    strategy = Naive(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        train_mb_size=hyperparams["batch_size"],
        train_epochs=hyperparams["num_epochs"],
        eval_mb_size=hyperparams["batch_size"],
        device=device,
        evaluator=eval_plugin
    )

    # Training Loop
    accuracy_matrix = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"Start training on experience {exp_id}")
        strategy.train(experience)
        scheduler.step()
        print(f"End training on experience {exp_id}")

        # Save model
        model_path = os.path.join(hyperparams["model_save_dir"], f"model_exp_{exp_id}.pth")
        torch.save(model.state_dict(), model_path)

        # Evaluation
        results = strategy.eval(benchmark.test_stream)
        accs = []
        for test_exp in benchmark.test_stream:
            acc = results[f"Top1_Acc_Exp{test_exp.current_experience}/stream"]
            accs.append(acc)
        accuracy_matrix.append(accs)

    # Save accuracy matrix
    acc_matrix_path = os.path.join(hyperparams["log_dir"], "accuracy_matrix.txt")
    with open(acc_matrix_path, "w") as f:
        for row in accuracy_matrix:
            f.write(",".join([str(a) for a in row]) + "\n")

    # Compute CLEAR metrics
    accuracy_matrix_np = np.array(accuracy_matrix)
    final_acc = accuracy_matrix_np[-1, :].mean()
    backward_transfer = (accuracy_matrix_np[-1, :] - np.diag(accuracy_matrix_np)).mean()
    forward_transfer = (accuracy_matrix_np[0, :] - accuracy_matrix_np[:, 0]).mean()

    clear_metrics_path = os.path.join(hyperparams["log_dir"], "clear_metrics.txt")
    with open(clear_metrics_path, "w") as f:
        f.write(f"Final Accuracy: {final_acc:.4f}\n")
        f.write(f"Backward Transfer: {backward_transfer:.4f}\n")
        f.write(f"Forward Transfer: {forward_transfer:.4f}\n")

if __name__ == "__main__":
    main()
```