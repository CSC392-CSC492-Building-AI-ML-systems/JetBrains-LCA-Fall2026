```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from avalanche.benchmarks.classic import CLEAR
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics
from avalanche.benchmarks.utils import AvalancheDataset
import os
import random
import numpy as np

# Hyperparameters
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 10
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.1
SEED = 1234
EVAL_PROTOCOL = "ni"  # or "nc", "nic", etc.
MODEL_SAVE_DIR = "./models"
RESULTS_FILE = "./clear_results.txt"
ACCURACY_MATRIX_FILE = "./accuracy_matrix.pt"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Set seed depending on protocol
    if EVAL_PROTOCOL == "ni":
        set_seed(SEED)
    else:
        set_seed(SEED + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)

    # Normalization and transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Benchmark
    benchmark = CLEAR(
        './data',
        protocol=EVAL_PROTOCOL,
        train_transform=train_transform,
        eval_transform=test_transform,
        download=True
    )

    # Loggers
    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open(RESULTS_FILE, "w"))
    stdout_logger = InteractiveLogger()

    # Evaluation Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(stream=True),
        timing_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[tb_logger, text_logger, stdout_logger]
    )

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Continual learning strategy
    cl_strategy = Naive(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        train_mb_size=BATCH_SIZE,
        train_epochs=NUM_EPOCHS,
        eval_mb_size=BATCH_SIZE,
        device=device,
        evaluator=eval_plugin,
        eval_every=-1
    )

    # Training loop
    accuracy_matrix = []
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        cl_strategy.train(experience)
        scheduler.step()
        print(f"End training on experience {experience.current_experience}")

        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_exp_{experience.current_experience}.pth")
        torch.save(model.state_dict(), model_path)

        # Evaluate on test stream
        results = cl_strategy.eval(benchmark.test_stream)
        accs = []
        for exp in benchmark.test_stream:
            acc = results[f"Top1_Acc_Exp{exp.current_experience}/stream"]
            accs.append(acc)
        accuracy_matrix.append(accs)

    # Save accuracy matrix
    accuracy_matrix_tensor = torch.tensor(accuracy_matrix)
    torch.save(accuracy_matrix_tensor, ACCURACY_MATRIX_FILE)

    # Compute CLEAR metrics
    final_acc = accuracy_matrix_tensor[-1].mean().item()
    backward_transfer = (accuracy_matrix_tensor[-1] - accuracy_matrix_tensor.diag()).mean().item()
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\nFinal Average Accuracy: {final_acc:.4f}\n")
        f.write(f"Backward Transfer: {backward_transfer:.4f}\n")

if __name__ == "__main__":
    main()
```