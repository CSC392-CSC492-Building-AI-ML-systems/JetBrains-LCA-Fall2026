```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from avalanche.benchmarks.classic import CLEAR
from avalanche.models import avalanche_resnet18
from avalanche.training.strategies import Naive
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
    confusion_matrix_metrics,
    timing_metrics,
    cpu_usage_metrics,
    disk_usage_metrics
)
from avalanche.training.plugins import EvaluationPlugin
import numpy as np
import random
import os

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
MODEL_SAVE_DIR = "./clear_checkpoints"
LOG_DIR = "./clear_logs"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_clear_benchmark(protocol, seed):
    return CLEAR(
        protocol=protocol,
        seed=seed,
        return_task_id=True
    )

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = avalanche_resnet18(num_classes=10)
    model.to(device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Benchmark
    benchmark = get_clear_benchmark(EVAL_PROTOCOL, SEED)
    benchmark.train_stream = [exp.with_transforms(train_transform) for exp in benchmark.train_stream]
    benchmark.test_stream = [exp.with_transforms(test_transform) for exp in benchmark.test_stream]

    # Loggers
    tb_logger = TensorboardLogger(log_dir=LOG_DIR)
    text_logger = TextLogger(open(os.path.join(LOG_DIR, "log.txt"), "w"))
    interactive_logger = InteractiveLogger()

    # Evaluation Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(save_image=False, stream=True),
        timing_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[tb_logger, text_logger, interactive_logger]
    )

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Continual Learning Strategy
    cl_strategy = Naive(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        train_mb_size=BATCH_SIZE,
        train_epochs=NUM_EPOCHS,
        eval_mb_size=BATCH_SIZE,
        device=device,
        evaluator=eval_plugin
    )

    # Training Loop
    accuracy_matrix = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"\n--- Training on experience {exp_id} (Task {experience.task_label}) ---")
        cl_strategy.train(experience)
        scheduler.step()
        results = cl_strategy.eval(benchmark.test_stream)
        # Collect accuracy for CLEAR metrics
        accs = []
        for test_exp in benchmark.test_stream:
            acc = results['Top1_Acc_Exp/eval_phase/test_stream/Task' + str(test_exp.task_label) + '/Exp' + str(test_exp.current_experience)]
            accs.append(acc)
        accuracy_matrix.append(accs)
        # Save model
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"model_exp_{exp_id}.pth"))

    # Compute CLEAR metrics
    accuracy_matrix = np.array(accuracy_matrix)
    num_exps = accuracy_matrix.shape[0]
    final_acc = accuracy_matrix[-1]
    avg_final_acc = np.mean(final_acc)
    backward_transfer = np.mean([final_acc[i] - accuracy_matrix[i, i] for i in range(num_exps)])
    forward_transfer = np.mean([accuracy_matrix[i, i] - accuracy_matrix[0, i] for i in range(1, num_exps)])

    # Log CLEAR metrics
    with open(os.path.join(LOG_DIR, "clear_metrics.txt"), "w") as f:
        f.write("Accuracy Matrix:\n")
        f.write(str(accuracy_matrix) + "\n")
        f.write(f"Average Final Accuracy: {avg_final_acc:.4f}\n")
        f.write(f"Backward Transfer: {backward_transfer:.4f}\n")
        f.write(f"Forward Transfer: {forward_transfer:.4f}\n")

if __name__ == "__main__":
    main()
```