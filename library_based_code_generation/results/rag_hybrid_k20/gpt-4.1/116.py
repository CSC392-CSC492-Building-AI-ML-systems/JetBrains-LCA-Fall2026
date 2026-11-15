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
import numpy as np
import random
import os

# Hyperparameters
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 10
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.1
SEED = 1234
MODEL_SAVE_DIR = "./clear_models"
RESULTS_FILE = "./clear_results.txt"
ACCURACY_MATRIX_FILE = "./clear_accuracy_matrix.npy"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define normalization and transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Create CLEAR benchmark
    benchmark = CLEAR(
        train_transform=train_transform,
        eval_transform=test_transform
    )

    # Initialize model
    model = models.resnet18(pretrained=False, num_classes=benchmark.n_classes)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Loggers
    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open(RESULTS_FILE, "a"))
    stdout_logger = InteractiveLogger()

    # Evaluation plugin
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

    # Continual learning strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=BATCH_SIZE,
        train_epochs=NUM_EPOCHS,
        eval_mb_size=BATCH_SIZE,
        device=device,
        evaluator=eval_plugin
    )

    # Training loop
    accuracy_matrix = []
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    for idx, experience in enumerate(benchmark.train_stream):
        print(f"Start training on experience {idx}")
        strategy.train(experience)
        scheduler.step()
        # Save model
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"model_exp_{idx}.pth"))
        # Evaluate on test stream
        results = strategy.eval(benchmark.test_stream)
        # Collect accuracy for each test experience
        accs = []
        for test_exp in benchmark.test_stream:
            acc = results[f"Top1_Acc_Exp{test_exp.current_experience}/stream"]
            accs.append(acc)
        accuracy_matrix.append(accs)

    accuracy_matrix = np.array(accuracy_matrix)
    np.save(ACCURACY_MATRIX_FILE, accuracy_matrix)

    # Compute CLEAR metrics
    final_acc = accuracy_matrix[-1].mean()
    backward_transfer = (accuracy_matrix[-1] - np.diag(accuracy_matrix)).mean()
    forward_transfer = (np.diag(accuracy_matrix) - accuracy_matrix[0]).mean()

    with open(RESULTS_FILE, "a") as f:
        f.write(f"\nFinal Average Accuracy: {final_acc:.4f}\n")
        f.write(f"Backward Transfer: {backward_transfer:.4f}\n")
        f.write(f"Forward Transfer: {forward_transfer:.4f}\n")

if __name__ == "__main__":
    main()
```