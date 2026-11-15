```python
import os
import torch
import random
import numpy as np
from avalanche.benchmarks.classic import CLEAR
from avalanche.models import avalanche_resnet18
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
    confusion_matrix_metrics,
    timing_metrics,
    cpu_usage_metrics,
    disk_usage_metrics
)
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

# Hyperparameters
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 10
STEP_SIZE = 5
GAMMA = 0.1
SEED = 1234
EVAL_PROTOCOL = "ni"  # options: "ni", "nc", "nic"
LOG_DIR = "./logs"
MODEL_SAVE_DIR = "./models"
RESULTS_FILE = "./results.txt"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Set seed depending on protocol
    if EVAL_PROTOCOL == "ni":
        seed = 1234
    elif EVAL_PROTOCOL == "nc":
        seed = 5678
    elif EVAL_PROTOCOL == "nic":
        seed = 91011
    else:
        seed = SEED
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = avalanche_resnet18(num_classes=10)
    model.to(device)

    # Normalization and transformations
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Benchmark
    benchmark = CLEAR(
        './data',
        protocol=EVAL_PROTOCOL,
        train_transform=train_transform,
        eval_transform=test_transform,
        seed=seed
    )

    # Loggers
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=LOG_DIR)
    text_logger = TextLogger(open(os.path.join(LOG_DIR, "log.txt"), "w"))
    interactive_logger = InteractiveLogger()

    # Evaluation Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(stream=True, save_image=False),
        timing_metrics(epoch=True),
        cpu_usage_metrics(epoch=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[tb_logger, text_logger, interactive_logger]
    )

    # Optimizer and LR scheduler
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Continual learning strategy
    strategy = Naive(
        model,
        optimizer,
        train_mb_size=BATCH_SIZE,
        train_epochs=NUM_EPOCHS,
        eval_mb_size=BATCH_SIZE,
        device=device,
        evaluator=eval_plugin,
        eval_every=1,
        plugins=[],
        criterion=torch.nn.CrossEntropyLoss()
    )

    # Training loop
    accuracy_matrix = []
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience, num_workers=4)
        print(f"End training on experience {experience.current_experience}")

        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_exp_{experience.current_experience}.pth")
        torch.save(model.state_dict(), model_path)

        # Evaluation
        results = strategy.eval(benchmark.test_stream)
        accs = []
        for exp_id, exp in enumerate(benchmark.test_stream):
            acc = results[f"Top1_Acc_Exp{exp_id}"]
            accs.append(acc)
        accuracy_matrix.append(accs)

    # Compute CLEAR metrics
    accuracy_matrix_np = np.array(accuracy_matrix)
    final_acc = accuracy_matrix_np[-1].mean()
    backward_transfer = (accuracy_matrix_np[-1] - accuracy_matrix_np.diagonal()).mean()
    forward_transfer = (accuracy_matrix_np[0] - accuracy_matrix_np.diagonal()).mean()

    # Log results
    with open(RESULTS_FILE, "w") as f:
        f.write("Accuracy Matrix:\n")
        f.write(str(accuracy_matrix_np))
        f.write("\nFinal Accuracy: {:.4f}\n".format(final_acc))
        f.write("Backward Transfer: {:.4f}\n".format(backward_transfer))
        f.write("Forward Transfer: {:.4f}\n".format(forward_transfer))

if __name__ == "__main__":
    main()
```