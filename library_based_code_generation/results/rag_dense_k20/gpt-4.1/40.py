```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from nni.compression.pytorch.pruning import TaylorPruner, AGPPruner
from nni.compression.pytorch.quantization import QATQuantizer, Quantizer
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.distillation import DynamicLayerwiseDistiller
from nni.compression.pytorch.utils import count_flops_params

# 1. Fine-tune ResNet18 on CIFAR10 for 30 epochs and evaluate accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

model = torchvision.models.resnet18(num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

for epoch in range(30):
    model.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

acc = evaluate(model)
print(f'Fine-tuned ResNet18 accuracy: {acc:.2f}%')

# 2. Create teacher model by duplicating the fine-tuned model
import copy
teacher_model = copy.deepcopy(model)

# 3. Create pruner using TaylorPruner and AGPPruner
pruner_config = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
taylor_pruner = TaylorPruner(model, pruner_config, optimizer=optimizer, trainer=None, num_iterations=100, pruning_algorithm='TaylorFOWeight')
agp_pruner = AGPPruner(model, pruner_config, optimizer=optimizer, trainer=None, num_iterations=100, total_times=30)

# For demonstration, use TaylorPruner (could also use AGPPruner similarly)
pruner = taylor_pruner
pruner.compress()

# 4. Create quantizer using QATQuantizer
qat_config = [{
    'quant_types': ['weight', 'input'],
    'quant_bits': {'weight': 8, 'input': 8},
    'op_types': ['Conv2d', 'BatchNorm2d'],
    'start_step': 100
}]
qat_quantizer = QATQuantizer(model, qat_config, optimizer=optimizer)
qat_quantizer.compress()

# 5. Create distiller using DynamicLayerwiseDistiller
distill_config = [{
    'op_types': ['Conv2d'],
    'distillation': 'mse'
}]
distiller = DynamicLayerwiseDistiller(model, teacher_model, distill_config)

# 6. Compress the model using the distiller for 60 iterations of 100 steps each
for iteration in range(60):
    distiller.step(steps=100)

# 7. Speed up the model using ModelSpeedup and the masks from the pruner
dummy_input = torch.randn(1, 3, 32, 32).to(device)
masks = pruner.get_masks()
ModelSpeedup(model, dummy_input, masks).speedup_model()

# 8. Evaluate the accuracy of the compressed model
acc_compressed = evaluate(model)
print(f'Compressed model accuracy: {acc_compressed:.2f}%')

# 9. Simulate quantization by updating the calibration configuration using Quantizer
quant_config = [{
    'quant_types': ['weight', 'input'],
    'quant_bits': {'weight': 8, 'input': 8},
    'op_types': ['Conv2d', 'BatchNorm2d']
}]
quantizer = Quantizer(model, quant_config)
quantizer.compress()

# 10. Evaluate the accuracy of the compressed and quantized model
acc_quantized = evaluate(model)
print(f'Compressed & quantized model accuracy: {acc_quantized:.2f}%')
```