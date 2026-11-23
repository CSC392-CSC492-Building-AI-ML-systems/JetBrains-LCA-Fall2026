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
import copy

# 1. Load CIFAR10 and ResNet18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

def get_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    return model

model = get_resnet18().to(device)

# 1. Fine-tune ResNet18
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(model, loader, optimizer, criterion, device):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

for epoch in range(30):
    train(model, trainloader, optimizer, criterion, device)
    scheduler.step()

acc = test(model, testloader, device)
print(f'Fine-tuned model accuracy: {acc:.2f}%')

# 2. Create teacher model
teacher_model = copy.deepcopy(model)

# 3. Create pruners
prune_config = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d'],
}]
taylor_pruner = TaylorPruner(model, prune_config, trainer=lambda m, i: train(m, trainloader, optimizer, criterion, device), 
                             num_iterations=100, pruning_algorithm='TaylorFOWeight')
agp_pruner = AGPPruner(model, prune_config, trainer=lambda m, i: train(m, trainloader, optimizer, criterion, device), 
                       total_times=30, steps=100)

# Run TaylorPruner for 100 steps
taylor_pruner.compress()
# Run AGPPruner for 30 times, each 100 steps
agp_pruner.compress()

# 4. Create quantizer
quant_config = [
    {'op_types': ['Conv2d', 'BatchNorm2d'], 'quant_types': ['weight', 'input'], 'bit': 8}
]
qat_quantizer = QATQuantizer(model, quant_config, optimizer, start_step=100)
qat_quantizer.compress()

# 5. Create distiller
distill_config = [{
    'op_types': ['Conv2d'],
    'distillation': 'mse'
}]
distiller = DynamicLayerwiseDistiller(model, teacher_model, distill_config, criterion, optimizer)

# 6. Compress with distiller for 60 iterations of 100 steps each
for i in range(60):
    distiller.step(trainloader, steps=100)

# 7. Speed up model
dummy_input = torch.randn(1, 3, 32, 32).to(device)
masks = agp_pruner.get_masks()
ModelSpeedup(model, dummy_input, masks).speedup_model()

# 8. Evaluate compressed model
acc_compressed = test(model, testloader, device)
print(f'Compressed model accuracy: {acc_compressed:.2f}%')

# 9. Simulate quantization
quantizer = Quantizer(model, quant_config)
quantizer.compress()
quantizer.calibrate(trainloader, device=device)

# 10. Evaluate compressed and quantized model
acc_quantized = test(model, testloader, device)
print(f'Compressed & Quantized model accuracy: {acc_quantized:.2f}%')
```