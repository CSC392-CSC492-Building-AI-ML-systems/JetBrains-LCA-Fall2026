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

# 1. Fine-tune ResNet18 on CIFAR10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

def train(model, trainloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

train(model, trainloader, criterion, optimizer, device, epochs=30)
acc = evaluate(model, testloader, device)
print(f'Fine-tuned model accuracy: {acc:.2f}%')

# 2. Create teacher model
teacher_model = copy.deepcopy(model)

# 3. Create pruner (TaylorPruner and AGPPruner)
prune_config = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = TaylorPruner(model, prune_config, optimizer=optimizer, trainer=None, num_iterations=100, pruning_algorithm='TaylorFOWeight')
agp_pruner = AGPPruner(model, prune_config, optimizer=optimizer, trainer=None, num_iterations=100, total_times=30)

# 4. Create quantizer (QATQuantizer)
quant_config = [{
    'quant_types': ['weight', 'input'],
    'quant_bits': {'weight': 8, 'input': 8},
    'op_types': ['Conv2d', 'BatchNorm2d']
}]
qat_quantizer = QATQuantizer(model, quant_config, optimizer=optimizer, start_step=100)

# 5. Create distiller (DynamicLayerwiseDistiller)
distill_config = [{
    'op_types': ['Conv2d'],
    'distillation_type': 'mse'
}]
distiller = DynamicLayerwiseDistiller(model, teacher_model, distill_config, criterion=criterion, optimizer=optimizer)

# 6. Compress model using distiller
def dummy_train_func(model, optimizer, criterion, trainloader, device, steps):
    model.train()
    step = 0
    while step < steps:
        for inputs, targets in trainloader:
            if step >= steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            step += 1

for i in range(60):
    dummy_train_func(distiller, optimizer, criterion, trainloader, device, 100)

# 7. Speed up model using ModelSpeedup and pruner masks
dummy_input = torch.randn(1, 3, 224, 224).to(device)
pruner.compress()
masks = pruner.get_masks()
ModelSpeedup(model, dummy_input, masks).speedup_model()

# 8. Evaluate accuracy of compressed model
acc_compressed = evaluate(model, testloader, device)
print(f'Compressed model accuracy: {acc_compressed:.2f}%')

# 9. Simulate quantization by updating calibration config using Quantizer
quantizer = Quantizer(model, quant_config)
quantizer.compress()

# 10. Evaluate accuracy of compressed and quantized model
acc_quantized = evaluate(model, testloader, device)
print(f'Compressed & Quantized model accuracy: {acc_quantized:.2f}%')
```