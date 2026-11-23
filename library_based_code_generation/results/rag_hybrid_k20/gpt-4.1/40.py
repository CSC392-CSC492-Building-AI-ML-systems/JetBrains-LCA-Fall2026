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

# 1. Fine-tune ResNet18 on CIFAR10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(30):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

acc = evaluate(model)
print(f'Fine-tuned model accuracy: {acc:.2f}%')

# 2. Create teacher model by duplicating the fine-tuned model
import copy
teacher_model = copy.deepcopy(model)

# 3. Create pruner (TaylorPruner and AGPPruner)
prune_config = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = TaylorPruner(model, prune_config, optimizer=optimizer, trainer=None, num_iterations=100, pruning_algorithm='TaylorFOWeight')
agp_pruner = AGPPruner(model, prune_config, optimizer=optimizer, trainer=None, total_iteration=30, steps_per_epoch=100)

# Run pruner for 100 steps and 30 total times
def dummy_trainer(model, optimizer, criterion, trainloader, steps):
    model.train()
    step = 0
    for epoch in range(1000):
        for inputs, labels in trainloader:
            if step >= steps:
                return
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1

pruner._trainer = lambda model, optimizer, criterion, dataloader: dummy_trainer(model, optimizer, criterion, trainloader, 100)
pruner.compress()
agp_pruner._trainer = lambda model, optimizer, criterion, dataloader: dummy_trainer(model, optimizer, criterion, trainloader, 100)
agp_pruner.compress()

# 4. Create quantizer (QATQuantizer)
quant_config = [{
    'quant_types': ['weight', 'input'],
    'quant_bits': {'weight': 8, 'input': 8},
    'op_types': ['Conv2d', 'BatchNorm2d']
}]
qat_quantizer = QATQuantizer(model, quant_config, optimizer=optimizer, trainer=None, start_step=100)
qat_quantizer._trainer = lambda model, optimizer, criterion, dataloader: dummy_trainer(model, optimizer, criterion, trainloader, 100)
qat_quantizer.compress()

# 5. Create distiller (DynamicLayerwiseDistiller)
distill_config = [{
    'op_types': ['Conv2d'],
    'distillation': 'mse'
}]
distiller = DynamicLayerwiseDistiller(model, teacher_model, distill_config, optimizer=optimizer, criterion=criterion, train_dataloader=trainloader, val_dataloader=testloader)

# 6. Compress model using distiller for 60 iterations of 100 steps each
for i in range(60):
    distiller.train(epoch=1, steps=100)

# 7. Speed up model using ModelSpeedup and pruner masks
dummy_input = torch.randn(1, 3, 32, 32).to(device)
masks = pruner.get_masks()
msu = ModelSpeedup(model, dummy_input, masks)
msu.speedup_model()

# 8. Evaluate accuracy of compressed model
acc_compressed = evaluate(model)
print(f'Compressed model accuracy: {acc_compressed:.2f}%')

# 9. Simulate quantization by updating calibration config using Quantizer
quantizer = Quantizer(model, quant_config)
quantizer.compress()

# 10. Evaluate accuracy of compressed and quantized model
acc_quantized = evaluate(model)
print(f'Compressed & quantized model accuracy: {acc_quantized:.2f}%')
```