import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ResNet50 import *
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50
from torch.optim.lr_scheduler import MultiStepLR



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ResNet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)




# 设置超参数
num_epochs = 180
batch_size = 128
learning_rate = 0.1

# 数据预处理
transform = transforms.Compose([
    transforms.RandomRotation(20,resample=False,expand=False,center=None),
    transforms.Pad((4,4),fill=(0,0,0),padding_mode="constant"),
    transforms.RandomCrop(32,padding=4,fill=0,padding_mode="constant"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
print("正在加载数据集")
train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("数据集加载成功")

# 创建实例
model = ResNet50().to(device)

# 定义损失函数和优化器0
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay = 1e-4)
criterion = nn.CrossEntropyLoss()

scheduler = MultiStepLR(optimizer, milestones=[32, 48], gamma=0.1)

# 训练模型

for epoch in range(num_epochs):
    model.train()
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct = 0
        total = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            out = model(img)
            _, prediction = torch.max(out.data, 1)
            total = total + label.size(0)
            correct += (prediction == label).sum().item()

    print("Epoch：", epoch, " train 正确率为：" + str(correct / total))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            out = model(img)
            _, prediction = torch.max(out.data, 1)
            total = total + label.size(0)
            correct += (prediction == label).sum().item()

    print("Epoch：", epoch, " test 正确率为：" + str(correct / total))
