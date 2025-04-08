import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from models.cnn_cifar import CIFAR100CNN
from utils.train_eval import train_model, evaluate_model
from utils.analysis import analyze_model
from data.transforms import cifar_transform

# 数据加载
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型、损失、优化器
model = CIFAR100CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 测试模型
evaluate_model(model, test_loader, criterion)

# 模型分析
analyze_model(model, input_size=(1, 3, 32, 32))
