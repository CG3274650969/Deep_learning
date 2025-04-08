import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from models.mlp import MLP
from utils.train_eval import train_model, evaluate_model
from utils.analysis import analyze_model
from data.transforms import mnist_transform

# 数据加载
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型、损失、优化器
model = MLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 测试模型
evaluate_model(model, test_loader, criterion)

# 模型分析
analyze_model(model, input_size=(1, 1, 28, 28))
