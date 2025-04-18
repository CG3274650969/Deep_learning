import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from models.cnn_mnist import SimpleCNN
from utils.train_eval import train_model, evaluate_model
from utils.analysis import analyze_model
from data.transforms import mnist_transform
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建结果保存目录
os.makedirs("result", exist_ok=True)

# 数据加载
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 尝试多组CNN结构
kernel_sizes = [3, 5]
hidden_dims = [64, 128]

for k in kernel_sizes:
    for h in hidden_dims:
        print(f"\n--- 正在训练 CNN(kernel_size={k}, hidden_dim={h}) ---")

        # 初始化模型
        model = SimpleCNN(kernel_size=k, hidden_dim=h)

        # 设置损失函数与优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        train_model(model, train_loader, criterion, optimizer, num_epochs=10)

        # 测试模型
        evaluate_model(model, test_loader, criterion)

        # 模型复杂度分析 + 网络结构图保存
        analyze_model(model, input_size=(1, 1, 28, 28), name=f"cnn_k{k}_h{h}")

