import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from models.mlp import MLP
from utils.train_eval import train_model, evaluate_model
from data.transforms import mnist_transform
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 多种隐藏层设置下测试性能
accuracies = []
hidden_layer_options = [1, 2, 3, 4, 5]

for hidden_layers in hidden_layer_options:
    model = MLP(input_size=28*28, hidden_layers=hidden_layers, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=5)
    acc = evaluate_model(model, test_loader, criterion)
    accuracies.append(acc)
    print(f"MNIST: 隐藏层 {hidden_layers}, 精度: {acc:.4f}")

# 找到达到85%以上精度的最小隐藏层数
for i, acc in enumerate(accuracies):
    if acc >= 0.85:
        best_hidden = hidden_layer_options[i]
        break
else:
    best_hidden = hidden_layer_options[-1]

# 输出最佳模型信息
print(f"\n达到 85% 精度的最小隐藏层大小: {best_hidden}")

# 构造最佳模型并分析
model_best = MLP(input_size=28*28, hidden_layers=best_hidden, num_classes=10).to(device)
dummy_input = torch.randn(2, 1, 28, 28).to(device)  # batch size = 2

# 将输入 reshape 为 (batch_size, 784) 因为 MLP 接收的是 flatten 输入
dummy_input_flat = dummy_input.view(2, -1)

# 分析 FLOPs 和 参数量
flops, params = profile(model_best, inputs=(dummy_input_flat,), verbose=False)

# 重新训练最佳模型获取准确率
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_best.parameters(), lr=0.001)
train_model(model_best, train_loader, criterion, optimizer, num_epochs=5)
acc = evaluate_model(model_best, test_loader, criterion)

# 输出分析结果
print(f"\n=== 最佳模型 (隐藏层大小: {best_hidden}) ===")
print(f"准确率: {acc:.4f}")
print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")
print(f"参数量: {params / 1e6:.2f} M参数")
