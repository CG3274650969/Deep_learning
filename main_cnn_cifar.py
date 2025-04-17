import torch 
import torch.optim as optim
from torch.utils.data import DataLoader 
from torchvision import datasets
from models.cnn_cifar import CIFAR100CNN
from utils.train_eval import train_model, evaluate_model
from utils.analysis import analyze_model
from data.transforms import cifar_transform
from torchsummary import summary
from torchviz import make_dot
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = CIFAR100CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 测试模型
evaluate_model(model, test_loader, criterion)

# 模型结构摘要
print("\n模型结构摘要：")
summary(model, (3, 32, 32))

# 绘制网络结构图
dummy_input = torch.randn(1, 3, 32, 32).to(device)
output = model(dummy_input)
make_dot(output, params=dict(model.named_parameters())).render("cifar100_cnn", format="png")
print("\n网络结构图已生成：cifar100_cnn.png")

# 模型 FLOPs 和 参数量分析
print("\n模型 FLOPs / Params 分析：")
analyze_model(model, input_size=(1, 3, 32, 32))
