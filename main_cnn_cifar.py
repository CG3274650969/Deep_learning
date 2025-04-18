import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn_cifar import BasicCNN, ResNetStyleCNN, MobileNetStyleCNN
from torchsummary import summary
from torchviz import make_dot
import os

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_model(model):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def visualize_model(model, name):
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    out = model(dummy_input)
    make_dot(out, params=dict(model.named_parameters())).render(name, format="png")
    print(f"结构图已保存为：{name}.png")

if __name__ == '__main__':
    for ModelClass, name in zip([BasicCNN, ResNetStyleCNN, MobileNetStyleCNN], ["basic", "resnet", "mobilenet"]):
        print(f"\n=== 正在训练 {name} 模型 ===")
        model = ModelClass()
        train_model(model)
        evaluate_model(model)
        summary(model.to(device), (3, 32, 32))
        # visualize_model(model, f"cnn_{name}")
