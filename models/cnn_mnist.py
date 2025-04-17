import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=3, hidden_dim=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # 用固定大小的 AvgPool2d 替代 AdaptiveAvgPool2d
        self.global_pool = nn.AvgPool2d(kernel_size=7)  # 28x28经过两次池化变为7x7

        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.global_pool(x)               # 7x7 -> 1x1
        x = x.view(x.size(0), -1)             # Flatten to (batch_size, hidden_dim)
        x = self.fc(x)
        return x

