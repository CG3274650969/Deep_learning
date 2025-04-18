import torch.nn as nn
import torch.nn.functional as F

# 1. 基础 CNN
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        return self.net(x)

# 2. ResNet-style CNN（使用残差连接）
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNetStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(3, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        return self.net(x)

# 3. MobileNet-style CNN（深度可分离卷积）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = F.relu(self.depthwise(x))
        x = F.relu(self.pointwise(x))
        return x

class MobileNetStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(3, 64),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        return self.net(x)
