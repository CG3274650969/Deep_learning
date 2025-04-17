import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_layers=1, hidden_dim=512, num_classes=10):
        super().__init__()

        layers = [nn.Flatten(), nn.Linear(input_size, hidden_dim), nn.ReLU()]
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
