import torch
import torch.nn as nn

class AHINet(nn.Module):
    def __init__(self):
        super(AHINet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=5//2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=3//2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, padding=1//2)
        self.fc1 = nn.Linear(8 * 1024, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x