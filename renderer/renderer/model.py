"""Stroke neural network model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeFCN(nn.Module):
    def __init__(self) -> None:
        super(StrokeFCN, self).__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=2048)
        self.fc4 = nn.Linear(in_features=2048, out_features=4096)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        view = x.view(-1, 128, 128)
        return 1 - view
