import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(in_features=7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=2048)
        self.fc4 = nn.Linear(in_features=2048, out_features=4096)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=1
        )  # Boosted out_channels from 4 to 12. WTF and I doing?
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        # print("start", x.shape)
        x = F.relu(self.fc1(x))
        # print("fc1", x.shape)
        x = F.relu(self.fc2(x))
        # print("fc2", x.shape)
        x = F.relu(self.fc3(x))
        # print("fc3", x.shape)
        x = F.relu(self.fc4(x))
        # print("fc4", x.shape)
        x = x.view(-1, 16, 16, 16)
        # print("fc4-view", x.shape)
        x = F.relu(self.conv1(x))
        # print("conv1", x.shape)
        x = self.pixel_shuffle(self.conv2(x))
        # print("conv2", x.shape)
        x = F.relu(self.conv3(x))
        # print("conv3", x.shape)
        x = self.pixel_shuffle(self.conv4(x))
        # print("conv4", x.shape)
        x = F.relu(self.conv5(x))
        # print("conv5", x.shape)
        x = self.pixel_shuffle(self.conv6(x))
        # print("conv6", x.shape)
        x = torch.sigmoid(x)
        # print("sigmoid", x.shape)
        view = x.view(-1, 128, 128, 3)  # Reshaped to get same shape as image. Is this bad? - 3 first
        # print("end", x.shape)
        return 1 - view
