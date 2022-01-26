import torch
import torch.nn as nn
import torch.nn.functional as F


class MedCNN(nn.Module):
    def __init__(self, backbone, n_class):
        super(MedCNN, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Linear(2048, 256)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(256, 2)
        self.n_class = n_class

    def forward(self, x):
        x = self.backbone(x)
        x = self.drop(x)
        x = F.relu(self.fc(x))
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pool_kernel, pool_stride):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.pool(self.conv(x))))
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(Conv(3, 16, 3, 2, 2),
                                 Conv(16, 64, 3, 2, 2),
                                 Conv(64, 128, 3, 2, 2))
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean((2, 3))
        x = self.fc(x)
        x = F.softmax(x, 1)
        return x
