import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.depthwise = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 
                                   kernel_size=kernel_size, stride=stride, padding=kernel_size//2, 
                                   groups=in_channels * expand_ratio
                                  )
        self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.se = SqueezeExcitation(in_channels * expand_ratio, in_channels * expand_ratio // 4)
        self.project = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.expand(x)))
        x = F.relu(self.bn2(self.depthwise(x)))
        x = self.se(x)
        x = self.bn3(self.project(x))
        if self.shortcut:
            x += identity
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.se(x)
        return x * y.expand_as(x)

class MyNetwork_rev(nn.Module):
    def __init__(self, num_classes=200, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            MBConv(3, 32, 3, 1, expand_ratio=1),
            MBConv(32, 64, 3, 2, expand_ratio=6),
            MBConv(64, 128, 5, 2, expand_ratio=6),
            MBConv(128, 256, 3, 2, expand_ratio=6),
            MBConv(256, 256, 3, 1, expand_ratio=6),
            MBConv(256, 128, 3, 2, expand_ratio=6),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

