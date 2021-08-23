import torch.nn as nn
import torch.nn.functional as F


class Capsule(nn.Module):
    def __init__(self):
        super(Capsule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=9, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x
