import torch
import torch.nn as nn
import torch.nn.functional as F


class Capsule(nn.Module):
    def __init__(self):
        super(Capsule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=9, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, caps_dims=8):
        """
        caps_dim: dimension of the capsule vector
        """

        super(PrimaryCaps, self).__init__()

        caps = []
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.caps_dims = caps_dims

        for _ in range(out_channels):
            caps.append(nn.Conv2d(in_channels=in_channels, out_channels=caps_dims, kernel_size=kernel_size, stride=stride))

    def forward(self, x):
        outputs = []
        for i in range(self.out_channels):
            # outputs.append(

        outputs = torch.zeros(self.out_channels*(self.kernel_size**2), self.caps_dims)


class DigitCaps(nn.Module):
    def __init__(self):
        super(DigitCaps, self).__init__()


class CapsNet(nn.Module):
    def __init__(self, num_classes=10, primarycaps_dim=8, digitcaps_dim=16):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCaps(256, 32, 6, stride=2, caps_dim=primarycaps_dim)
        self.digit_caps = DigitCaps(num_classes, 8, 16, kernel_size=9, stride=2)
