import torch
from torch import nn

class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_channels=3, hid_channels=64, out_channels=1):
        super().__init__()
        """Initialize model."""
        self.net = nn.Sequential(
            # input(image) size: (img_channels) x 64 x 64
            nn.Conv2d(img_channels, hid_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # output size: (hid_channels) x 32 x 32
            nn.Conv2d(hid_channels, hid_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            # output size: (hid_channels*2) x 16 x 16
            nn.Conv2d(hid_channels*2, hid_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            # output size: (hid_channels*4) x 8 x 8
            nn.Conv2d(hid_channels*4, hid_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
            # output size: (hid_channels*8) x 4 x 4
            nn.Conv2d(hid_channels*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fx = self.net(x)
        return fx