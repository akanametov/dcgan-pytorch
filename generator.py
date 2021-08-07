import torch
from torch import nn

class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dimension=100, hid_channels=64, img_channels=3):
        super().__init__()
        """Initialize model."""
        self.net = nn.Sequential(
            # input(Z) size: (z_dimension) x 1 x 1
            nn.ConvTranspose2d(z_dimension, hid_channels*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hid_channels*8),
            nn.ReLU(True),
            # output size: (hid_channels*8) x 4 x 4
            nn.ConvTranspose2d(hid_channels*8, hid_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels*4),
            nn.ReLU(True),
            # output size: (hid_channels*4) x 8 x 8
            nn.ConvTranspose2d(hid_channels*4, hid_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels*2),
            nn.ReLU(True),
            # output size: (hid_channels*2) x 16 x 16
            nn.ConvTranspose2d(hid_channels*2, hid_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            # output size: (hid_channels) x 32 x 32
            nn.ConvTranspose2d(hid_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output size: (img_channels) x 64 x 64
        )

    def forward(self, x):
        fx = self.net(x)
        return fx