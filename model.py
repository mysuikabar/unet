import torch
import torch.nn as nn

from model_parts import DoubleConv, DownConv, UpConv


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = DoubleConv(in_channels=3, out_channels=64)
        self.down1 = DownConv(in_channels=64, out_channels=128)
        self.down2 = DownConv(in_channels=128, out_channels=256)
        self.down3 = DownConv(in_channels=256, out_channels=512)
        self.up3 = UpConv(in_channels=512, out_channels=256)
        self.up2 = UpConv(in_channels=256, out_channels=128)
        self.up1 = UpConv(in_channels=128, out_channels=64)
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)  # (3, 256, 256) -> (64, 256, 256)
        x2 = self.down1(x1)  # (64, 256, 256) -> (128, 128, 128)
        x3 = self.down2(x2)  # (128, 128, 128) -> (256, 64, 64)
        x = self.down3(x3)  # (256, 64, 64) -> (512, 32, 32)
        x = self.up3(x, x3)  # (512, 32, 32) -> (256, 64, 64)
        x = self.up2(x, x2)  # (256, 64, 64) -> (128, 128, 128)
        x = self.up1(x, x1)  # (128, 128, 128) -> (64, 256, 256)
        out = self.out(x)  # (64, 256, 256) -> (1, 256, 256)
        return torch.sigmoid(out)
