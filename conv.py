import torch
from torch import nn
from torch.nn import functional as F

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                             nn.BatchNorm1d(out_channels)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv1dTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                            nn.BatchNorm1d(out_channels)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

