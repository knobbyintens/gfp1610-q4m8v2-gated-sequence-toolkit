"""Layer zoo of the gated sequence toolkit."""

import torch
import torch.nn as nn


class SGRKBlock(nn.Module):
    """SGRK-style recurrent kernel block (one of several supported layers)."""

    def __init__(self, channels=64, bank=8):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, h):
        return torch.relu(self.conv(h))


class GatedConvBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.gate = nn.Conv1d(channels, channels, 1)

    def forward(self, h):
        return self.conv(h) * torch.sigmoid(self.gate(h))
