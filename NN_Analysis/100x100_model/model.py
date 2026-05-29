#!/usr/bin/env python3
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, k=3, s=1, p=1, norm="group", groups=8):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)]

    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == "group":
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        layers.append(nn.GroupNorm(g, out_ch))
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {norm}")

    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class EnergyRegressionCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.1, norm: str = "group"):
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(in_channels, 32,  k=3, s=1, p=1, norm=norm),
            conv_block(32,          64,  k=3, s=2, p=1, norm=norm),
            conv_block(64,          128, k=3, s=2, p=1, norm=norm),
            conv_block(128,         128, k=3, s=1, p=1, norm=norm),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.gap(f).flatten(1)      # (N,128)
        out = self.head(f).squeeze(-1)  # (N,)
        return out