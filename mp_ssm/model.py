"""Main-text MP-SSM architecture. See docs/paper_mapping.md for choices."""

import torch
from torch import nn
from torch.nn import functional as F

from .scan import SelectiveScan2D


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def conv_bn(in_channels, out_channels, kernel=3, stride=1, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
              nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class LocalSpatialBlock(nn.Module):
    """Paper Eqs. 10-11, with no hidden channel expansion."""
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(), nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.branch(x)


class ResidualGatedBlock(nn.Module):
    """The gated increment in Eqs. 12-13; caller owns the residual."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels, channels, 1)
        self.value = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                   nn.Conv2d(channels, channels, 7, padding=3, groups=channels), nn.SiLU())

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.value(x)


class VSSBlock(nn.Module):
    def __init__(self, channels, d_state, expand, scan_backend):
        super().__init__()
        self.local = LocalSpatialBlock(channels)
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.scan = SelectiveScan2D(channels, d_state, expand, scan_backend)
        self.gate = ResidualGatedBlock(channels)

    def forward(self, x):
        x = self.local(x)
        x = x + self.scan(self.norm1(x))
        return x + self.gate(self.norm2(x))


class SPPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.project = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):
        features = [x]
        for _ in range(3):
            features.append(self.pool(features[-1]))
        return self.project(torch.cat(features, dim=1))


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, width, d_state, expand, scan_backend):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_channels])
        self.norms = nn.ModuleList([LayerNorm2d(width) for _ in range(3)])
        self.attention = nn.Sequential(nn.Conv2d(width, width, 1),
                                       nn.Conv2d(width, width, 3, padding=1, groups=width), nn.Sigmoid())
        def branch():
            return nn.Sequential(nn.Conv2d(width, width, 1),
                                 nn.Conv2d(width, width, 3, padding=1, groups=width), nn.SiLU(),
                                 SelectiveScan2D(width, d_state, expand, scan_backend), LayerNorm2d(width))
        self.small, self.large = branch(), branch()
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, features):
        # small/medium/large refer to spatial grids: 1/16, 1/8, 1/4.
        target = features[-1].shape[-2:]
        aligned = [F.interpolate(p(x), size=target, mode="bilinear", align_corners=False)
                   for p, x in zip(self.projections, features)]
        small, medium, large = [norm(x) for norm, x in zip(self.norms, aligned)]
        attention = self.attention(small + medium + large)
        return self.output(attention * (self.small(small) + self.large(large))) + aligned[0] + aligned[2]


class MPSSM(nn.Module):
    def __init__(self, channels=(16, 32, 48, 64), fusion_channels=32, d_state=16,
                 expand=2, scan_backend="cuda"):
        super().__init__()
        if len(channels) != 4 or any(c < 2 for c in channels):
            raise ValueError("channels must contain four widths >= 2")
        settings = (d_state, expand, scan_backend)
        self.stem = nn.Sequential(conv_bn(3, channels[0] // 2, stride=2),
                                  conv_bn(channels[0] // 2, channels[0], stride=2, activation=False))
        self.encoder = nn.ModuleList([VSSBlock(c, *settings) for c in channels])
        self.downsample = nn.ModuleList([conv_bn(channels[i], channels[i + 1], stride=2) for i in range(3)])
        self.sppf = SPPF(channels[-1])
        self.up_project = nn.ModuleList([nn.Conv2d(channels[i + 1], channels[i], 1) for i in (2, 1, 0)])
        self.decoder = nn.ModuleList([VSSBlock(channels[i], *settings) for i in (2, 1, 0)])
        self.fusion = MultiScaleFusion([channels[i] for i in (2, 1, 0)], fusion_channels, *settings)
        self.head = nn.Conv2d(fusion_channels, 1, 1)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("Expected nonempty [B,3,H,W] RGB images")
        height, width = images.shape[-2:]
        padded = F.pad(images, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        x = self.stem(padded)
        features = []
        for i, block in enumerate(self.encoder):
            if i:
                x = self.downsample[i - 1](x)
            x = block(x)
            features.append(x)
        x = self.sppf(x)
        decoded = []
        for project, block, skip in zip(self.up_project, self.decoder, reversed(features[:-1])):
            x = F.interpolate(project(x), size=skip.shape[-2:], mode="bilinear", align_corners=False) + skip
            x = block(x)
            decoded.append(x)
        logits = self.head(self.fusion(decoded))
        logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :height, :width]
