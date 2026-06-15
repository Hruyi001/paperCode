import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels, max_groups=32):
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            _group_norm(out_channels),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class MSFA(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.project = ConvBNAct(channels, hidden, kernel_size=1)
        self.local = ConvBNAct(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.strip = nn.Sequential(
            ConvBNAct(hidden, hidden, kernel_size=(1, 5), padding=(0, 2), groups=hidden),
            ConvBNAct(hidden, hidden, kernel_size=(5, 1), padding=(2, 0), groups=hidden),
        )
        self.dilated = ConvBNAct(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, kernel_size=7, padding=3, bias=False),
            _group_norm(hidden),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 3, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 3, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.out = ConvBNAct(hidden * 3, channels, kernel_size=1, act=False)

    def forward(self, x):
        x_proj = self.project(x)
        multi = torch.cat([self.local(x_proj), self.strip(x_proj), self.dilated(x_proj)], dim=1)
        multi = multi * self.channel(multi)
        chunks = torch.chunk(multi, 3, dim=1)
        spatial = self.spatial(multi)
        fused = torch.cat([chunk * spatial for chunk in chunks], dim=1)
        return self.out(fused) + x


class MGSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.rgb_proj = ConvBNAct(channels, hidden, kernel_size=1)
        self.normal_proj = ConvBNAct(channels, hidden, kernel_size=1)
        self.query = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.value = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.restore = ConvBNAct(hidden, channels, kernel_size=1, act=False)
        self.msfa = MSFA(channels, reduction=reduction)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_features, normal_features=None):
        if normal_features is None:
            normal_features = rgb_features
        rgb = self.rgb_proj(rgb_features)
        normal = self.normal_proj(normal_features)
        q = self.query(rgb)
        k = self.key(normal)
        v = self.value(normal)
        attention = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / math.sqrt(max(q.size(1), 1)))
        geometry = v * attention
        gate = self.gate(torch.cat([rgb, geometry], dim=1))
        fused = rgb + gate * geometry
        enhancement = self.msfa(self.restore(fused) + rgb_features) - rgb_features
        return rgb_features + self.residual_scale * enhancement


class SSPM(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        split = max(channels // groups, 1)
        kernels = [3, 5, 7, 9]
        self.h_convs = nn.ModuleList([
            nn.Conv1d(split, split, kernel_size=k, padding=k // 2, groups=split, bias=False) for k in kernels
        ])
        self.w_convs = nn.ModuleList([
            nn.Conv1d(split, split, kernel_size=k, padding=k // 2, groups=split, bias=False) for k in kernels
        ])
        norm_groups = min(groups, channels)
        while channels % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        self.h_fuse = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(norm_groups, channels), nn.Sigmoid())
        self.w_fuse = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(norm_groups, channels), nn.Sigmoid())

    def _apply_axis(self, x, convs):
        chunks = torch.chunk(x, len(convs), dim=1)
        outputs = []
        for chunk, conv in zip(chunks, convs):
            if chunk.size(1) == conv.in_channels:
                outputs.append(conv(chunk))
            else:
                outputs.append(chunk)
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        h_context = x.mean(dim=3)
        w_context = x.mean(dim=2)
        h_attn = self.h_fuse(self._apply_axis(h_context, self.h_convs)).unsqueeze(3)
        w_attn = self.w_fuse(self._apply_axis(w_context, self.w_convs)).unsqueeze(2)
        return x * h_attn * w_attn


class SFEM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.q = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.k = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.v = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)
        tokens = pooled.flatten(2).transpose(1, 2)
        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)
        attn = torch.softmax(torch.matmul(q.transpose(1, 2), k), dim=-1)
        weights = torch.matmul(v, attn).transpose(1, 2).view_as(pooled)
        weights = self.channel(weights)
        return x * weights + x


class SEM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sspm = SSPM(channels)
        self.sfem = SFEM(channels)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        enhancement = self.sfem(self.sspm(x)) - x
        return x + self.residual_scale * enhancement


class FrequencyFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
        )
        self.high = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.mix = ConvBNAct(channels * 2, channels, kernel_size=1, act=False)

    def forward(self, x):
        low = self.low(x)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high = self.high(x - low)
        return self.mix(torch.cat([low, high], dim=1)) + x


class ABS(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.bn = _group_norm(channels)
        self.spa_feat = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.spa_gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.pa = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = ConvBNAct(channels * 3, channels, kernel_size=1, act=False)

    def forward(self, x):
        base = self.bn(x)
        spatial = self.spa_feat(base) * self.spa_gate(base)
        pixel = base * self.pa(base)
        channel = base * self.ca(base)
        return x + self.out(torch.cat([spatial, pixel, channel], dim=1))


class FSM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.frequency = FrequencyFilter(channels)
        self.abs = ABS(channels)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        enhancement = self.abs(self.frequency(x)) - x
        return x + self.residual_scale * enhancement
