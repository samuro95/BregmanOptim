import numpy as np

import torch
import torch.nn as nn

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1, device='cpu'):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels, device=device) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels, device=device))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


class NoiseEmbedding(torch.nn.Module):
    def __init__(self, num_channels=1, emb_channels=512, device='cpu'):
        super().__init__()
        self.emb_fourier = MPFourier(num_channels, device=device)
        self.emb_noise = MPConv(num_channels, emb_channels, kernel=[])

    def forward(self, sigma):
        emb = self.emb_noise(self.emb_fourier(sigma))
        emb = mp_silu(emb)
        return emb

class CondResBlock(nn.Module):
    def __init__(
            self,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            emb_channels=512,
    ):
        super(CondResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."

        self.gain = torch.nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x, emb_sigma):
        u = self.conv1(mp_silu(x))
        c = self.emb_linear(emb_sigma, gain=self.gain) + 1
        y = mp_silu(u * c.unsqueeze(2).unsqueeze(3).to(u.dtype))
        y = self.conv2(y)
        return x + y