"""Reference SNN models in plain PyTorch.

These deliberately do the naive thing CLAUDE.md's thesis says is slow: a
Python for-loop over T, each step round-tripping membrane potential through
ordinary elementwise ops and launching a full kernel per op. This is the
baseline Workstream A profiles against SpikingJelly's fused multi-step
kernels and against a future PyMantle persistent kernel.
"""

import torch
import torch.nn as nn


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward."""

    scale = 10.0

    @staticmethod
    def forward(ctx, v_minus_threshold):
        ctx.save_for_backward(v_minus_threshold)
        return (v_minus_threshold > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v_minus_threshold,) = ctx.saved_tensors
        sg = 1.0 / (SurrGradSpike.scale * v_minus_threshold.abs() + 1.0) ** 2
        return grad_output * sg


spike_fn = SurrGradSpike.apply


class LIFCell(nn.Module):
    def __init__(self, decay=0.9, threshold=1.0):
        super().__init__()
        self.decay = decay
        self.threshold = threshold

    def forward(self, x, v):
        v = self.decay * v + x
        spike = spike_fn(v - self.threshold)
        v = v * (1.0 - spike)  # hard reset
        return spike, v


class MLPSNN(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.lif1 = LIFCell()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.lif2 = LIFCell()
        self.hidden = hidden
        self.out_dim = out_dim

    def forward(self, x):
        # x: [T, B, in_dim]
        T, B = x.shape[0], x.shape[1]
        v1 = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
        v2 = torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)
        out_spikes = []
        for t in range(T):
            s1, v1 = self.lif1(self.fc1(x[t]), v1)
            s2, v2 = self.lif2(self.fc2(s1), v2)
            out_spikes.append(s2)
        return torch.stack(out_spikes, dim=0)  # [T, B, out_dim]


class ConvSNN(nn.Module):
    def __init__(self, in_channels=1, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = LIFCell()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = LIFCell()
        self.fc = nn.Linear(64 * 7 * 7, out_dim)
        self.lif3 = LIFCell()
        self.out_dim = out_dim

    def forward(self, x):
        # x: [T, B, C, H, W] with H=W=28 (MNIST-shaped)
        T, B = x.shape[0], x.shape[1]
        device, dtype = x.device, x.dtype
        v1 = torch.zeros(B, 32, 14, 14, device=device, dtype=dtype)
        v2 = torch.zeros(B, 64, 7, 7, device=device, dtype=dtype)
        v3 = torch.zeros(B, self.out_dim, device=device, dtype=dtype)
        out_spikes = []
        for t in range(T):
            c1 = self.pool1(self.conv1(x[t]))
            s1, v1 = self.lif1(c1, v1)
            c2 = self.pool2(self.conv2(s1))
            s2, v2 = self.lif2(c2, v2)
            flat = s2.flatten(1)
            s3, v3 = self.lif3(self.fc(flat), v3)
            out_spikes.append(s3)
        return torch.stack(out_spikes, dim=0)


class ALIFCell(nn.Module):
    """Adaptive-threshold LIF: threshold rises with recent spiking."""

    def __init__(self, decay=0.9, beta=1.8, adapt_decay=0.985, threshold=1.0):
        super().__init__()
        self.decay = decay
        self.beta = beta
        self.adapt_decay = adapt_decay
        self.threshold = threshold

    def forward(self, x, v, a):
        v = self.decay * v + x
        thresh = self.threshold + self.beta * a
        spike = spike_fn(v - thresh)
        v = v * (1.0 - spike)
        a = self.adapt_decay * a + spike
        return spike, v, a


class RecurrentSNN(nn.Module):
    """Single recurrent ALIF layer: hidden spikes feed back via a recurrent
    weight matrix each timestep, in addition to the feedforward input."""

    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.fc_rec = nn.Linear(hidden, hidden, bias=False)
        self.alif = ALIFCell()
        self.fc_out = nn.Linear(hidden, out_dim)
        self.lif_out = LIFCell()
        self.hidden = hidden
        self.out_dim = out_dim

    def forward(self, x):
        # x: [T, B, in_dim]
        T, B = x.shape[0], x.shape[1]
        device, dtype = x.device, x.dtype
        v = torch.zeros(B, self.hidden, device=device, dtype=dtype)
        a = torch.zeros(B, self.hidden, device=device, dtype=dtype)
        s = torch.zeros(B, self.hidden, device=device, dtype=dtype)
        v_out = torch.zeros(B, self.out_dim, device=device, dtype=dtype)
        out_spikes = []
        for t in range(T):
            drive = self.fc_in(x[t]) + self.fc_rec(s)
            s, v, a = self.alif(drive, v, a)
            s_out, v_out = self.lif_out(self.fc_out(s), v_out)
            out_spikes.append(s_out)
        return torch.stack(out_spikes, dim=0)


MODELS = {
    "mlp": MLPSNN,
    "conv": ConvSNN,
    "rnn": RecurrentSNN,
}
