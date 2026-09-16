"""Equivalent models built with SpikingJelly's activation_based API in
multi-step mode, so they use its fused CuPy multi-step kernels rather than
single-step Python looping. This is thesis-1's existing competitor: a
version of "fuse the time loop" that already ships. Workstream A needs to
know exactly where it plateaus.

Requires: pip install --no-deps spikingjelly (see README "Install" section for why --no-deps)
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional, surrogate


def _lif():
    return neuron.LIFNode(
        tau=1.0 / (1.0 - 0.9),
        surrogate_function=surrogate.Sigmoid(alpha=10.0),
        step_mode="m",
        backend="cupy",
    )


class MLPSNN_SJ(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            layer.Linear(in_dim, hidden),
            _lif(),
            layer.Linear(hidden, out_dim),
            _lif(),
        )
        functional.set_step_mode(self.net, step_mode="m")

    def forward(self, x):
        # x: [T, B, in_dim]
        functional.reset_net(self.net)
        return self.net(x)


class ConvSNN_SJ(nn.Module):
    def __init__(self, in_channels=1, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels, 32, 3, padding=1),
            layer.MaxPool2d(2),
            _lif(),
            layer.Conv2d(32, 64, 3, padding=1),
            layer.MaxPool2d(2),
            _lif(),
            layer.Flatten(),
            layer.Linear(64 * 7 * 7, out_dim),
            _lif(),
        )
        functional.set_step_mode(self.net, step_mode="m")

    def forward(self, x):
        # x: [T, B, C, H, W]
        functional.reset_net(self.net)
        return self.net(x)


class RecurrentSNN_SJ(nn.Module):
    """SpikingJelly doesn't have a stock ALIF-with-recurrent-weights block,
    so this composes its primitives step-by-step (single-step mode) --
    the fairest apples-to-apples comparison against models.RecurrentSNN,
    since neither gets a fused multi-step kernel for the recurrent case."""

    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.fc_rec = nn.Linear(hidden, hidden, bias=False)
        self.alif = neuron.LIFNode(
            tau=1.0 / (1.0 - 0.9),
            surrogate_function=surrogate.Sigmoid(alpha=10.0),
            step_mode="s",
            backend="cupy",
        )
        self.fc_out = nn.Linear(hidden, out_dim)
        self.lif_out = neuron.LIFNode(
            tau=1.0 / (1.0 - 0.9),
            surrogate_function=surrogate.Sigmoid(alpha=10.0),
            step_mode="s",
            backend="cupy",
        )
        self.hidden = hidden
        self.out_dim = out_dim

    def forward(self, x):
        # x: [T, B, in_dim]
        functional.reset_net(self)
        T, B = x.shape[0], x.shape[1]
        s = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
        out_spikes = []
        for t in range(T):
            drive = self.fc_in(x[t]) + self.fc_rec(s)
            s = self.alif(drive)
            s_out = self.lif_out(self.fc_out(s))
            out_spikes.append(s_out)
        return torch.stack(out_spikes, dim=0)


MODELS_SJ = {
    "mlp": MLPSNN_SJ,
    "conv": ConvSNN_SJ,
    "rnn": RecurrentSNN_SJ,
}
