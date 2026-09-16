"""Input data for the benchmarks.

Default is synthetic (no dataset download / disk I/O needed on a fresh
remote box). --data mnist rate-codes real MNIST digits into spike trains.

MNIST loading is hand-rolled (urllib + gzip + struct) instead of going
through torchvision: torchvision drags in Pillow and its own binary wheel
just to decode 28x28 uint8 images, which is a lot of disk for four small
IDX files we can parse in a few lines. See README's "disk space" section.
"""

import gzip
import os
import struct
import urllib.request

import torch

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte.gz"


def synthetic_batch(T, batch_size, in_shape, device, p_spike=0.1):
    """Bernoulli spike train, iid per timestep. in_shape excludes T and B,
    e.g. (784,) for MLP, (1, 28, 28) for conv."""
    shape = (T, batch_size) + tuple(in_shape)
    return (torch.rand(shape, device=device) < p_spike).float()


def _download(name, data_root):
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(MNIST_BASE_URL + name, path)
    return path


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        buf = f.read(n * rows * cols)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(n, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {path}"
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).long()


def mnist_rate_coded(T, batch_size, device, flatten=True, data_root="./data"):
    img_path = _download(MNIST_TRAIN_IMAGES, data_root)
    lbl_path = _download(MNIST_TRAIN_LABELS, data_root)

    images = _read_idx_images(img_path).to(device).float() / 255.0  # [N, 28, 28] in [0, 1]
    labels = _read_idx_labels(lbl_path).to(device)
    n = images.shape[0]

    def make_batch():
        idx = torch.randint(0, n, (batch_size,), device=device)
        batch = images[idx].unsqueeze(1)  # [B, 1, 28, 28]
        if flatten:
            batch = batch.flatten(1)  # [B, 784]
        # rate-code: spike probability proportional to pixel intensity
        spikes = torch.rand((T,) + batch.shape, device=device) < batch.unsqueeze(0)
        return spikes.float(), labels[idx]

    return make_batch
