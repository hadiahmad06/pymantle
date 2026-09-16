"""Input data for the benchmarks.

Default is synthetic (no dataset download / disk I/O needed on a fresh
remote box). --data mnist rate-codes real MNIST digits into spike trains
via torchvision, for when we want realistic firing-rate numbers rather than
just kernel timing.
"""

import torch


def synthetic_batch(T, batch_size, in_shape, device, p_spike=0.1):
    """Bernoulli spike train, iid per timestep. in_shape excludes T and B,
    e.g. (784,) for MLP, (1, 28, 28) for conv."""
    shape = (T, batch_size) + tuple(in_shape)
    return (torch.rand(shape, device=device) < p_spike).float()


def mnist_rate_coded(T, batch_size, device, flatten=True, data_root="./data"):
    import torchvision
    import torchvision.transforms as transforms

    tfm = transforms.ToTensor()
    ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def make_batch():
        images, labels = next(iter(loader))
        images = images.to(device)  # [B, 1, 28, 28] in [0, 1]
        if flatten:
            images = images.flatten(1)  # [B, 784]
        # rate-code: spike probability proportional to pixel intensity
        spikes = torch.rand((T,) + images.shape, device=device) < images.unsqueeze(0)
        return spikes.float(), labels.to(device)

    return make_batch
