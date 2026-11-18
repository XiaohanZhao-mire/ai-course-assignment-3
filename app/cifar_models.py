"""Small unconditional CIFAR-10 diffusion and energy models in RGB pixel space."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ImageStatistics(nn.Module):
    """Stationary RGB covariance estimated from training images in Fourier space."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("basis", torch.eye(3).expand(32, 17, 3, 3).contiguous())
        self.register_buffer("variance", torch.ones(32, 17, 3))

    @torch.no_grad()
    def fit(self, data, centers=None, labels=None):
        self.mean.copy_(data.float().mean((0, 2, 3), keepdim=True) / 127.5 - 1)
        moment = torch.zeros_like(self.basis)
        for start in range(0, len(data), 256):
            batch = data[start : start + 256]
            x = batch.float() / 127.5 - 1 - self.mean
            if centers is not None:
                x = x - centers[labels[start : start + 256]]
            spectrum = torch.fft.rfft2(x, norm="ortho").permute(2, 3, 0, 1)
            moment += (spectrum.mH @ spectrum).real
        eigenvalues, eigenvectors = torch.linalg.eigh(moment / len(data))
        self.variance.copy_(eigenvalues.clamp_min(0.0025))
        self.basis.copy_(eigenvectors)

    def filter(self, x, gain):
        spectrum = torch.fft.rfft2(x, norm="ortho").permute(0, 2, 3, 1)
        basis = self.basis.to(spectrum.dtype)
        coefficients = (spectrum.unsqueeze(-2) @ basis).squeeze(-2)
        transformed = ((gain * coefficients).unsqueeze(-2) @ basis.mT).squeeze(-2)
        return torch.fft.irfft2(
            transformed.permute(0, 3, 1, 2), s=(32, 32), norm="ortho"
        )

    def draw(self, noise):
        return self.mean + self.filter(noise, self.variance.sqrt())


def timestep_embedding(t, width=32):
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=t.device) / (width // 2)
    )
    angles = t[:, None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


def alpha_products(steps=64):
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
    return (1 - betas).cumprod(0).float()


class TimeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(48, channels)
        self.norm = nn.GroupNorm(4, channels)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, time):
        h = self.conv(x) + self.time(time)[:, :, None, None]
        return x + self.out(F.silu(self.norm(h)))


class DiffusionModel(nn.Module):
    """A timestep-conditioned U-Net learns a residual over a Gaussian noise estimate."""

    def __init__(self):
        super().__init__()
        self.stats = ImageStatistics()
        self.register_buffer("alpha_bar", alpha_products())
        self.time = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 48))
        self.input = nn.Conv2d(24, 24, 3, padding=1)
        self.encoder = TimeBlock(24)
        self.down = nn.Conv2d(24, 40, 4, stride=2, padding=1)
        self.middle = TimeBlock(40)
        self.up = nn.Conv2d(40, 24, 3, padding=1)
        self.decoder = TimeBlock(24)
        self.output = nn.Conv2d(24, 12, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t):
        a = self.alpha_bar[t, None, None, None]
        noise_scale = (1 - a).sqrt()
        gain = noise_scale / (a * self.stats.variance + 1 - a)
        baseline = self.stats.filter(x - a.sqrt() * self.stats.mean, gain)
        time = self.time(timestep_embedding(t))
        packed = F.pixel_unshuffle(torch.cat((x, baseline), dim=1), 2)
        skip = self.encoder(self.input(packed), time)
        low = self.middle(self.down(skip), time)
        high = self.up(F.interpolate(low, scale_factor=2, mode="nearest"))
        correction = F.pixel_shuffle(self.output(self.decoder(high + skip, time)), 2)
        return baseline + a.sqrt() * correction

    @torch.no_grad()
    def sample(self, rng):
        x = torch.randn(1, 3, 32, 32, generator=rng)
        for t in reversed(range(len(self.alpha_bar))):
            a = self.alpha_bar[t]
            prev = self.alpha_bar[t - 1] if t else x.new_tensor(1.0)
            clean = (
                (x - (1 - a).sqrt() * self(x, torch.tensor([t]))) / a.sqrt()
            ).clamp(-1, 1)
            beta = 1 - a / prev
            x = (prev.sqrt() * beta * clean + (a / prev).sqrt() * (1 - prev) * x) / (
                1 - a
            )
            if t:
                x = x + (beta * (1 - prev) / (1 - a)).sqrt() * torch.randn(
                    x.shape, generator=rng
                )
        return x


class EnergyModel(nn.Module):
    """A Gaussian-mixture image energy plus a trainable convolutional correction."""

    def __init__(self):
        super().__init__()
        self.stats = ImageStatistics()
        self.register_buffer("centers", torch.zeros(256, 3, 32, 32))
        self.register_buffer("precision_centers", torch.zeros_like(self.centers))
        self.register_buffer("log_weights", torch.zeros(256))
        self.register_buffer("offsets", torch.zeros(256))
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)

    @torch.no_grad()
    def fit_base(self, images):
        # Fit coarse layout clusters, then estimate their full-resolution means.
        low = torch.cat(
            [
                F.avg_pool2d(x.float() / 127.5 - 1, 4).flatten(1)
                for x in images.split(256)
            ]
        )
        k = len(self.centers)
        means = low[torch.randperm(len(low))[:k]]
        for _ in range(10):
            groups = torch.cat(
                [torch.cdist(batch, means).argmin(1) for batch in low.split(512)]
            )
            counts = torch.bincount(groups, minlength=k)
            sums = torch.zeros_like(means).index_add_(0, groups, low)
            means = torch.where(
                counts[:, None] > 0, sums / counts[:, None].clamp_min(1), means
            )
        for start in range(0, len(images), 256):
            x = images[start : start + 256].float() / 127.5 - 1
            self.centers.index_add_(0, groups[start : start + len(x)], x)
        self.centers.div_(counts.clamp_min(1)[:, None, None, None])
        self.centers.sub_(images.float().mean((0, 2, 3), keepdim=True) / 127.5 - 1)
        self.stats.fit(images, self.centers, groups)
        self.precision_centers.copy_(
            self.stats.filter(self.centers, self.stats.variance.reciprocal())
        )
        self.log_weights.copy_((counts.float() / len(low)).log())
        self.offsets.copy_(
            self.log_weights
            - 0.5 * (self.centers * self.precision_centers).sum((1, 2, 3))
        )

    def initial_images(self, noise, rng=None):
        group = torch.multinomial(
            self.log_weights.softmax(0), len(noise), replacement=True, generator=rng
        )
        return self.stats.draw(noise) + self.centers[group]

    def residual(self, x):
        return self.features(x).sum((1, 2, 3))

    def forward(self, x):
        centered = x - self.stats.mean
        precision_x = self.stats.filter(centered, self.stats.variance.reciprocal())
        logits = F.linear(
            centered.flatten(1), self.precision_centers.flatten(1), self.offsets
        )
        base = 0.5 * (centered * precision_x).sum((1, 2, 3)) - logits.logsumexp(1)
        return base + self.residual(x)

    def langevin(self, image, steps, rng=None, rate=0.025):
        # Constant preconditioner C speeds motion along high-variance RGB directions.
        for _ in range(steps):
            with torch.enable_grad():
                image = image.detach().requires_grad_(True)
                (gradient,) = torch.autograd.grad(self(image).sum(), image)
            with torch.no_grad():
                noise = torch.randn(image.shape, generator=rng, device=image.device)
                drift = self.stats.filter(gradient, self.stats.variance)
                diffusion = self.stats.filter(noise, self.stats.variance.sqrt())
                image = image - rate * drift + math.sqrt(2 * rate) * diffusion
        return image.detach()

    def sample(self, rng):
        x = self.initial_images(torch.randn(1, 3, 32, 32, generator=rng), rng)
        return self.langevin(x, 64, rng).clamp(-1, 1)
