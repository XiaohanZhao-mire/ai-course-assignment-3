"""Reproduce the bundled weights using only the CIFAR-10 training split."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.cifar_models import DiffusionModel, EnergyModel


def train(kind, images, epochs, batch_size, seed, out):
    torch.manual_seed(seed)
    model = DiffusionModel() if kind == "diffusion" else EnergyModel()
    if kind == "energy":
        model.fit_base(images)
    else:
        model.stats.fit(images)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001 if kind == "diffusion" else 0.0002
    )
    history = []
    for epoch in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(images)).split(batch_size):
            x = images[indices].float() / 127.5 - 1
            if kind == "diffusion":
                t = torch.randint(len(model.alpha_bar), (len(x),))
                a = model.alpha_bar[t, None, None, None]
                noise = torch.randn_like(x)
                noisy = a.sqrt() * x + (1 - a).sqrt() * noise
                loss = F.mse_loss(model(noisy, t), noise)
            else:
                # Short-run contrastive divergence. Detach generated negatives
                # before differentiating the energy difference w.r.t. weights.
                start = model.initial_images(torch.randn_like(x))
                negative = model.langevin(start, steps=3)
                real_e, neg_e = model.residual(x), model.residual(negative)
                # The fitted base energy is constant w.r.t. trainable parameters.
                loss = (real_e - neg_e).mean() + 0.01 * (
                    real_e.square() + neg_e.square()
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.item() * len(x)
        history.append(total / len(images))
        meta = dict(
            model=kind,
            dataset="CIFAR-10",
            split="train",
            training_images=len(images),
            epochs=epoch + 1,
            batch_size=batch_size,
            seed=seed,
            epoch_loss=history,
        )
        torch.save(
            {"state_dict": model.state_dict(), "metadata": meta},
            out / f"cifar_{kind}.pt",
        )
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["diffusion", "energy"])
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5560)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dataset = CIFAR10(args.data, train=True, download=True)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    train(
        args.model,
        images,
        args.epochs or (3 if args.model == "diffusion" else 1),
        args.batch_size,
        args.seed,
        ROOT / "models",
    )
