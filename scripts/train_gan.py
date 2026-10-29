"""Train the assignment GAN on all 60,000 MNIST training images."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.gan_model import Generator, Discriminator, LATENT_DIM, initialize_weights


def train_step(generator, discriminator, real, opt_g, opt_d):
    batch_size = real.shape[0]
    criterion = nn.BCEWithLogitsLoss()
    ones = torch.ones(batch_size, 1, device=real.device)
    zeros = torch.zeros_like(ones)
    discriminator.train()
    discriminator.requires_grad_(True)
    opt_d.zero_grad(set_to_none=True)
    noise = torch.randn(batch_size, LATENT_DIM, device=real.device)
    fake = generator(noise)
    # Detach prevents a discriminator update from changing the generator.
    loss_d = criterion(discriminator.logits(real), ones)
    loss_d = loss_d + criterion(discriminator.logits(fake.detach()), zeros)
    loss_d.backward()
    opt_d.step()

    # Freeze D weights while retaining the gradient path through D to G.
    discriminator.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    loss_g = criterion(discriminator.logits(fake), ones)
    loss_g.backward()
    opt_g.step()
    discriminator.requires_grad_(True)
    return loss_d.detach(), loss_g.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error('epochs >= 1, batch-size >= 2 and lr > 0 are required')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = MNIST(args.data_dir, train=True, download=True)
    # Keep this small dataset in memory; labels are deliberately unused.
    images = dataset.data.unsqueeze(1).to(device=device, dtype=torch.float32) / 127.5 - 1.0
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, LATENT_DIM, generator=torch.Generator().manual_seed(2026)).to(device)
    models = args.output_dir / 'models'
    examples = args.output_dir / 'examples' / 'gan_training'
    models.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    print(json.dumps({'device': device, 'training_images': len(dataset), 'epochs': args.epochs,
                      'batch_size': args.batch_size, 'seed': args.seed}), flush=True)
    for epoch in range(1, args.epochs + 1):
        generator.train()
        permutation = torch.randperm(len(images), device=device)
        totals = torch.zeros(2, device=device)
        epoch_start = time.perf_counter()
        for batch_index, indices in enumerate(permutation.split(args.batch_size)):
            loss_d, loss_g = train_step(generator, discriminator, images[indices], opt_g, opt_d)
            totals += torch.stack((loss_d, loss_g)) * len(indices)
            if batch_index % 100 == 0:
                print(f'epoch={epoch} batch={batch_index} D={loss_d.item():.4f} G={loss_g.item():.4f}', flush=True)
        means = (totals / len(images)).cpu().tolist()
        record = {'epoch': epoch, 'discriminator_loss': means[0], 'generator_loss': means[1],
                  'seconds': round(time.perf_counter() - epoch_start, 2)}
        history.append(record)
        generator.eval()
        with torch.inference_mode():
            samples = (generator(fixed_noise).cpu() + 1) / 2
        save_image(samples, examples / f'epoch_{epoch:02d}.png', nrow=8, padding=2)
        metadata = {
            'architecture': 'assignment3-mnist-gan-v1', 'latent_dim': LATENT_DIM,
            'dataset': 'MNIST training split', 'training_images': len(dataset),
            'epochs': epoch, 'batch_size': args.batch_size, 'learning_rate': args.lr,
            'optimizer': 'Adam', 'betas': [0.5, 0.999], 'seed': args.seed,
            'device': device, 'torch_version': str(torch.__version__),
            'normalization': 'uint8 / 127.5 - 1',
            'loss': 'BCEWithLogitsLoss; D real+fake; G non-saturating',
            'sample_seed': 2026, 'elapsed_seconds': round(time.perf_counter() - started, 2),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        torch.save({'state_dict': {k: v.cpu() for k, v in generator.state_dict().items()},
                    'metadata': metadata}, models / 'mnist_generator.pt')
        torch.save({'state_dict': {k: v.cpu() for k, v in discriminator.state_dict().items()},
                    'metadata': metadata}, models / 'mnist_discriminator.pt')
        (models / 'gan_training_history.json').write_text(json.dumps(history, indent=2) + '\n')
        (models / 'gan_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
