"""CPU inference using the bundled trained MNIST generator."""

from io import BytesIO
from pathlib import Path
import secrets
from threading import Lock

from PIL import Image
import torch

from app.gan_model import Generator, LATENT_DIM

CHECKPOINT = Path(__file__).resolve().parents[1] / 'models' / 'mnist_generator.pt'


class DigitGenerator:
    def __init__(self, checkpoint_path=CHECKPOINT):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.metadata = checkpoint['metadata']
        if self.metadata['architecture'] != 'assignment3-mnist-gan-v1':
            raise ValueError('Incompatible GAN checkpoint.')
        self.model = Generator()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.lock = Lock()

    def generate_png(self, count=1, seed=None):
        if seed is None:
            seed = secrets.randbelow(2**32)
        # A private RNG makes concurrent seeded requests reproducible.
        rng = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(count, LATENT_DIM, generator=rng)
        with self.lock, torch.inference_mode():
            pixels = ((self.model(noise) + 1) * 127.5).clamp(0, 255).round().to(torch.uint8)
        columns = min(count, 8)
        rows = (count + columns - 1) // columns
        canvas = Image.new('L', (columns * 28, rows * 28))
        for index, digit in enumerate(pixels[:, 0].numpy()):
            canvas.paste(Image.fromarray(digit), ((index % columns) * 28, (index // columns) * 28))
        output = BytesIO()
        canvas.save(output, format='PNG')
        return output.getvalue(), seed
