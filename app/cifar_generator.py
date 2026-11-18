"""Load both checkpoints once and render a single unconditional RGB sample."""

from io import BytesIO
from pathlib import Path
import secrets

from PIL import Image
import torch

from app.cifar_models import DiffusionModel, EnergyModel


class CifarGenerators:
    def __init__(self):
        directory = Path(__file__).resolve().parents[1] / "models"
        self.models = {}
        for name, architecture in (
            ("diffusion", DiffusionModel),
            ("energy", EnergyModel),
        ):
            checkpoint = torch.load(
                directory / f"cifar_{name}.pt", map_location="cpu", weights_only=True
            )
            model = architecture()
            model.load_state_dict(checkpoint["state_dict"])
            self.models[name] = model.eval().requires_grad_(False)

    def generate(self, name, seed):
        seed = secrets.randbits(32) if seed is None else seed
        rng = torch.Generator().manual_seed(seed)
        image = self.models[name].sample(rng)[0]
        pixels = ((image + 1) * 127.5).round().clamp(0, 255).byte()
        output = BytesIO()
        Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(output, format="PNG")
        return output.getvalue(), seed
