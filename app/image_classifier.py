"""Load the trained CNN once and classify uploaded images on CPU."""
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from app.cnn_model import AssignmentCNN, CLASSES, image_transform

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pt"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


class ImageClassifier:
    def __init__(self, checkpoint_path=CHECKPOINT):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order does not match the application.")
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.transform = image_transform()

    def predict(self, content: bytes):
        if not content:
            raise ValueError("The uploaded image is empty.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise ValueError("Upload a PNG or JPEG image.")
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image must contain at most 16 million pixels.")
                    image.load()
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    inputs = self.transform(rgb).unsqueeze(0)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Upload a valid, readable PNG or JPEG image.") from exc
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0]
        index = int(probabilities.argmax().item())
        return {
            "class_id": index,
            "class_name": CLASSES[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {name: float(value) for name, value in zip(CLASSES, probabilities.tolist())},
        }
