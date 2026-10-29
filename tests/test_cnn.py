"""Verify the assigned architecture, saved weights, and real inference endpoint."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch
from torch import nn

from app.cnn_model import AssignmentCNN, CLASSES, image_transform
from app.image_classifier import CHECKPOINT, MAX_UPLOAD_BYTES, ImageClassifier
from app.main import app


@pytest.fixture(scope="module")
def client():
    torch.set_num_threads(2)
    with TestClient(app) as instance:
        yield instance


def test_architecture_and_gradient_update():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = AssignmentCNN()
    x = torch.rand(2, 3, 64, 64)
    sizes = []
    handles = [layer.register_forward_hook(lambda module, inputs, output: sizes.append(tuple(output.shape)))
               for layer in model.children()]
    logits = model(x)
    for handle in handles:
        handle.remove()
    assert sizes == [(2,16,64,64), (2,16,64,64), (2,16,32,32),
                     (2,32,32,32), (2,32,32,32), (2,32,16,16),
                     (2,8192), (2,100), (2,100), (2,10)]
    assert [type(layer) for layer in model.children()] == [
        nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU,
        nn.MaxPool2d, nn.Flatten, nn.Linear, nn.ReLU, nn.Linear]
    assert sum(p.numel() for p in model.parameters()) == 825398
    for layer in [model.conv1, model.conv2]:
        assert layer.kernel_size == (3,3) and layer.stride == (1,1) and layer.padding == (1,1)
    for layer in [model.pool1, model.pool2]:
        assert layer.kernel_size == 2 and layer.stride == 2
    original = model.conv1.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    nn.CrossEntropyLoss()(logits, torch.tensor([0,1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(original, model.conv1.weight)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros(1,3,32,32))


def test_saved_checkpoint_provenance():
    metrics = json.loads(Path("models/metrics.json").read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert metrics["train_samples"] == 50000 and metrics["test_samples"] == 10000
    assert metrics["test_accuracy"] == metrics["test_correct"] / 10000
    assert metrics["test_accuracy"] > 0.3
    assert len(metrics["history"]) == metrics["epochs"]
    matrix = torch.tensor(metrics["confusion_matrix_rows_true_columns_predicted"])
    assert int(matrix.sum()) == 10000 and int(matrix.diag().sum()) == metrics["test_correct"]


def test_api_matches_direct_checkpoint_inference(client):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    direct = AssignmentCNN()
    direct.load_state_dict(checkpoint["state_dict"])
    direct.eval()
    predictions = []
    for example in json.loads(Path("examples/labels.json").read_text()):
        path = Path("examples") / example["file"]
        response = client.post("/classify", files={"file": (path.name, path.read_bytes(), "image/png")})
        assert response.status_code == 200
        result = response.json()
        with Image.open(path) as image, torch.inference_mode():
            expected = direct(image_transform()(image.convert("RGB")).unsqueeze(0)).softmax(1)[0]
        assert result["class_id"] == int(expected.argmax())
        assert result["class_name"] == CLASSES[result["class_id"]]
        assert list(result["probabilities"]) == list(CLASSES)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert result["confidence"] == pytest.approx(float(expected.max()), abs=1e-6)
        torch.testing.assert_close(torch.tensor(list(result["probabilities"].values())), expected)
        predictions.append(result["probabilities"])
    assert predictions[0] != predictions[1]


@pytest.mark.parametrize("mode,format", [("L","PNG"), ("RGBA","PNG"), ("RGB","JPEG")])
def test_supported_images_are_converted_to_rgb(client, mode, format):
    buffer = BytesIO()
    Image.new(mode, (83,45)).save(buffer, format=format)
    assert client.post("/classify", files={"file": ("image", buffer.getvalue())}).status_code == 200


def test_invalid_uploads(client):
    assert client.post("/classify").status_code == 422
    for content in [b"", b"not an image", Path("examples/test_00_cat.png").read_bytes()[:40]]:
        assert client.post("/classify", files={"file": ("bad.png", content, "image/png")}).status_code == 422
    gif = BytesIO()
    Image.new("RGB", (8,8)).save(gif, format="GIF")
    assert client.post("/classify", files={"file": ("image.gif", gif.getvalue())}).status_code == 422
    assert client.post("/classify", files={"file": ("large.png", b"x" * (MAX_UPLOAD_BYTES+1))}).status_code == 413


def test_no_silent_random_model_when_weights_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageClassifier(tmp_path / "missing.pt")


def test_question_six_modes():
    dropout = nn.Dropout(1.0)
    assert dropout.training
    x = torch.ones(4, requires_grad=True)
    assert torch.equal(dropout(x), torch.zeros(4))
    dropout.eval()
    dropout(x).sum().backward()
    assert torch.equal(x.grad, torch.ones(4))
    batchnorm = nn.BatchNorm1d(2)
    batch = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    initial = batchnorm.running_mean.clone()
    batchnorm(batch)
    assert not torch.equal(initial, batchnorm.running_mean)
    batchnorm.eval()
    after_train = batchnorm.running_mean.clone()
    batchnorm(batch * 10)
    assert torch.equal(after_train, batchnorm.running_mean)
