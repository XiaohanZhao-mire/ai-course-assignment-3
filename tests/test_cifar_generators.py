from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import pytest
import torch

from app.main import app
from app.cifar_models import EnergyModel, timestep_embedding


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as service:
        yield service


@pytest.mark.parametrize("model", ["diffusion", "energy"])
def test_generation(client, model):
    path = f"/generate-image/{model}"
    r = client.get(path, params={"seed": 42})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["x-seed"] == "42"
    assert r.content == client.get(path, params={"seed": 42}).content
    assert r.content != client.get(path, params={"seed": 43}).content
    with Image.open(BytesIO(r.content)) as image:
        assert image.size == (32, 32) and image.mode == "RGB"
        assert np.asarray(image).std() > 1
    assert client.get(path, params={"seed": -1}).status_code == 422


def test_langevin_gradient_target():
    torch.manual_seed(13)
    model = EnergyModel()
    x = torch.randn(1, 3, 32, 32)
    before = x.clone()
    result = model.langevin(x, steps=2)
    assert torch.equal(before, x) and not torch.equal(result, x)
    assert not result.requires_grad
    assert all(p.grad is None for p in model.parameters())
    # The detached negative can now be used to train the energy network.
    (model(x) - model(result)).sum().backward()
    assert model.features[-1].weight.grad.abs().sum() > 0


def test_embedding_order():
    vector = timestep_embedding(torch.tensor([1]), 8)[0]
    angle = torch.tensor([1.0, 0.1, 0.01, 0.001])
    torch.testing.assert_close(vector[::2], angle.sin())
    torch.testing.assert_close(vector[1::2], angle.cos())
