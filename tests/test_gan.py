from io import BytesIO
import math

from PIL import Image
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient

from app.gan_model import Generator, Discriminator
from app.digit_generator import DigitGenerator
from app.main import app
from scripts.train_gan import train_step


def test_architecture_shapes_and_probabilities():
    generator, discriminator = Generator(), Discriminator()
    shapes = []
    handles = [layer.register_forward_hook(lambda layer, args, out: shapes.append(tuple(out.shape)))
               for layer in generator.layers]
    fake = generator(torch.randn(4, 100))
    for handle in handles:
        handle.remove()
    assert shapes == [(4,128,7,7), (4,64,14,14), (4,64,14,14),
                      (4,64,14,14), (4,1,28,28), (4,1,28,28)]
    assert generator.fc.in_features == 100
    assert generator.fc.out_features == 6272
    assert all(layer.kernel_size == (4,4) and layer.stride == (2,2)
               and layer.padding == (1,1) for layer in generator.layers
               if isinstance(layer, nn.ConvTranspose2d))
    convs = [layer for layer in discriminator.layers if isinstance(layer, nn.Conv2d)]
    assert [(layer.in_channels, layer.out_channels) for layer in convs] == [(1,64),(64,128)]
    assert all(layer.kernel_size == (4,4) and layer.stride == (2,2)
               and layer.padding == (1,1) for layer in convs)
    assert [layer.negative_slope for layer in discriminator.layers
            if isinstance(layer, nn.LeakyReLU)] == [0.2,0.2]
    assert fake.min() >= -1 and fake.max() <= 1
    probabilities = discriminator(fake)
    assert probabilities.shape == (4,1)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_training_step_updates_both_models():
    generator, discriminator = Generator(), Discriminator()
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    before_g = generator.fc.weight.detach().clone()
    before_d = discriminator.layers[0].weight.detach().clone()
    loss_d, loss_g = train_step(generator, discriminator, torch.rand(4,1,28,28)*2-1, opt_g, opt_d)
    assert math.isfinite(loss_d.item()) and math.isfinite(loss_g.item())
    assert not torch.equal(before_g, generator.fc.weight)
    assert not torch.equal(before_d, discriminator.layers[0].weight)


def test_theory_calculations_against_pytorch():
    for stride, expected in [(2,17),(3,24)]:
        layer = nn.ConvTranspose2d(1,1,4,stride,1,output_padding=1)
        assert layer(torch.zeros(1,1,8,8)).shape[-2:] == (expected,expected)
    assert nn.ConvTranspose2d(1,1,2,2)(torch.zeros(1,1,16,16)).shape[-2:] == (32,32)
    data = torch.tensor([6.,8.,10.,6.]).reshape(4,1)
    result = nn.BatchNorm1d(1,affine=False,eps=0)(data).flatten()
    expected = torch.tensor([-3,1,5,-3]) / math.sqrt(11)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.var(unbiased=False), torch.tensor(1.))


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_actual_checkpoint_and_seeded_png(client):
    response = client.get('/generate-digit', params={'seed':42})
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert response.headers['x-seed'] == '42'
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (28,28) and image.mode == 'L'
        assert image.getextrema()[1] - image.getextrema()[0] > 100
    assert response.content == client.get('/generate-digit?seed=42').content
    assert response.content != client.get('/generate-digit?seed=43').content
    metadata = client.get('/gan-info').json()
    assert metadata['training_images'] == 60000 and metadata['epochs'] >= 1


def test_api_pixels_match_direct_trained_generator(client):
    checkpoint = torch.load('models/mnist_generator.pt', map_location='cpu', weights_only=True)
    direct = Generator()
    direct.load_state_dict(checkpoint['state_dict'])
    direct.eval()
    noise = torch.randn(1,100,generator=torch.Generator().manual_seed(99))
    with torch.inference_mode():
        expected = ((direct(noise) + 1) * 127.5).clamp(0,255).round().to(torch.uint8)[0,0]
    response = client.get('/generate-digit?seed=99')
    with Image.open(BytesIO(response.content)) as image:
        assert list(image.getdata()) == expected.flatten().tolist()


@pytest.mark.parametrize('count,size', [(1,(28,28)),(9,(224,56)),(64,(224,224))])
def test_grid_dimensions(client, count, size):
    response = client.get('/generate-digit',params={'count':count,'seed':1})
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == size


@pytest.mark.parametrize('query', ['count=0','count=65','count=1.5','count=abc',
                                 'seed=-1','seed=4294967296','seed=abc'])
def test_invalid_gan_inputs(client, query):
    assert client.get('/generate-digit?' + query).status_code == 422


def test_unseeded_generation_returns_reproducible_seed(client):
    response = client.get('/generate-digit')
    assert response.status_code == 200
    seed = response.headers['x-seed']
    assert response.content == client.get('/generate-digit?seed=' + seed).content


def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        DigitGenerator(tmp_path / 'missing.pt')
