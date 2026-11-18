# Assignment 4 · Advanced image generation

This extends the Module 6 FastAPI app with two unconditional CIFAR-10 generators. Both trained checkpoints ship in the image and run on CPU. The assignment answers, derivations and generated samples are in [Assignment4_XiaohanZhao.pdf](docs/Assignment4_XiaohanZhao.pdf).

## Run with Docker

From the directory containing the Dockerfile:

```sh
docker build -t assignment4-cifar .
docker run --rm -p 8000:8000 assignment4-cifar
```

The build downloads dependencies, including the English vector model used by the earlier embedding endpoint. Inference uses the bundled weights. Open [Swagger UI](http://localhost:8000/docs) or run these commands in a second terminal.

```sh
curl -f 'http://localhost:8000/generate-image/diffusion?seed=42' -o diffusion.png
curl -f 'http://localhost:8000/generate-image/energy?seed=42' -o energy.png
```

Both routes return one 32 × 32 RGB PNG. Omit `seed` for a random sample. The `X-Seed` header gives the unsigned 32-bit seed used, so a request can be repeated on the same software and hardware. The endpoints generate unlabeled scenes. Small-scale textures and colors are more consistent than object identity.

## Training and sampling

Both models use all 50,000 CIFAR-10 training images scaled to [−1, 1], batch size 64 and seed 5560. Labels and the CIFAR-10 test split are excluded from training.

| Model | Training | Inference |
| --- | --- | --- |
| Diffusion | 64-step cosine schedule. A time-conditioned U-Net predicts noise with a Gaussian covariance estimate and a learned residual. Three epochs of noise-prediction MSE, averaging 0.08324, 0.07817 and 0.07472. | Start with independent Gaussian noise and draw 64 reverse DDPM updates with a clipped clean-image estimate. |
| Energy | Fit 256 layout clusters and a shared spatial RGB covariance. Learn a convolutional correction to the mixture energy by contrasting real samples with three-step Langevin negatives for one epoch. A squared-energy penalty regularizes the correction. | Draw a fresh mixture sample, then run 64 covariance-preconditioned Langevin updates on the pixels. Output pixels are clipped to the display range. |

Inside the EBM sampler, `torch.autograd.grad(energy.sum(), image)` targets pixels. Each update detaches the previous image, and parameter gradients do not accumulate during sampling. In the learning step, `loss.backward()` differentiates through the learned energies of fixed positive and negative samples to update the CNN.

## Reproduce and verify

With Python 3.12 and uv installed, run the following commands from this directory. Retraining downloads CIFAR-10 on first use and writes the trained checkpoints into `models/`.

```sh
uv sync --frozen
uv run python scripts/train_cifar_generators.py diffusion
uv run python scripts/train_cifar_generators.py energy
uv run python scripts/check_theory.py
uv run pytest -q
```

The architecture and samplers are in `app/cifar_models.py`. FastAPI loads the weights through `app/cifar_generator.py`. The existing text, embedding, classification and MNIST routes are available at `POST /generate`, `GET /embedding`, `POST /classify`, `GET /generate-digit` and `GET /gan-info`.

The GitHub Actions workflow builds the Docker image, runs the test suite in it, then calls the running service over HTTP. For a local deployment check, start the container and run `python3 scripts/smoke_test.py`.
