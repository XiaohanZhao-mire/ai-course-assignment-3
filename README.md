# Assignment 3: MNIST GAN

Repository: [XiaohanZhao-mire/ai-course-assignment-3](https://github.com/XiaohanZhao-mire/ai-course-assignment-3)

A PyTorch GAN with the architecture in Assignment3.pdf, trained on MNIST and added to the course FastAPI application. Trained weights are included; API startup does not train a model or download MNIST.

## Run with Docker

```bash
docker build -t assignment3-gan .
docker run --rm -p 8000:8000 assignment3-gan
```

Open [Swagger UI](http://localhost:8000/docs). The container uses CPU inference and needs no GPU.

## Generate digits

`GET /generate-digit` returns a 28 x 28 grayscale PNG. Optional `count` (1-64) returns a grid; optional `seed` makes a request reproducible.

```bash
curl "http://localhost:8000/generate-digit?seed=42" -o digit.png
curl "http://localhost:8000/generate-digit?count=64&seed=2026" -o digits.png
```

`GET /gan-info` returns the actual training settings. This is an unconditional GAN: it generates random digit styles and cannot select a requested digit label. Some generated images are ambiguous.

Previous endpoints remain available: `POST /generate`, `GET /embedding?word=apple`, and `POST /classify` (image upload).

## Code and answers

- `app/gan_model.py`: generator and discriminator.
- `scripts/train_gan.py`: MNIST training with Adam and binary cross-entropy.
- `models/mnist_generator.pt`: trained generator used by the API.
- `models/gan_metadata.json` and `gan_training_history.json`: training settings and epoch losses.
- `examples/gan_training/`: the same 64 noise vectors rendered after each epoch.
- `docs/Assignment3_XiaohanZhao.docx`: all seven theory answers with calculations.

The discriminator's public output is a sigmoid probability. Training uses its logits with `BCEWithLogitsLoss`, which combines sigmoid and BCE stably. Real images are scaled to [-1, 1] to match the generator's Tanh output. MNIST labels are not used.

## Retrain or test locally

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --frozen
uv run python scripts/train_gan.py --epochs 20 --batch-size 128 --seed 42
uv run pytest -q
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# In another terminal:
uv run python scripts/smoke_test.py
```

Training selects CUDA, Apple MPS, or CPU automatically; results can vary across hardware. GitHub Actions builds the Docker image, runs the tests, and queries the running container over HTTP.
