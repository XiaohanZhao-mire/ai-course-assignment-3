"""Train the required CNN on CIFAR-10; run from the project root.

uv run python -m scripts.train_cnn --epochs 10 --batch-size 32 --device auto
Adapted from the explicit training loop in Module 4 Practical 3.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive

from app.cnn_model import AssignmentCNN, CLASSES, image_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--dataset-url", default=None, help="Optional mirror; the official CIFAR-10 MD5 is still enforced")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device}; epochs: {args.epochs}; batch size: {args.batch_size}", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset_url:
        download_and_extract_archive(args.dataset_url, args.data_dir,
                                     filename=CIFAR10.filename, md5=CIFAR10.tgz_md5)
    train_data = CIFAR10(args.data_dir, train=True, download=True, transform=image_transform())
    # Official test split is evaluated only after all fixed training epochs.
    test_data = CIFAR10(args.data_dir, train=False, download=True, transform=image_transform())
    assert tuple(train_data.classes) == CLASSES
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_start = time.perf_counter()
        for batch, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if batch % 400 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} batch {batch}/{len(train_loader)} loss {total_loss/total:.4f}", flush=True)
        row = {"epoch": epoch + 1, "train_loss": total_loss / total,
               "train_accuracy": correct / total, "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predicted = logits.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            indices = (labels * 10 + predicted).cpu()
            confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    checkpoint = output / "cifar10_cnn.pt"
    torch.save({"state_dict": weights, "classes": list(CLASSES), "input_shape": [3, 64, 64],
                "epochs": args.epochs, "seed": args.seed}, checkpoint)
    metrics = {
        "dataset": "CIFAR-10 official split", "train_samples": len(train_data), "test_samples": total,
        "dataset_download_url": args.dataset_url or CIFAR10.url,
        "dataset_archive_expected_md5": CIFAR10.tgz_md5,
        "test_correct": correct, "test_accuracy": correct / total, "test_loss": total_loss / total,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(), "classes": list(CLASSES),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": "Adam", "loss": "CrossEntropyLoss", "seed": args.seed,
        "training_device": device_name, "torch_version": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()),
        "preprocessing": "PIL RGB; bilinear resize to 64x64; ToTensor scales to [0,1]; no augmentation or normalization",
        "model_selection": f"Final epoch of a fixed {args.epochs}-epoch run; no selection using test results",
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Fixed first ten test images provide transparent examples, not selected successes.
    raw_test = CIFAR10(args.data_dir, train=False, download=False)
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    examples = []
    for index in range(10):
        image, label = raw_test[index]
        filename = f"test_{index:02d}_{CLASSES[label]}.png"
        image.save(examples_dir / filename)
        examples.append({"file": filename, "test_index": index, "true_class_id": label, "true_class_name": CLASSES[label]})
    (examples_dir / "labels.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(json.dumps({"test_correct": correct, "test_total": total, "test_accuracy": correct / total,
                      "checkpoint": str(checkpoint), "sha256": metrics["checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
