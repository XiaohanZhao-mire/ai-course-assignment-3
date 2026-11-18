"""Numerical checks for Assignment 4. Run from any directory."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.cifar_models import timestep_embedding

print("Q2", timestep_embedding(torch.tensor([1]), 8).tolist()[0])
x = torch.tensor([2.0], requires_grad=True)
(x**2 + 3 * x).backward()
print("Q6a", x.grad)
x = torch.tensor([2.0], requires_grad=False)
try:
    (x**2 + 3 * x).backward()
except RuntimeError as exc:
    print("Q6b", str(exc), "x.grad =", x.grad)
print("Q6c / Q7c", torch.tensor([2.0]).requires_grad)

for track_weights in (False, True):
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0], requires_grad=track_weights)
    y = w[0] * x**2 + w[1] * x
    y.backward()
    print("Q7", "x.grad =", x.grad, "w.grad =", w.grad)

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.detach()
try:
    (z * 2).backward()
except RuntimeError as exc:
    print("Q8 original", str(exc))
z = y.clone()
(z * 2).backward()
print("Q8 fixed", x.grad)

for reset in (False, True):
    x = torch.tensor([1.0], requires_grad=True)
    (x * 2).backward()
    first = x.grad.clone()
    if reset:
        x.grad = None
    (x * 3).backward()
    print("Q9", "reset =", reset, "first =", first, "second =", x.grad)
