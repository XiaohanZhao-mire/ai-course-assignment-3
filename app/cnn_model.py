"""Assignment 2 architecture, adapted from Module 4 Practical 3 SimpleCNN."""
import torch
from torch import nn
from torchvision import transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def image_transform():
    # CIFAR-10 is 32x32; the assignment explicitly requires 64x64 RGB input.
    # This same transform is used for training, evaluation, and API inference.
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
    ])


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
            raise ValueError("Expected input shape [batch, 3, 64, 64].")
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.fc1(self.flatten(x)))
        # CrossEntropyLoss expects logits, so there is no training-time softmax.
        return self.fc2(x)
