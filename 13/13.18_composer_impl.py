# Install the mosaicml library
# pip install mosaicml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import ComposerClassifier
from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for MNIST classification.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        # Define the network layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16 * 16, 32)  # Adjusted for output size
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


# Define data transformation and load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Set up the training configuration
trainer = Trainer(
    model=ComposerClassifier(module=SimpleCNN(), num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",  # Duration of training
    algorithms=[
        LabelSmoothing(smoothing=0.1),  # Apply label smoothing
        CutMix(alpha=1.0),  # Apply CutMix data augmentation
        ChannelsLast(),  # Optimize for channels-last format
    ],
)

# Start training
trainer.fit()

# Notebook here:
# https://colab.research.google.com/github/mosaicml/composer/blob/9f594876f957c912758e540598ac9f47a468c39d/examples/getting_started.ipynb
