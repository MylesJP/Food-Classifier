# Using Food-101 dataset
# Starting out with smaller version with 3 classes

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchinfo

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE=32

# This is standard train/test directory format
image_path = Path("pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"

###
### Model 0 - Using TinyVGG arch without data augmentation to test the difference
###

# Create a simple transform
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data_simple = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform
)

test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform
)

train_dataloader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=True)

test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=False)  # No need to shuffle test data

# Create TinyVGG model class
class TinyVGG(nn.Module):
    """Model arch copying TinyVGG"""
    def __init__(self, 
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))  # This utilizes GPU better (operator fusion)
    
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,  # 3 colour channels
                  hidden_units=10,  # From TinyVGG arch
                  output_shape=3).to(device)  # Pizza, sushi, or steak (full one will be 101)

# Do a dummy forward pass on a single image batch to make sure hidden layers and shapes are correct

print(torchinfo.summary(model=model_0, input_size=[1, 3, 64, 64]))