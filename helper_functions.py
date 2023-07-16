# Helper functions for food 101
import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
import data_setup
import engine
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import save_model
import os
import zipfile
from pathlib import Path
import requests

def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def download_data(
    source: str,
    destination: str,
    remove_source: bool = True) -> Path:
    """Downloads zipped dataset from source and unzips to destination"""
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print("Directory already exists, skipping download")
    else:
        image_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print("Downloading target file from source")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path / target_file)

    return image_path

def create_effnetb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    # Freeze base model layers since they're already good
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head
    set_seeds()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=OUT_FEATURES)  # Check docs for proper p and in_features
    ).to(device)

    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model...")
    return model