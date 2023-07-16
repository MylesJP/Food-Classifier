# Food vision big with 101 classes - 101,000 images

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
from helper_functions import set_seeds, download_data, create_effnetb2
import gradio as gr
from time import timer
import random
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

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

data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")
                                    
train_dir = data_20_percent_path / "train"
test_dir = data_20_percent_path / "test"

def create_effnetb2_model(num_classes:int=3,  # Default is 3 (pizza, steak, sushi)
                          seed:int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False  # Don't want to track the gradients

    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3, seed=42)

# Create dataloaders
train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=effnetb2_transforms,
                                                                                                 batch_size=32)

# Let's train - need loss function, optimizer, training function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=effnetb2.parameters(),
                             lr=1e-3)
set_seeds()
effnetb2_results = engine.train(model=effnetb2,
                                train_dataloader=train_dataloader_effnetb2,
                                test_dataloader=test_dataloader_effnetb2,
                                epochs=10,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)

# Save to file
save_model.save_model(model=effnetb2,
                      target_dir="models",
                      model_name="pretrained_effnetb2_20.pth")

# Working with gradio
# Need to put model on CPU
effnetb2 = effnetb2.to("cpu")

def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()
    
    # Transform input image for use with the model
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into eval mode, make prediction
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create prediction label and prediction prop dict
    pred_labels_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    return pred_labels_probs, pred_time

# List of examples
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
random_image_path = random.sample(test_data_paths, k=1)[0]
image = Image.open(random_image_path)

example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

# Let's use Gradio
# Create title, description, article
title = "Food Classifier Mini"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food."
article = "Link to my GitHub"