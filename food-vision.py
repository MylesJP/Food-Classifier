# Using Food-101 dataset
# Starting out with smaller version with 3 classes

import torch
import os

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        ...
