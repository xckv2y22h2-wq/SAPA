"""
Dataset builder for ImageNet-100 from parquet files.

References:
    https://huggingface.co/datasets/imagenet-1k/blob/main/imagenet-1k.py
"""

import os
from pathlib import Path
from typing import List, Any, Tuple, Callable, Optional
import glob
import datasets
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io
import numpy as np
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

try:
    from .classes import IMAGENET100_CLASSES
except ImportError:
    from classes import IMAGENET100_CLASSES


_CITATION = """\
@inproceedings{tian2020contrastive,
  title={Contrastive multiview coding},
  author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XI 16},
  pages={776--794},
  year={2020},
  organization={Springer}
}
"""

_HOMEPAGE = "https://github.com/HobbitLong/CMC"

_DESCRIPTION = """\
ImageNet-100 is a subset of ImageNet with 100 classes randomly selected from the original ImageNet-1k dataset.
"""


class Imagenet100(VisionDataset):
    """ImageNet-100 dataset for loading from parquet files."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template: str = "A photo of {}"
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self._split = verify_str_arg(split, "split", ("train", "validation", "val", "test"))
        self._base_folder = Path(self.root) / "imagenet-100"
        self._images_folder = self._base_folder / "data"  # parquet files are stored in 'data' folder
        
        if download:
            self._download()
        
        if not self._check_exists():
            raise RuntimeError(f"Dataset not found at {self._images_folder}. You can use download=True to download it")
        
        # Load the dataset from parquet files
        self._load_dataset()
        
        # Setup classes and mappings
        self.classes = list(IMAGENET100_CLASSES.values())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Setup prompt template for CLIP
        self.prompt_template = prompt_template
        self.clip_prompts = [
            prompt_template.format(label.lower().replace('_', ' ').replace('-', ' '))
            for label in self.classes
        ]
    
    def _load_dataset(self):
        # Define features
        features = datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=list(IMAGENET100_CLASSES.values()))
        })
        
        # Determine which files to load based on split
        if self._split == "train":
            pattern = str(self._images_folder / "train-*.parquet")
        elif self._split in ["validation", "val", "test"]:
            pattern = str(self._images_folder / "validation-*.parquet")
        else:
            raise ValueError(f"Unknown split: {self._split}")
        
        # Check if parquet files exist
        parquet_files = glob.glob(pattern)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found matching pattern: {pattern}")
        
        # Load dataset using HuggingFace datasets
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=pattern,
            split="train",  # datasets always uses "train" for single split
            features=features
        )
        
        # Cache the dataset length
        self._length = len(self.dataset)
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._length}")
        
        # Get item from HuggingFace dataset
        item = self.dataset[idx]
        
        # Extract and process image
        image = item['image']
        
        # Handle different image formats
        if isinstance(image, dict) and 'bytes' in image:
            # Image stored as {'bytes': b'...', 'path': '...'}
            image = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, bytes):
            # Direct bytes
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            # Numpy array
            image = Image.fromarray(image.astype('uint8'))
        elif not isinstance(image, Image.Image):
            # Try to convert to PIL Image
            try:
                image = Image.open(image) if isinstance(image, str) else image
            except:
                raise ValueError(f"Cannot convert image of type {type(image)} to PIL Image")
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract label
        label = item['label']
        
        # Convert label to index if it's a string
        if isinstance(label, str):
            if label in self.class_to_idx:
                label = self.class_to_idx[label]
            elif label in IMAGENET100_CLASSES:
                # If label is a synset ID, convert to class name first
                label = self.class_to_idx[IMAGENET100_CLASSES[label]]
            else:
                raise ValueError(f"Unknown label: {label}")
        elif not isinstance(label, (int, np.integer)):
            raise ValueError(f"Unexpected label type: {type(label)}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def extra_repr(self) -> str:
        return f"split={self._split}"
    
    def _check_exists(self) -> bool:
        if not self._images_folder.exists():
            return False
        
        # Check for parquet files
        train_files = list(self._images_folder.glob("train-*.parquet"))
        val_files = list(self._images_folder.glob("validation-*.parquet"))
        
        return len(train_files) > 0 or len(val_files) > 0
    
    def _download(self) -> None:
        if self._check_exists():
            return
        
        # Create directories if they don't exist
        self._images_folder.mkdir(parents=True, exist_ok=True)
        
        # Note: Actual download implementation would go here
        # For now, just print instructions
        print(f"Dataset not found at {self._images_folder}")
        print("To download ImageNet-100, you can:")
        print("1. Clone from HuggingFace: git lfs clone https://huggingface.co/datasets/clane9/imagenet-100")
        print("2. Or download manually and place parquet files in:", self._images_folder)