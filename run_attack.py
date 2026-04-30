#!/usr/bin/env python3
'''
This script evaluates the adversarial robustness of different CLIP models against various attacks.
It supports evaluating the original CLIP model and several fine-tuned models (TeCoA, PMG, FARE, TRADES).
The script allows for the following attacks:
- PGD
- CW
- AutoAttack
- SAPA (Semantic Aware Perturbation Attack)
'''

import argparse
from email import parser
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import time
import datetime
import warnings
import copy
import matplotlib.pyplot as plt
import json
from pathlib import Path
# Add to existing imports
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

# import get_dataset and model_loader from replace.tv_datasets and replace.model_loader
from replace.tv_datasets.dataset_ops import get_dataset
from replace.model_loader import (
    load_model, load_clip, load_tecoa, load_fare, load_pmg, 
    load_trades, load_audience, load_mobileclip, load_mobileclip2,
    get_text_tokens
)

from modified_clip.model_compatibility import CLIPCompatibilityWrapper
from sapa.semantic_wordnet_anchor import WordNetSemanticAnchor
from sapa.semantic_feature_perturbation import SemanticFeatureSpacePerturbation
from sapa.semantic_feature_perturbation_multilayer import SemanticFeatureSpacePerturbationMultiLayer
from sapa.semantic_feature_perturbation_adaptive import SemanticFeatureSpacePerturbationAdaptive
from sapa.llava_text_adaption import LLaVATextAdaptation
from sapa.adversarial_text_generation import MultiModalSemanticAttack


try:
    from paper.cross_model_sta_integrated import CrossModelSTACalculator, add_cross_model_sta_to_attack_output
    CROSS_MODEL_STA_AVAILABLE = True
except ImportError:
    CROSS_MODEL_STA_AVAILABLE = False
    print("Note: Cross-model STA evaluation not available. Install with: pip install open_clip_torch")

# Make sure these imports are correct for your setup
try:
    from attacks import (attack_pgd,
                         attack_CW,
                         attack_pgd_targeted_semantic,
                         attack_CW_targeted_semantic,
                         attack_auto_new,
                         attack_semantic_flow,
                         attack_semantic_flow_enhanced,
                         attack_semantic_flow_robust_enhanced,
                         SemanticCoherenceMetrics,
                         SemanticControlEvaluator,
                         compare_attacks_semantic_control
                         )
    from models.model import multiGPU_CLIP, multiGPU_CLIP_Text_Prompt_Tuning, clip_img_preprocessing
    from modified_clip import clip
    import open_clip 
    from mobileclip.modules.common.mobileone import reparameterize_model
    from utils import accuracy, AverageMeter, save_checkpoint, refine_classname
    from models.prompters import TokenPrompter, NullPrompter, PromptLearner, NonePrompter
    
    # Import custom datasets from replace.tv_datasets
    from replace.tv_datasets import StanfordCars, Food101, SUN397, EuroSAT, DTD, Imagenet100, \
        Caltech101, Caltech256, Country211, Flowers102, PCAM, FGVCAircraft, OxfordIIITPet
    
    # Import standard datasets from torchvision
    from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the required modules are available in your Python path.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate adversarial robustness of different CLIP models')

    # Model options
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (.pth.tar file). Not needed for CLIP.')
    parser.add_argument('--model_type', type=str, default='CLIP',
                        choices=['CLIP', 'TeCoA', 'PMG', 'FARE', 'TRADES', 'AUDIENCE','MobileCLIP','MobileCLIP2'],
                        help='Type of model to evaluate')
    parser.add_argument('--arch', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101',
                                # MobileCLIP architecture (will use OpenCLIP)
                                'MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP-S2',
                                'MobileCLIP-B', 'MobileCLIP-BLT', 'MobileCLIP-L-14',
                                 # MobileCLIP2 architectures (will use OpenCLIP)
                                'MobileCLIP2-S0', 'MobileCLIP2-S2', 'MobileCLIP2-B', 
                                'MobileCLIP2-S3', 'MobileCLIP2-L-14', 'MobileCLIP2-S4'
                                ],
                        help='CLIP , MobileCLIP or MobileCLIP2 architecture')
    
    # Attack options

    # In parse_args() function, change this line:
    parser.add_argument('--attack', type=str, default='pgd',
                    choices=['pgd', 'CW', 'AA', 'clean', 'sapa', 'targeted_pgd', 'targeted_cw'],
                    help='Attack method (targeted_pgd and targeted_cw are Priority 1A baselines)')
    parser.add_argument('--norm', type=str, default='l_inf',
                        choices=['l_inf', 'l_2'],
                        help='Attack norm')
    parser.add_argument('--epsilon', type=float, default=8/255.,
                        help='Attack radius')
    parser.add_argument('--attack_iters', type=int, default=20,
                        help='Number of attack iterations')
    parser.add_argument('--stepsize', type=float, default=0.01,
                        help='Attack step size') 
    parser.add_argument('--restarts', type=int, default=1,
                        help='Number of attack restarts')
    parser.add_argument('--tta_frequency', type=int, default=10,
                        help='LLaVA TTA update frequency (SAPA only)')
    parser.add_argument('--sapa_variant', type=str, default=None,
                        help='SAPA variant (multilayer, adaptive, or ablation variants)')
    parser.add_argument('--layer_weights', type=str, default=None,
                        help='Layer weights for adaptive SAPA: L3,L6,L9,Final (e.g., 0.15,0.2,0.25,0.4)')
    parser.add_argument('--adaptive_weights', action='store_true',
                        help='Enable dynamic weight adjustment based on attack progress (adaptive SAPA)')
    parser.add_argument('--adv_text_weight', type=float, default=0.5,
                        help='Weight for adversarial text generation (0=WordNet only, 1=AdvText only)')
    parser.add_argument('--use_llava_text', action='store_true',
                        help='Use LLaVA for adversarial text generation (else use templates)')

    # Priority 1B: Cross-model STA evaluation
    parser.add_argument('--compute_cross_model_sta', action='store_true',
                        help='Compute cross-model STA using independent evaluation model (Priority 1B)')
    parser.add_argument('--eval_model', type=str, default='openclip_vit_l',
                        choices=['openclip_vit_l', 'openclip_vit_h', 'siglip'],
                        help='Independent model for cross-model STA evaluation')

    # Text perturbation step size (used by SAPA)
    parser.add_argument('--text_perb_stepsize', type=float, default=0.01,
                        help='Step size for text perturbation')
    
    # Advanced attack parameters
    parser.add_argument('--adaptive_weight', type=float, default=0.2,
                        help='Weight for adaptive progress term')
    parser.add_argument('--image_weight', type=float, default=0.6,
                        help='Weight for image modality in cross-modal attack')
    parser.add_argument('--text_weight', type=float, default=0.4,
                        help='Weight for text modality in cross-modal attack')
    
    # Multi-model evaluation
    parser.add_argument('--model_list', type=str, nargs='+', default=None,
                        help='List of model paths for dual-target attack')
    parser.add_argument('--model_types', type=str, nargs='+', default=None,
                        help='List of model types corresponding to model_list')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'ImageNet', 'STL10', 'SUN397', 
                             'StanfordCars', 'Food101', 'OxfordPets', 'Flowers102',
                             'DTD', 'EuroSAT', 'FGVC', 'PCAM', 'Caltech101', 'Caltech256',
                             'Country211', 'FGVCAircraft'],
                    help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./datasets',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_test_samples', type=int, default=None,
                        help='Limit the number of test samples to use (useful for quick testing)')
    parser.add_argument('--balanced_sampling', action='store_true', default=True,
                        help='Use balanced sampling across classes (default: True)')
    
    # Prompt learning options
    parser.add_argument('--ctx', type=int, default=16,
                        help='Number of context vectors for prompt learner')
    parser.add_argument('--ctx_init', type=str, default='This is a photo of a',
                        help='Initial context string for prompt learner')
    parser.add_argument('--position', type=str, default='end',
                        help='Context prompt position: end|middle|front')
    
    # Misc options
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize adversarial examples')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare multiple models on the same dataset/attack')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output during evaluation')
    
    # Add these new arguments:
    
    parser.add_argument('--skip_clean_eval', action='store_true',
                        help='Skip clean accuracy evaluation (use cached value)')
    parser.add_argument('--cached_clean_accuracy', type=float, default=None,
                        help='Pre-computed clean accuracy to use in results')
    parser.add_argument('--result_filename', type=str, default=None,
                        help='Custom filename for results (default: {model_type}_{attack}_results.txt)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output (semantic targeting details, embeddings, etc.)')
    
    # ADD SAPA-SPECIFIC PARAMETERS
    parser.add_argument('--embedding_loss_weight', type=float, default=0.3,
                        help='Weight for embedding space loss in SAPA')
    parser.add_argument('--semantic_anchor_weight', type=float, default=0.2,
                        help='Weight for semantic anchor manipulation in SAPA')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.1,
                        help='Weight for contrastive learning inversion in SAPA')
    parser.add_argument('--adversarial_loss_weight', type=float, default=1.0,
                        help='Weight for adversarial loss (push away from true class) in SAPA')
    parser.add_argument('--semantic_alignment_weight', type=float, default=2.0,
                        help='Weight for semantic alignment loss (pull toward target) in SAPA')
    parser.add_argument('--semantic_strategy', type=str, default='similar',
                        choices=['similar', 'related', 'distant'],
                        help='Semantic anchor selection strategy: similar (default), related, or distant (hypernyms)')

     # Add these new arguments for semantic control evaluation
    parser.add_argument('--evaluate_semantic_control', action='store_true',
                       help='Evaluate semantic control capabilities using SCI, SDS, SPI metrics')
    parser.add_argument('--semantic_control_batches', type=int, default=5,
                       help='Number of batches for semantic control evaluation')
    parser.add_argument('--compare_semantic_attacks', action='store_true',
                       help='Compare semantic control across different attacks')
    
    return parser.parse_args()

''' import from dataset_ops.py 
def get_dataset(args):
    """Load the specified dataset with appropriate transforms"""
    print(f"Loading dataset: {args.dataset} from {args.data_path}")
    
    # Define standard transforms for different image sizes
    transform_32 = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Determine correct image size based on architecture
    img_size = 224  # Default for most models
    
    if args.arch.startswith('MobileCLIP'):
        # MobileCLIP-S0,S1,S2,S3, S4 use 256x256 
        # MobileCLIP-B, L14 use 224x224
        # MobileCLIP2-S0, S2, S3, S4 use 256x256
        # MobileCLIP2-B, L14 use 224x224
        if any(x in args.arch for x in ['S0', 'S1', 'S2', 'S3', 'S4']):
            img_size = 256
        else:  # B, L14
            img_size = 224
    
    print(f"Using image size: {img_size}×{img_size} for {args.arch}")
    
    # Create transform based on determined size
    transform_224 = transforms.Compose([
        transforms.Resize(img_size + 32),  # Resize to slightly larger
        transforms.CenterCrop(img_size),   # Crop to exact size
        transforms.ToTensor(),
    ])
    
    transform_resize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Helper function to create train/test split if needed
    def create_train_test_split(dataset, train_ratio=0.8, shuffle=True):
        from torch.utils.data import Subset
        
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        
        if shuffle:
            indices = torch.randperm(dataset_size).tolist()
        else:
            indices = list(range(dataset_size))
            
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size+test_size]
        
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Make sure class names are accessible
        if hasattr(dataset, 'classes'):
            train_dataset.classes = dataset.classes
            test_dataset.classes = dataset.classes
        
        return train_dataset, test_dataset
    
    # Dictionary to track which transform to use for each dataset
    transform_map = {
        'cifar10': transform_32,
        'cifar100': transform_32,
        'STL10': transform_32,
        'ImageNet': transform_224,
        'Caltech101': transform_224,
        'OxfordPets': transform_224,
        'Flowers102': transform_224,
        'Food101': transform_224,
        'DTD': transform_224,
        'EuroSAT': transform_224,
        'StanfordCars': transform_224,
        'SUN397': transform_224,
        'FGVC': transform_224,
        'PCAM': transform_224,
        'Caltech256': transform_224,
        'Country211': transform_224,
        'FGVCAircraft': transform_224,
    }
    
    # Make sure we're using the right transform
    transform = transform_map.get(args.dataset, transform_224)
    
    # Load dataset based on name
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
        class_names = train_dataset.classes

    elif args.dataset == 'STL10':
        train_dataset = STL10(root=args.data_path, split='train', download=True, transform=transform)
        test_dataset = STL10(root=args.data_path, split='test', download=True, transform=transform)
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        
    elif args.dataset == 'Caltech101':
        try:
            # Using custom dataset from replace.tv_datasets
            dataset = Caltech101(root=args.data_path, target_type='category', transform=transform_resize, download=True)
            train_dataset, test_dataset = create_train_test_split(dataset)
            # class_names = dataset.categories if hasattr(dataset, 'categories') else dataset.classes
            class_names = dataset.classes
        except Exception as e:
            print(f"Error loading Caltech101: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'caltech101'), transform=transform_resize)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading Caltech101 fallback: {e2}")
                raise ValueError(f"Could not load Caltech101: {e}, {e2}")
        
    elif args.dataset == 'OxfordPets':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = OxfordIIITPet(root=args.data_path, split='trainval', transform=transform, download=True)
            test_dataset = OxfordIIITPet(root=args.data_path, split='test', transform=transform, download=True)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading OxfordIIITPet: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'oxford-iiit-pet'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading OxfordPets fallback: {e2}")
                raise ValueError(f"Could not load OxfordPets: {e}, {e2}")
        
    elif args.dataset == 'Flowers102':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = Flowers102(root=args.data_path, split='train', download=True, transform=transform)
            val_dataset = Flowers102(root=args.data_path, split='val', download=True, transform=transform)
            test_dataset = Flowers102(root=args.data_path, split='test', download=True, transform=transform)
            
            # Combine train and val for training
            train_dataset = ConcatDataset([train_dataset, val_dataset])
            
            # Get class names
            if hasattr(test_dataset, '_classes'): 
                class_names = test_dataset._classes
            else:
                class_names = [f'flower_{i+1}' for i in range(102)]
                
        except Exception as e:
            print(f"Error loading Flowers102: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'flowers102'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading Flowers102 fallback: {e2}")
                raise ValueError(f"Could not load Flowers102: {e}, {e2}")
        
    elif args.dataset == 'Food101':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = Food101(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = Food101(root=args.data_path, split='test', download=True, transform=transform)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading Food101: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'food-101'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading Food101 fallback: {e2}")
                raise ValueError(f"Could not load Food101: {e}, {e2}")
        
    elif args.dataset == 'DTD':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = DTD(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = DTD(root=args.data_path, split='test', download=True, transform=transform)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading DTD: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'dtd', 'images'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading DTD fallback: {e2}")
                raise ValueError(f"Could not load DTD: {e}, {e2}")
        
    elif args.dataset == 'EuroSAT':
        try:
            # Using custom dataset from replace.tv_datasets
            dataset = EuroSAT(root=args.data_path, download=True, transform=transform)
            train_dataset, test_dataset = create_train_test_split(dataset)
            class_names = dataset.classes if hasattr(dataset, 'classes') else [
                'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
                'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
            ]
        except Exception as e:
            print(f"Error loading EuroSAT: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'eurosat'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading EuroSAT fallback: {e2}")
                raise ValueError(f"Could not load EuroSAT: {e}, {e2}")
    
    elif args.dataset == 'StanfordCars':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = StanfordCars(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = StanfordCars(root=args.data_path, split='test', download=True, transform=transform)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading StanfordCars: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'stanford_cars'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading StanfordCars fallback: {e2}")
                raise ValueError(f"Could not load StanfordCars: {e}, {e2}")
    
    elif args.dataset == 'Country211':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = Country211(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = Country211(root=args.data_path, split='test', download=True, transform=transform)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading Country211: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'country211'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading Country211 fallback: {e2}")
                raise ValueError(f"Could not load Country211: {e}, {e2}")
                
    elif args.dataset == 'FGVCAircraft':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = FGVCAircraft(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = FGVCAircraft(root=args.data_path, split='test', download=True, transform=transform)
            class_names = train_dataset.classes
        except Exception as e:
            print(f"Error loading FGVCAircraft: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'fgvc-aircraft'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading FGVCAircraft fallback: {e2}")
                raise ValueError(f"Could not load FGVCAircraft: {e}, {e2}")
                
    elif args.dataset == 'PCAM':
        try:
            # Using custom dataset from replace.tv_datasets
            train_dataset = PCAM(root=args.data_path, split='train', download=True, transform=transform)
            test_dataset = PCAM(root=args.data_path, split='test', download=True, transform=transform)
            class_names = ['normal', 'tumor']  # Binary classification
        except Exception as e:
            print(f"Error loading PCAM: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'pcam'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading PCAM fallback: {e2}")
                raise ValueError(f"Could not load PCAM: {e}, {e2}")
                
    elif args.dataset == 'ImageNet':
        try:
            # Try to load directly - ImageNet 100 
            # Import the ImageNet-100 dataset class         
            # Create train and test datasets using the custom ImageNet-100 loader
            train_dataset = Imagenet100(
                root=args.data_path, 
                split='train',
                transform=transform,
                download=True  # Will check if data exists
            )
            test_dataset = Imagenet100(
                root=args.data_path, 
                split='validation',  # or 'val'
                transform=transform,
                download=True
            )
            # Get class names from the dataset
            class_names = train_dataset.classes
                
        except Exception as e:
            print(f"Error loading ImageNet from standard directory structure: {e}")
            # Try alternative paths
            try:
                train_dataset = ImageFolder(os.path.join(args.data_path, 'ILSVRC/Data/CLS-LOC/train'), transform=transform)
                test_dataset = ImageFolder(os.path.join(args.data_path, 'ILSVRC/Data/CLS-LOC/val'), transform=transform)
                class_names = test_dataset.classes
            except Exception as e2:
                print(f"Error loading ImageNet from alternative paths: {e2}")
                raise ValueError(f"Could not load ImageNet: {e}, {e2}")
                
    elif args.dataset == 'SUN397':
        try:
            # Using custom dataset from replace.tv_datasets
            dataset = SUN397(root=args.data_path, download=True, transform=transform)
            train_dataset, test_dataset = create_train_test_split(dataset)
            class_names = dataset.classes
        except Exception as e:
            print(f"Error loading SUN397: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'SUN397'), transform=transform)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading SUN397 fallback: {e2}")
                raise ValueError(f"Could not load SUN397: {e}, {e2}")
    
    elif args.dataset == 'Caltech256':
        try:
            # Using custom dataset from replace.tv_datasets
            dataset = Caltech256(root=args.data_path, transform=transform_resize, download=True)
            train_dataset, test_dataset = create_train_test_split(dataset)
            class_names = dataset.classes
        except Exception as e:
            print(f"Error loading Caltech256: {e}")
            # Fallback
            try:
                dataset = ImageFolder(os.path.join(args.data_path, 'caltech256'), transform=transform_resize)
                train_dataset, test_dataset = create_train_test_split(dataset)
                class_names = dataset.classes
            except Exception as e2:
                print(f"Error loading Caltech256 fallback: {e2}")
                raise ValueError(f"Could not load Caltech256: {e}, {e2}")
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Handle class names for special cases
    if isinstance(train_dataset, Subset) and not hasattr(train_dataset, 'classes') and hasattr(train_dataset.dataset, 'classes'):
        train_dataset.classes = train_dataset.dataset.classes
        test_dataset.classes = test_dataset.dataset.classes
        
    # Ensure all datasets have accessible class names
    if not isinstance(class_names, list):
        class_names = list(class_names)
    
    # Limit test samples if specified
    if args.num_test_samples is not None:
        if args.balanced_sampling:
            # Balanced sampling: equal samples per class
            print(f"Using balanced sampling across {len(class_names)} classes...")
            
            # Get labels for all samples
            if isinstance(test_dataset, Subset):
                base_dataset = test_dataset.dataset
                base_indices = test_dataset.indices
            else:
                base_dataset = test_dataset
                base_indices = list(range(len(test_dataset)))
            
            # Group indices by class
            from collections import defaultdict
            class_indices = defaultdict(list)
            for idx in base_indices:
                if hasattr(base_dataset, 'targets'):
                    label = base_dataset.targets[idx]
                elif hasattr(base_dataset, '_labels'):
                    label = base_dataset._labels[idx]
                else:
                    # Fallback: access via __getitem__
                    _, label = base_dataset[idx]
                    if hasattr(label, 'item'):
                        label = label.item()
                class_indices[label].append(idx)
            
            # Calculate samples per class
            num_classes = len(class_indices)
            samples_per_class = max(1, args.num_test_samples // num_classes)
            
            # Select balanced indices
            import random
            random.seed(args.seed)
            balanced_indices = []
            for label, indices in sorted(class_indices.items()):
                if len(indices) >= samples_per_class:
                    selected = random.sample(indices, samples_per_class)
                else:
                    selected = indices  # Use all if not enough
                balanced_indices.extend(selected)
            
            # Shuffle to mix classes
            random.shuffle(balanced_indices)
            
            # Create subset with balanced indices
            if isinstance(test_dataset, Subset):
                test_dataset = Subset(base_dataset, balanced_indices)
            else:
                test_dataset = Subset(test_dataset, balanced_indices)
            test_dataset.classes = train_dataset.classes
            
            print(f"Balanced sampling: {samples_per_class} samples/class × {num_classes} classes = {len(balanced_indices)} total")
        else:
            # Original sequential sampling
            if isinstance(test_dataset, Subset):
                indices = list(range(min(args.num_test_samples, len(test_dataset))))
                test_dataset = Subset(test_dataset.dataset, 
                                      [test_dataset.indices[i] for i in indices])
                test_dataset.classes = train_dataset.classes
            else:
                indices = list(range(min(args.num_test_samples, len(test_dataset))))
                test_dataset = Subset(test_dataset, indices)
                test_dataset.classes = train_dataset.classes
            
            print(f"Limited test dataset to {len(test_dataset)} samples")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    # Refine class names for CLIP
    original_class_names = class_names.copy()
    class_names = refine_classname(class_names)
    
    print(f"Dataset loaded: {args.dataset}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, class_names, original_class_names
'''
def get_text_tokens(texts, model, device):
    if hasattr(model, 'is_openclip') and model.is_openclip:
        # Use MobileCLIP's specific tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            print("Using MobileCLIP-specific tokenizer")
            tokens = model.tokenizer(texts)
            # Handle different return formats
            if isinstance(tokens, dict):
                print("Tokenizer returned a dict, extracting 'input_ids'")
                tokens = tokens['input_ids']
            return tokens.to(device)
        else:
            # Fallback
            print("Warning: Model marked as OpenCLIP but no tokenizer found, using open_clip.tokenize")
            return open_clip.tokenize(texts).to(device)
    else:
        print("Using modified CLIP tokenizer")
        return clip.tokenize(texts).to(device)

''' use load_target_model 
def load_model(args, class_names):
    """Load model checkpoint"""
    print(f"=> Loading {args.model_type} model: {args.arch}")
    
        # Determine if we need OpenCLIP (for MobileCLIP)
    use_openclip = args.arch.startswith('MobileCLIP')
    
    if use_openclip:
        print("Using OpenCLIP to load MobileCLIP...")
        
        # CRITICAL: Set image_mean and image_std for MobileCLIP2
        model_kwargs = {}
        if not (args.arch.endswith("S3") or args.arch.endswith("S4") or args.arch.endswith("L-14")):
            # For S0, S1, S2, B: disable internal normalization
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
            print(f"Using custom normalization for {args.arch}: mean=(0,0,0), std=(1,1,1)")

        # Map architecture name
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.arch,
            pretrained=f'{args.model_path}/{args.arch.lower().replace("-", "_")}.pt',
            device=device,
            **model_kwargs # Pass normalization kwargs if needed
        )
        
        # IMPORTANT: Reparameterize MobileCLIP for inference
        model.eval()
        model = reparameterize_model(model)    
        # Wrap for compatibility
        model = CLIPCompatibilityWrapper(model, source='openclip')
        source = 'openclip'
        model.is_openclip = True  # Mark as OpenCLIP model
        # MobileCLIP uses a specific tokenizer 
        tokenizer = open_clip.get_tokenizer(args.arch)
        model.tokenizer = tokenizer  # Store it in model for easy access
        model.arch = args.arch  # Store architecture name, to be used by clip_img_preprocessing
        
    else:
        print("Using modified CLIP...")
        model, preprocess = clip.load(args.arch, device=device, jit=False) 
        # Wrap for consistency (even though modified CLIP already has ind_prompt)
        model = CLIPCompatibilityWrapper(model, source='modified_clip')
        source = 'modified_clip'
        model.is_openclip = False  # Mark as not OpenCLIP model
        model.tokenizer = None 
        model.arch = args.arch  # Store architecture name 
        model.preprocess = preprocess 
    
    
    # Convert model to FP32
    model.float()
    
    # Create prompters
    prompter = NullPrompter().to(device)
    add_prompter = TokenPrompter(0).to(device)  # No token prompting for evaluation
    prompt_learner = None
    
    # If evaluating a fine-tuned model
    if args.model_type != 'CLIP' and not use_openclip:
        # Add weights_only=False to handle the unpickling error
        # checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        try:
            # Try loading with weights_only=False
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            print(f"Successfully loaded checkpoint with weights_only=False")
        except Exception as e1:
            print(f"Failed to load with weights_only=False: {e1}")
            try:
                # Try loading with weights_only=True
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
                print(f"Successfully loaded checkpoint with weights_only=True")
            except Exception as e2:
                print(f"Failed to load with weights_only=True: {e2}")
                # Create an empty checkpoint to continue
                print("Using empty checkpoint to continue evaluation")
                checkpoint = {}
        # Load vision encoder weights
        if 'vision_encoder_state_dict' in checkpoint:
            model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            print(f"Loaded vision encoder weights from checkpoint")
        else:
            print("Warning: No vision encoder weights found in checkpoint")
        
        # Load prompter weights if available
        if 'state_dict' in checkpoint and hasattr(prompter, 'load_state_dict'):
            prompter.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded prompter weights from checkpoint")
        
        # Load add_prompter weights if available
        if 'add_prompter' in checkpoint and hasattr(add_prompter, 'load_state_dict'):
            add_prompter.load_state_dict(checkpoint['add_prompter'])
            print(f"Loaded add_prompter weights from checkpoint")
    
    # For AUDIENCE model, we need to load the prompt learner
    if args.model_type == 'AUDIENCE' and  not use_openclip:
        # Create an instance of the PromptLearner for AUDIENCE model
        # This needs to match how the model was trained
        template = 'This is a photo of a {}'
        texts = [template.format(label) for label in class_names]
        
        prompt_learner = PromptLearner(args, class_names, model).to(device)
        
        # Load prompt learner weights if available
        if 'prompt_learner' in checkpoint and hasattr(prompt_learner, 'load_state_dict'):
            prompt_learner.load_state_dict(checkpoint['prompt_learner'])
            print(f"Loaded prompt learner weights from checkpoint")
        else:
            print("Warning: No prompt learner weights found in checkpoint, using initialized weights")
    
    return model, prompter, add_prompter, prompt_learner
'''

def load_target_model(args, device, class_names=None):
    model_type = args.model_type.upper()
    
    print(f"\nLoading {args.model_type} model...")
    
    # Load model based on type
    if model_type == 'CLIP':
        model, preprocess, prompter, add_prompter, prompt_learner = load_clip(
            arch=args.arch,
            device=device
        )
    elif model_type == 'TECOA':
        model, preprocess, prompter, add_prompter, prompt_learner = load_tecoa(
            arch=args.arch,
            device=device,
            model_dir=args.model_path
        )
    elif model_type == 'FARE':
        model, preprocess, prompter, add_prompter, prompt_learner = load_fare(
            arch=args.arch,
            device=device,
            model_dir=args.model_path 
        )
    elif model_type == 'PMG':
        model, preprocess, prompter, add_prompter, prompt_learner = load_pmg(
            arch=args.arch,
            device=device,
            model_dir=args.model_path 
        )
    elif model_type == 'TRADES':
        model, preprocess, prompter, add_prompter, prompt_learner = load_trades(
            arch=args.arch,
            device=device,
            model_dir=args.model_path 
        )
    elif model_type == 'AUDIENCE':
        if class_names is None:
            raise ValueError("AUDIENCE model requires class_names. Load dataset first.")
        model, preprocess, prompter, add_prompter, prompt_learner = load_audience(
            arch=args.arch,
            device=device,
            model_dir=args.model_path,
            class_names=class_names
        )
    elif model_type == 'MOBILECLIP':
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip(
            arch=args.arch,
            device=device,
            mobileclip_dir=args.mobileclip_dir
        )
    elif model_type == 'MOBILECLIP2':
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip2(
            arch=args.arch,
            device=device,
            mobileclip_dir=args.mobileclip_dir
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"✓ Loaded {args.model_type} ({args.arch})")
    print(f"  Is OpenCLIP: {getattr(model, 'is_openclip', False)}")
    print(f"  Has PromptLearner: {prompt_learner is not None}")
    
    return model, preprocess, prompter, add_prompter, prompt_learner


def visualize_examples(clean_images, adv_images, true_labels, pred_clean, pred_adv, class_names, args):
    num_examples = min(5, clean_images.size(0))
    
    plt.figure(figsize=(15, 3*num_examples))
    for i in range(num_examples):
        # Clean image
        plt.subplot(num_examples, 3, i*3 + 1)
        clean_img = clean_images[i].detach().cpu().permute(1, 2, 0).numpy()
        clean_img = np.clip(clean_img, 0, 1)
        plt.imshow(clean_img)
        plt.title(f"Clean: {class_names[true_labels[i]]}\nPred: {class_names[pred_clean[i]]}")
        plt.axis('off')
        
        # Adversarial image
        plt.subplot(num_examples, 3, i*3 + 2)
        adv_img = adv_images[i].detach().cpu().permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0, 1)
        plt.imshow(adv_img)
        plt.title(f"Adv: {class_names[true_labels[i]]}\nPred: {class_names[pred_adv[i]]}")
        plt.axis('off')
        
        # Difference (magnified)
        plt.subplot(num_examples, 3, i*3 + 3)
        diff = (adv_img - clean_img)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)  # Normalize to [0,1]
        plt.imshow(diff)
        plt.title(f"Difference (Magnified)")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_{args.attack}_examples.png"))
    plt.close()

    # Check if results file already exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Use custom filename if provided, otherwise default naming
    # Format: {model}_{dataset}_{attack}_eps{epsilon}.txt
    if hasattr(args, 'result_filename') and args.result_filename:
        results_file = os.path.join(args.output_dir, args.result_filename)
    else:
        epsilon_float = args.epsilon / 255.0 if args.epsilon > 1 else args.epsilon
        results_file = os.path.join(args.output_dir, f"{args.model_type}_{args.dataset}_{args.attack}_eps{epsilon_float:.4f}.txt")
    
    if os.path.exists(results_file):
        print(f"Results file found: {results_file}")
        print("Loading existing results...")
        
        try:
            # Parse existing results
            clean_acc = None
            adv_acc = None
            
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Clean accuracy:"):
                        clean_acc = float(line.split(":")[1].strip().replace("%", ""))
                    elif line.startswith("Adversarial accuracy:"):
                        adv_acc = float(line.split(":")[1].strip().replace("%", ""))
            
            if clean_acc is not None and adv_acc is not None:
                print(f"\nLoaded Results for {args.model_type} model with {args.attack} attack:")
                print(f"Clean accuracy: {clean_acc:.2f}%")
                print(f"Adversarial accuracy: {adv_acc:.2f}%")
                print(f"Robustness gap: {clean_acc - adv_acc:.2f}%")
                print("Evaluation bypassed - using cached results.")
                
                return clean_acc, adv_acc
            else:
                print("Warning: Could not parse accuracy values from existing file. Re-running evaluation.")
                
        except Exception as e:
            print(f"Warning: Error reading existing results file: {e}. Re-running evaluation.")
    
    # If we reach here, either no results file exists or there was an error reading it
    print("Running evaluation...")

    # Check for existing results and load if available (skip re-evaluation)
    # Use custom filename if provided, otherwise default naming
    # Format: {model}_{dataset}_{attack}_eps{epsilon}.txt
    if hasattr(args, 'result_filename') and args.result_filename:
        results_file = os.path.join(args.output_dir, args.result_filename)
    else:
        epsilon_float = args.epsilon / 255.0 if args.epsilon > 1 else args.epsilon
        results_file = os.path.join(args.output_dir, f"{args.model_type}_{args.dataset}_{args.attack}_eps{epsilon_float:.4f}.txt")

    if os.path.exists(results_file):
        print(f"Results file found: {results_file}")
        print("Loading existing results...")

        try:
            # Parse existing results
            clean_acc = None
            adv_acc = None

            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Clean accuracy:"):
                        clean_acc = float(line.split(":")[1].strip().replace("%", ""))
                    elif line.startswith("Adversarial accuracy:"):
                        adv_acc = float(line.split(":")[1].strip().replace("%", ""))

            if clean_acc is not None and adv_acc is not None:
                print(f"\nLoaded Results for {args.model_type} model with {args.attack} attack:")
                print(f"Clean accuracy: {clean_acc:.2f}%")
                print(f"Adversarial accuracy: {adv_acc:.2f}%")
                print(f"Robustness gap: {clean_acc - adv_acc:.2f}%")
                print("Evaluation bypassed - using cached results.")

                return clean_acc, adv_acc
            else:
                print("Warning: Could not parse accuracy values from existing file. Re-running evaluation.")

        except Exception as e:
            print(f"Warning: Error reading existing results file: {e}. Re-running evaluation.")


def evaluate_robustness(test_loader, texts, model, prompter, add_prompter, prompt_learner, dataset_classes, original_dataset_classes, args):
    criterion = torch.nn.CrossEntropyLoss()
    
    # Tracking metrics
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    # Setup attack parameters
    if args.attack == 'pgd':
        attack_fn = attack_pgd
        attack_kwargs = {
            'prompter': prompter,
            'model': model,
            'add_prompter': add_prompter,
            'criterion': criterion,
            'alpha': args.stepsize,
            'attack_iters': args.attack_iters,
            'norm': args.norm,
            'restarts': args.restarts,
            'epsilon': args.epsilon,
            'debug': args.debug
        }
    elif args.attack == 'CW':
        attack_fn = attack_CW
        attack_kwargs = {
            'prompter': prompter,
            'model': model,
            'add_prompter': add_prompter,
            'criterion': criterion,
            'alpha': args.stepsize,
            'attack_iters': args.attack_iters,
            'norm': args.norm,
            'restarts': args.restarts,
            'epsilon': args.epsilon,
            'debug': args.debug
        }
    elif args.attack == 'AA':
        attack_fn = attack_auto_new
        # AA has different parameter order
        attack_kwargs = None
    elif args.attack == 'targeted_pgd':
        # Priority 1A: Targeted PGD toward semantic anchor (same as SAPA uses)
        attack_fn = attack_pgd_targeted_semantic
        attack_kwargs = {
            'prompter': prompter,
            'model': model,
            'add_prompter': add_prompter,
            'criterion': criterion,
            'alpha': args.stepsize,
            'attack_iters': args.attack_iters,
            'norm': args.norm,
            'restarts': args.restarts,
            'epsilon': args.epsilon,
            'semantic_target_class': None,  # Will be set per batch using WordNet
            'text_embeddings': None,  # Will be computed once
            'debug': args.debug
        }
    elif args.attack == 'targeted_cw':
        # Priority 1A: Targeted C&W toward semantic anchor (same as SAPA uses)
        attack_fn = attack_CW_targeted_semantic
        attack_kwargs = {
            'prompter': prompter,
            'model': model,
            'add_prompter': add_prompter,
            'criterion': criterion,
            'alpha': args.stepsize,
            'attack_iters': args.attack_iters,
            'norm': args.norm,
            'restarts': args.restarts,
            'epsilon': args.epsilon,
            'semantic_target_class': None,  # Will be set per batch using WordNet
            'text_embeddings': None,  # Will be computed once
            'kappa': 0,
            'debug': args.debug
        }
    elif args.attack == 'sapa': # obsoleted, using run_semantic_attack_comparison.py instead 
        # SAPA (Semantic-Aware Perturbation Attack)
        attack_fn = attack_semantic_flow_robust_enhanced
        attack_kwargs = {
            'prompter': prompter,
            'model': model,
            'add_prompter': add_prompter,
            'criterion': criterion,
            'alpha': args.stepsize,
            'attack_iters': args.attack_iters,
            'norm': args.norm,
            'epsilon': args.epsilon,
            # SAPA-specific parameters
            'dataset_classes': dataset_classes,
            'original_dataset_classes': dataset_classes,
            'semantic_strategy': args.semantic_strategy,
            'embedding_loss_weight': args.embedding_loss_weight,
            'semantic_anchor_weight': args.semantic_anchor_weight,
            'contrastive_loss_weight': args.contrastive_loss_weight,
            'dataset_name': args.dataset,
            'model_type': args.model_type
        }


    # Switch to evaluation mode
    model.eval()
    prompter.eval()
    add_prompter.eval()
    if prompt_learner is not None:
        prompt_learner.eval()
    
    # Tokenize text prompts
    text_tokens = clip.tokenize(texts).to(device)
    
    # For visualization
    if args.visualize:
        vis_images = None
        vis_adv_images = None
        vis_targets = None
        vis_clean_preds = None
        vis_adv_preds = None

    # For targeted attacks, initialize CLIP semantic anchor before the loop
    clip_anchor = None
    sta_values = []  # Track STA for targeted attacks (Priority 1A)
    if args.attack in ['targeted_pgd', 'targeted_cw']:
        print("Initializing CLIP Semantic Anchor for targeted attacks...")
        from sapa.semantic_clip_anchor import CLIPSemanticAnchor
        clip_anchor = CLIPSemanticAnchor(model, device, dataset_classes)

    # Track semantic targets for STA computation (Priority 1A)
    all_semantic_targets = []

    # Testing loop
    start_time = time.time()
    for i, (images, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images, target = images.to(device), target.to(device)
        batch_size = images.size(0)
        total += batch_size
        
        # Clean accuracy
        with torch.no_grad():
            processed_images = clip_img_preprocessing(images)
            prompted_images = prompter(processed_images)
            prompt_token = add_prompter()
            
            # For AUDIENCE model, use the prompt_learner
            if args.model_type == 'AUDIENCE' and prompt_learner is not None:
                clean_output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                clean_output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
                
            clean_preds = clean_output.argmax(dim=1)
            clean_correct += (clean_preds == target).sum().item()
        
        # Generate adversarial examples
        if args.attack == 'AA':
            # AA has a different interface
            adv_images = attack_fn(model, images, target, text_tokens, prompter, add_prompter, epsilon=args.epsilon)
        elif args.attack in ['targeted_pgd', 'targeted_cw']:
            # For targeted attacks: find semantic anchor for each sample in batch
            semantic_targets_batch = []
            for label_idx in target:
                label = label_idx.item()
                # Find semantic anchor (same as SAPA uses)
                anchor = clip_anchor.find_semantic_anchor(
                    label,
                    strategy='similar',  # Use same strategy as SAPA
                    similarity_range=(0.3, 0.85)
                )
                if anchor is not None:
                    # Convert class name to index
                    anchor_idx = dataset_classes.index(anchor)
                    semantic_targets_batch.append(anchor_idx)
                else:
                    # Fallback to random class if no anchor found
                    anchor = torch.randint(0, len(dataset_classes), (1,), device=device).item()
                    while anchor == label:
                        anchor = torch.randint(0, len(dataset_classes), (1,), device=device).item()
                    semantic_targets_batch.append(anchor)

            semantic_targets = torch.tensor(semantic_targets_batch, device=device)

            # Update attack_kwargs with semantic target for this batch
            attack_kwargs['X'] = images
            attack_kwargs['target'] = target
            attack_kwargs['text_tokens'] = text_tokens
            attack_kwargs['semantic_target_class'] = semantic_targets

            delta, _ = attack_fn(**attack_kwargs)
            adv_images = torch.clamp(images + delta, 0, 1)

            # Track semantic targets for STA computation (Priority 1A)
            all_semantic_targets.append(semantic_targets.clone())
        else:
            # Common interface for PGD, CW, SAPA
            attack_kwargs['X'] = images
            attack_kwargs['target'] = target # target is ground label
            attack_kwargs['text_tokens'] = text_tokens

            result = attack_fn(**attack_kwargs)
            # Some attacks (SAPA, targeted) return (delta, semantic_targets)
            if isinstance(result, tuple):
                delta, semantic_targets = result
                # Track semantic targets for STA computation (Priority 1A)
                if semantic_targets is not None:
                    all_semantic_targets.append(semantic_targets.clone())
            else:
                delta = result
            adv_images = torch.clamp(images + delta, 0, 1)

        # Evaluate on adversarial examples
        with torch.no_grad():
            processed_adv = clip_img_preprocessing(adv_images)
            prompted_adv = prompter(processed_adv)
            prompt_token = add_prompter()
            
            # For AUDIENCE model, use the prompt_learner
            if args.model_type == 'AUDIENCE' and prompt_learner is not None:
                adv_output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_adv, text_tokens, prompt_token, prompt_learner)
            else:
                adv_output, _ = multiGPU_CLIP(model, prompted_adv, text_tokens, prompt_token)
                
            adv_preds = adv_output.argmax(dim=1)
            adv_correct += (adv_preds == target).sum().item()

            # Priority 1A: Compute STA for targeted attacks and SAPA
            if args.attack in ['targeted_pgd', 'targeted_cw', 'sapa'] and len(all_semantic_targets) > 0:
                # Get the current batch's semantic targets
                current_semantic_targets = all_semantic_targets[-1]
                # STA = cosine similarity between adv image embedding and semantic target text embedding
                try:
                    # Get image embeddings for adversarial examples (using is_embedding=True)
                    _, _, adv_image_embed, _ = multiGPU_CLIP(model, prompted_adv, text_tokens, prompt_token, is_embedding=True)

                    # Get text embeddings for semantic targets (one per sample in batch)
                    target_texts = [f"This is a photo of a {dataset_classes[t.item()]}" for t in current_semantic_targets]
                    target_text_tokens = clip.tokenize(target_texts).to(device)
                    with torch.no_grad():
                        target_text_embed = model.encode_text(target_text_tokens)
                        target_text_embed = target_text_embed / target_text_embed.norm(dim=-1, keepdim=True)

                    if args.debug and i == 0:
                        print(f"DEBUG: adv_image_embed.shape = {adv_image_embed.shape}")
                        print(f"DEBUG: target_text_embed.shape = {target_text_embed.shape}")

                    # Compute cosine similarity (STA) for each sample
                    # Both should be (batch_size, 512)
                    sta_batch = (adv_image_embed * target_text_embed).sum(dim=-1)
                    sta_values.extend(sta_batch.cpu().tolist())
                    if args.debug and i == 0:
                        print(f"STA batch {i}: {sta_batch}")
                except Exception as e:
                    if args.debug:
                        print(f"STA computation error: {e}")
                    import traceback
                    traceback.print_exc()

        # Save examples for visualization
        if args.visualize and vis_images is None:
            vis_images = images.detach().clone()
            vis_adv_images = adv_images.detach().clone()
            vis_targets = target.detach().clone()
            vis_clean_preds = clean_preds.detach().clone()
            vis_adv_preds = adv_preds.detach().clone()
    
    # Calculate final accuracies
    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total
    
    # Visualize examples
    if args.visualize:
        visualize_examples(
            vis_images, vis_adv_images, vis_targets, 
            vis_clean_preds, vis_adv_preds, 
            [t.replace('This is a photo of a ', '') for t in texts],
            args
        )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nResults for {args.model_type} model with {args.attack} attack:")
    print(f"Clean accuracy: {clean_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    print(f"Robustness gap: {clean_acc - adv_acc:.2f}%")
    print(f"Evaluation time: {elapsed_time:.2f} seconds")
    
    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)
    # Use custom filename if provided, otherwise default naming
    # Format: {model}_{dataset}_{attack}_eps{epsilon}.txt
    if hasattr(args, 'result_filename') and args.result_filename:
        results_file = os.path.join(args.output_dir, args.result_filename)
    else:
        epsilon_float = args.epsilon / 255.0 if args.epsilon > 1 else args.epsilon
        results_file = os.path.join(args.output_dir, f"{args.model_type}_{args.dataset}_{args.attack}_eps{epsilon_float:.4f}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_type} ({args.arch})\n")
        f.write(f"Attack: {args.attack} (epsilon={args.epsilon}, steps={args.attack_iters})\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Clean accuracy: {clean_acc:.2f}%\n")
        f.write(f"Adversarial accuracy: {adv_acc:.2f}%\n")
        f.write(f"Robustness gap: {clean_acc - adv_acc:.2f}%\n")

        # Priority 1A: Write STA for targeted attacks and SAPA
        if args.attack in ['targeted_pgd', 'targeted_cw', 'sapa'] and len(sta_values) > 0:
            avg_sta = sum(sta_values) / len(sta_values)
            f.write(f"Semantic Target Alignment (STA): {avg_sta:.3f}\n")

    # Print STA for targeted attacks and SAPA
    if args.attack in ['targeted_pgd', 'targeted_cw', 'sapa'] and len(sta_values) > 0:
        avg_sta = sum(sta_values) / len(sta_values)
        print(f"Semantic Target Alignment (STA): {avg_sta:.3f}")

    return clean_acc, adv_acc

def compare_models(models_to_compare, args):
    results = []
    
    # Get dataset
    _, test_loader, class_names, original_class_names = get_dataset(args)
    
    # Create text prompts
    template = 'This is a photo of a {}'
    texts = [template.format(label) for label in class_names]
    
    for model_type, model_path in models_to_compare:
        # Update args for current model
        args.model_type = model_type
        args.model_path = model_path if model_path is not None else None
        
        # Load model
        model, preprocess, prompter, add_prompter, prompt_learner = load_target_model(args, class_names)
        
        # Evaluate
        clean_acc, adv_acc = evaluate_robustness(test_loader, texts, model, prompter, add_prompter, prompt_learner, args)
        
        # Store results
        results.append({
            'model_type': model_type,
            'clean_acc': clean_acc,
            'adv_acc': adv_acc,
            'gap': clean_acc - adv_acc
        })
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    models = [r['model_type'] for r in results]
    clean_accs = [r['clean_acc'] for r in results]
    adv_accs = [r['adv_acc'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, clean_accs, width, label='Clean Accuracy')
    plt.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Comparison - {args.attack.upper()} Attack (ε={args.epsilon})')
    plt.xticks(x, models)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"model_comparison_{args.attack}.png"))
    
    # Save comparison results to file
    comparison_file = os.path.join(args.output_dir, f"model_comparison_{args.attack}.txt")
    with open(comparison_file, 'w') as f:
        f.write(f"Attack: {args.attack} (epsilon={args.epsilon}, steps={args.attack_iters})\n")
        f.write(f"Dataset: {args.dataset}\n\n")
        
        # Header
        header = f"{'Model':<10} {'Clean Acc':<10} {'Adv Acc':<10} {'Gap':<10}"
        f.write(header + "\n")
        f.write("-" * 50 + "\n")
        
        # Results
        for r in results:
            line = f"{r['model_type']:<10} {r['clean_acc']:<10.2f} {r['adv_acc']:<10.2f} {r['gap']:<10.2f}"
            f.write(line + "\n")
    
    print(f"\nResults saved to {comparison_file}")
    print("\nSummary of robustness against attack:")
    
    # Sort models by adversarial accuracy
    sorted_results = sorted(results, key=lambda x: x['adv_acc'])

    for i, r in enumerate(sorted_results):
        print(f"{i+1}. {r['model_type']}: {r['adv_acc']:.2f}% adversarial accuracy, {r['gap']:.2f}% gap")

    return clean_acc, adv_acc


# Clean-only evaluation function:Evaluate only clean accuracy and save results
def evaluate_clean_only(test_loader, texts, model, prompter, add_prompter, prompt_learner, args):
    # Get the underlying CLIP model for SAPA components
    # The wrapper provides access to the original model
    sapa_model = model.model if hasattr(model, 'model') else model
    # Check if results file already exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Use proper filename format: [model]_clean_[dataset].txt
    results_file = os.path.join(args.output_dir, f"{args.model_type}_clean_{args.dataset}.txt")

    if os.path.exists(results_file):
        # Load existing results
        with open(results_file, 'r') as f:
            for line in f:
                if line.startswith("Clean accuracy:"):
                    return float(line.split(":")[1].strip().replace("%", ""))

    # If no cached results, compute clean accuracy
    print(f"Computing clean accuracy for {args.model_type} on {args.dataset}...")

    model.eval()
    prompter.eval()
    add_prompter.eval()
    if prompt_learner is not None:
        prompt_learner.eval()

    # Tokenize text prompts
    text_tokens = clip.tokenize(texts).to(device)

    clean_correct = 0
    total = 0

    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Evaluating clean accuracy"):
            images = images.to(device)
            target = target.to(device)
            batch_size = images.size(0)
            total += batch_size

            # Preprocess and get predictions
            processed_images = clip_img_preprocessing(images, model=model)
            prompted_images = prompter(processed_images)
            prompt_token = add_prompter()

            if args.model_type == 'AUDIENCE' and prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

            preds = output.argmax(dim=1)
            clean_correct += (preds == target).sum().item()

    # Calculate clean accuracy
    clean_acc = 100.0 * clean_correct / total

    # Save results
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Clean accuracy: {clean_acc:.2f}%\n")

    return clean_acc


if __name__ == "__main__":
    args = parse_args()
    print(args)
if __name__ == "__main__":
    args = parse_args()
    print(args)
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    _, test_loader, class_names, original_class_names = get_dataset(args)
    
    # Create text prompts
    template = 'This is a photo of a {}'
    texts = [template.format(label) for label in class_names]
    
    # Load model
    model, preprocess, prompter, add_prompter, prompt_learner = load_target_model(args, device=device, class_names=class_names)

    # Priority 1B: Initialize cross-model STA calculator
    cross_model_calculator = None
    if args.compute_cross_model_sta and CROSS_MODEL_STA_AVAILABLE:
        print("\n" + "="*80)
        print("Initializing Cross-Model STA Calculator (Priority 1B)")
        print("="*80)
        cross_model_calculator = CrossModelSTACalculator(
            eval_model_name=args.eval_model,
            device=device
        )
        if cross_model_calculator.available:
            print(f"✓ Cross-model STA enabled using {args.eval_model}")
        else:
            print(f"⚠ Cross-model STA unavailable - install: pip install open_clip_torch")
    elif args.compute_cross_model_sta and not CROSS_MODEL_STA_AVAILABLE:
        print("\n⚠ Cross-model STA requested but module not available")
        print("  Install with: pip install open_clip_torch")

    # Evaluate based on attack type
    if args.attack == 'clean':
        # Clean-only evaluation
        clean_acc = evaluate_clean_only(test_loader, texts, model, prompter, add_prompter, prompt_learner, args)
    else:
        # Regular adversarial evaluation
        clean_acc, adv_acc = evaluate_robustness(test_loader, texts, model, prompter, add_prompter, prompt_learner, class_names, original_class_names, args)
