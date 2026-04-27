import torch
import os 
import random
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from utils import refine_classname

# Import custom datasets from replace.tv_datasets
from replace.tv_datasets import StanfordCars, Food101, SUN397, EuroSAT, DTD, Imagenet100, \
        Caltech101, Caltech256, Country211, Flowers102, PCAM, FGVCAircraft, OxfordIIITPet


def limit_test_samples_balanced(test_dataset, train_dataset, num_samples, seed=42, verbose=True):
    
    random.seed(seed)
    
    # Handle Subset case
    if isinstance(test_dataset, Subset):
        base_dataset = test_dataset.dataset
        available_indices = test_dataset.indices
    else:
        base_dataset = test_dataset
        available_indices = list(range(len(test_dataset)))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BALANCED SAMPLING")
        print(f"{'='*70}")
        print(f"Total available samples: {len(available_indices)}")
    
    # Group samples by class
    class_to_indices = defaultdict(list)
    for idx in available_indices:
        _, label = base_dataset[idx]
        class_to_indices[label].append(idx)
    
    num_classes = len(class_to_indices)
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    if verbose:
        print(f"Found {num_classes} unique classes")
        print(f"Target: {samples_per_class} samples per class")
        if remainder > 0:
            print(f"  + {remainder} extra samples to distribute")
    
    # Sample from each class
    sampled_indices = []
    classes_with_insufficient = []
    
    for class_label in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_label]
        
        # Determine how many to sample from this class
        n_to_sample = samples_per_class
        if remainder > 0:
            n_to_sample += 1
            remainder -= 1
        
        # Sample
        if len(indices) >= n_to_sample:
            sampled = random.sample(indices, n_to_sample)
        else:
            sampled = indices  # Take all if not enough
            classes_with_insufficient.append((class_label, len(indices), n_to_sample))
        
        sampled_indices.extend(sampled)
    
    if verbose and classes_with_insufficient:
        print(f"\nWarning: {len(classes_with_insufficient)} classes have insufficient samples:")
        for class_label, available, wanted in classes_with_insufficient[:3]:
            print(f"  Class {class_label}: {available} available (wanted {wanted})")
        if len(classes_with_insufficient) > 3:
            print(f"  ... and {len(classes_with_insufficient)-3} more")
    
    # Shuffle to mix classes
    random.shuffle(sampled_indices)
    
    # Create balanced subset
    balanced_dataset = Subset(base_dataset, sampled_indices)
    balanced_dataset.classes = train_dataset.classes
    
    if verbose:
        # Verify distribution
        actual_distribution = defaultdict(int)
        for idx in sampled_indices:
            _, label = base_dataset[idx]
            actual_distribution[label] += 1
        
        print(f"\n✓ Balanced sampling complete:")
        print(f"  Total samples: {len(sampled_indices)}")
        print(f"  Classes covered: {len(actual_distribution)}/{num_classes}")
        print(f"  Samples per class: {min(actual_distribution.values())}-{max(actual_distribution.values())}")
        
        # Show first few classes
        print(f"\n  Sample distribution (first 5 classes):")
        for i, label in enumerate(sorted(actual_distribution.keys())[:5]):
            print(f"    Class {label}: {actual_distribution[label]} samples")
        if len(actual_distribution) > 5:
            print(f"    ... and {len(actual_distribution)-5} more classes")
        print(f"{'='*70}\n")
    
    return balanced_dataset


def get_dataset(args):

    print(f"Loading dataset: {args.dataset} from {args.data_path}")
    
    # Define standard transforms for different image sizes
    transform_32 = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ✅ Determine correct image size based on architecture
    img_size = 224  # Default for most models
    
    if hasattr(args, 'arch') and args.arch.startswith('MobileCLIP'):
        # MobileCLIP-S0,S1,S2,S3, S4 use 256x256 
        # MobileCLIP-B, L14 use 224x224
        # MobileCLIP2-S0, S2, S3, S4 use 256x256
        # MobileCLIP2-B, L14 use 224x224
        if any(x in args.arch for x in ['S0', 'S1', 'S2', 'S3', 'S4']):
            img_size = 256
        else:  # B, L14
            img_size = 224
    
    print(f"Using image size: {img_size}×{img_size} for {args.arch if hasattr(args, 'arch') else 'default'}")
    
    # ✅ Create transform based on determined size
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
    
    # ===================================================================
    # CRITICAL FIX: Balanced sampling for test dataset
    # ===================================================================
    if hasattr(args, 'num_test_samples') and args.num_test_samples is not None:
        # Use balanced sampling to ensure coverage of all classes
        seed = getattr(args, 'seed', 42)
        
        test_dataset = limit_test_samples_balanced(
            test_dataset=test_dataset,
            train_dataset=train_dataset,
            num_samples=args.num_test_samples,
            seed=seed,
            verbose=True
        )
    
    # Create DataLoaders
    batch_size = getattr(args, 'batch_size', 32)
    num_workers = getattr(args, 'workers', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Refine class names for CLIP
    original_class_names = class_names.copy()
    class_names = refine_classname(class_names)
    
    print(f"\nDataset loaded: {args.dataset}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, class_names, original_class_names