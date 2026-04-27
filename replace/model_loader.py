"""
Complete Model Loader for SAPA

Supports all robust models:
- CLIP (standard OpenAI)
- TeCoA (Text-guided Contrastive)
- PMG (Prompt-based Multi-modal Guidance)
- FARE (Feature-level Adversarial Robustness Enhancement)
- AUDIENCE (Adversarial Understanding)
- TRADES (TRadeoff-inspired Adversarial Defense via Surrogate-loss)
- MobileCLIP (Efficient CLIP variants: S0, S1, S2, B, BLT)
- MobileCLIP2 (Next-gen MobileCLIP: S0, S2, S3, S4, B, L-14)
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Tuple, Optional, Any, List, Dict

# Add parent path for imports
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_PATH not in sys.path:
    sys.path.insert(0, PARENT_PATH)

# Add path to modified CLIP
MODIFIED_CLIP_PATH = os.path.join(PARENT_PATH, 'modified_clip')
if os.path.exists(MODIFIED_CLIP_PATH) and MODIFIED_CLIP_PATH not in sys.path:
    sys.path.insert(0, MODIFIED_CLIP_PATH)

# Default paths
MODEL_CLIP_PATH = './models/'
MOBILECLIP_PATH = './models/mobileclip_models/'


# ============================================================================
# CLIP Compatibility Wrapper
# ============================================================================

class CLIPCompatibilityWrapper(nn.Module):
    """
    Wrapper to make OpenCLIP/MobileCLIP models compatible with modified CLIP interface.
    Provides the `ind_prompt()` method expected by attack functions.
    """
    def __init__(self, model, source='openclip'):
        super().__init__()
        self.model = model
        self.source = source
        
        # Store references to model's encode methods
        self._encode_image = model.encode_image
        self._encode_text = model.encode_text
        
        # Copy visual encoder reference (don't register as submodule to avoid conflicts)
        self._visual = model.visual if hasattr(model, 'visual') else None
        
        # Store logit_scale reference (don't re-register as parameter)
        if hasattr(model, 'logit_scale'):
            self._logit_scale = model.logit_scale
        else:
            # Only create new parameter if model doesn't have one
            self._logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
            self.register_parameter('_logit_scale_param', self._logit_scale)
    
    @property
    def visual(self):
        return self._visual
    
    @property
    def logit_scale(self):
        return self._logit_scale
    
    def encode_image(self, images):
        return self._encode_image(images)
    
    def encode_text(self, text_tokens):
        return self._encode_text(text_tokens)
    
    def forward(self, image, text, ind_prompt=None, prompts=None, tokenized_prompts=None, forward_type="Origin"):
        if self.source == 'modified_clip':
            # Delegate to underlying modified CLIP model
            return self.model(image, text, ind_prompt, prompts, tokenized_prompts, forward_type)
        
        # OpenCLIP / MobileCLIP - encode and return in modified CLIP format
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Return in modified CLIP format: (image_features, logit_scale * text_features)
        logit_scale = self._logit_scale.exp()
        return image_features, logit_scale * text_features
    
    def ind_prompt(self, images, text_tokens, prompter=None, add_prompter=None, 
                   prompt_learner=None, return_features=False):
        # Apply image prompter
        prompted_images = prompter(images) if prompter is not None else images
        
        # Encode images
        image_features = self.encode_image(prompted_images)
        
        # Handle text encoding
        if prompt_learner is not None:
            text_features = prompt_learner(text_tokens, self.model)
        else:
            text_features = self.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_features:
            return image_features, text_features
        
        # Return logits
        logit_scale = self._logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
    
    def get_image_features(self, images):
        features = self.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True)
    
    def get_text_features(self, text_tokens):
        features = self.encode_text(text_tokens)
        return features / features.norm(dim=-1, keepdim=True)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ============================================================================
# Prompter Classes (for evaluation - no learnable parameters)
# ============================================================================

class NullPrompter(nn.Module):
    """Prompter that does nothing - passes images through unchanged"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class TokenPrompter(nn.Module):
    """
    Token prompter for text prompts.
    For evaluation, prompt_len=0 (no token prompting)
    """
    def __init__(self, prompt_len=0):
        super().__init__()
        self.prompt_len = prompt_len
    
    def forward(self, x):
        return x


def _get_prompter_classes():
    try:
        from models.prompters import TokenPrompter as RealTokenPrompter
        from models.prompters import NullPrompter as RealNullPrompter
        return RealNullPrompter, RealTokenPrompter
    except ImportError:
        return NullPrompter, TokenPrompter


# ============================================================================
# Model Configurations
# ============================================================================

def get_model_config(model_type: str, model_dir: str = MODEL_CLIP_PATH,
                     mobileclip_dir: str = MOBILECLIP_PATH) -> Dict:
    
    configs = {
        'CLIP': {
            'type': 'CLIP',
            'path': None,
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'Standard OpenAI CLIP'
        },
        'TECOA': {
            'type': 'TeCoA',
            'path': os.path.join(model_dir, 'TeCoA_ViT-B_model_best.pth.tar'),
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'Text-guided Contrastive Adversarial robustness'
        },
        'PMG': {
            'type': 'PMG',
            'path': os.path.join(model_dir, 'PMG_model_best.pth.tar'),
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'Prompt-based Multi-modal Guidance'
        },
        'FARE': {
            'type': 'FARE',
            'path': os.path.join(model_dir, 'FARE_ViT-B_model_best.pth.tar'),
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'Feature-level Adversarial Robustness Enhancement'
        },
        'AUDIENCE': {
            'type': 'AUDIENCE',
            'path': os.path.join(model_dir, 'AUDIENCE_model_best.pth.tar'),
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'Adversarial Understanding via Dedicated Instance-specific Evidence'
        },
        'TRADES': {
            'type': 'TRADES',
            'path': os.path.join(model_dir, 'TRADES_model_best.pth.tar'),
            'is_openclip': False,
            'arch': 'ViT-B/32',
            'description': 'TRadeoff-inspired Adversarial Defense via Surrogate-loss'
        },
        'MOBILECLIP': {
            'type': 'MobileCLIP',
            'path': mobileclip_dir,
            'is_openclip': True,
            'arch': 'MobileCLIP-S2',
            'available_archs': ['MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP-S2', 
                               'MobileCLIP-B', 'MobileCLIP-BLT'],
            'description': 'Efficient MobileCLIP models'
        },
        'MOBILECLIP2': {
            'type': 'MobileCLIP2',
            'path': mobileclip_dir,
            'is_openclip': True,
            'arch': 'MobileCLIP2-S2',
            'available_archs': ['MobileCLIP2-S0', 'MobileCLIP2-S2', 'MobileCLIP2-S3', 
                               'MobileCLIP2-S4', 'MobileCLIP2-B', 'MobileCLIP2-L-14'],
            'description': 'Next-generation MobileCLIP2 models'
        }
    }
    
    # Case-insensitive lookup
    model_type_upper = model_type.upper().replace('-', '').replace('_', '')
    configs_upper = {k.upper().replace('-', '').replace('_', ''): v for k, v in configs.items()}
    
    if model_type_upper not in configs_upper:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(configs.keys())}")
    
    return configs_upper[model_type_upper]


def get_all_model_configs(model_dir: str = MODEL_CLIP_PATH,
                          mobileclip_dir: str = MOBILECLIP_PATH) -> List[Dict]:
    
    all_models = []
    model_types = ['CLIP', 'TeCoA', 'PMG', 'FARE', 'AUDIENCE', 'TRADES', 
                   'MobileCLIP', 'MobileCLIP2']
    
    for model_type in model_types:
        config = get_model_config(model_type, model_dir, mobileclip_dir)
        
        # Check availability
        if config['path'] is not None:
            if config['is_openclip']:
                config['available'] = os.path.isdir(config['path'])
            else:
                config['available'] = os.path.isfile(config['path'])
        else:
            config['available'] = True
        
        all_models.append(config)
    
    return all_models


# ============================================================================
# MobileCLIP Loading
# ============================================================================

def load_mobileclip_model(arch: str,
                          model_path: str,
                          device: str = 'cuda') -> Tuple[nn.Module, Any, Any]:
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    
    print(f"Loading MobileCLIP model: {arch}")
    print(f"  Model path: {model_path}")
    
    # Construct pretrained path
    pretrained_filename = f'{arch.lower().replace("-", "_")}.pt'
    pretrained_path = os.path.join(model_path, pretrained_filename)
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
    
    print(f"  Loading weights from: {pretrained_path}")
    
    # Set normalization kwargs based on architecture
    model_kwargs = {}
    if not (arch.endswith("S3") or arch.endswith("S4") or arch.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
        print(f"  ✓ Using custom normalization: mean=(0,0,0), std=(1,1,1)")
    else:
        print(f"  ✓ Using default normalization")
    
    # Create model
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch,
        pretrained=pretrained_path,
        device=device,
        **model_kwargs
    )
    
    # Reparameterize for inference
    model.eval()
    model = reparameterize_model(model)
    
    # Wrap for compatibility
    model = CLIPCompatibilityWrapper(model, source='openclip')
    
    # Get tokenizer
    tokenizer = open_clip.get_tokenizer(arch)
    
    # Store metadata
    model.is_openclip = True
    model.tokenizer = tokenizer
    model.arch = arch
    
    # Convert to float32
    model.float()
    
    print(f"  ✓ Successfully loaded {arch}")
    
    return model, preprocess, tokenizer


# ============================================================================
# CLIP Loading
# ============================================================================

def load_clip_model(arch: str = 'ViT-B/32', 
                   device: str = 'cuda',
                   checkpoint_path: Optional[str] = None,
                   wrap_model: bool = True) -> Tuple[nn.Module, Any]:
    print(f"Loading CLIP model...")
    print(f"  Architecture: {arch}")
    print(f"  Device: {device}")
    
    # Import modified CLIP
    try:
        from modified_clip import clip
    except ImportError:
        import clip
    
    # Load base model
    model, preprocess = clip.load(arch, device=device, jit=False)
    print(f"  ✓ Loaded CLIP model")
    
    # Load checkpoint if provided (only vision encoder weights)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from: {checkpoint_path}")
        _load_vision_encoder_weights(model, checkpoint_path, device)
    
    # Ensure float32
    model.float()
    model.eval()
    
    # Wrap model
    if wrap_model:
        model = CLIPCompatibilityWrapper(model, source='modified_clip')
        model.is_openclip = False
        model.tokenizer = None
        model.arch = arch
        model.preprocess = preprocess
    
    return model, preprocess


def _load_vision_encoder_weights(model: nn.Module, checkpoint_path: str, device: str):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"    Loaded checkpoint")
        
        # Load vision encoder weights (this is what run_attack.py does)
        if 'vision_encoder_state_dict' in checkpoint:
            if hasattr(model, 'visual'):
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
                print(f"    ✓ Loaded vision encoder weights")
            else:
                print(f"    ⚠ Model has no 'visual' attribute")
        else:
            print(f"    ⚠ No 'vision_encoder_state_dict' in checkpoint")
            print(f"    Available keys: {list(checkpoint.keys())}")
        
    except Exception as e:
        print(f"    ✗ Error loading checkpoint: {e}")


# ============================================================================
# Main Model Loading Interface
# ============================================================================

def load_model(model_type: str,
               arch: str = None,
               device: str = 'cuda',
               model_dir: str = MODEL_CLIP_PATH,
               mobileclip_dir: str = MOBILECLIP_PATH,
               class_names: list = None,
               wrap_model: bool = True) -> Tuple[nn.Module, Any, nn.Module, nn.Module, Any]:
    print(f"\n{'='*70}")
    print(f"Loading Model: {model_type}")
    print(f"{'='*70}")
    
    # Get configuration
    config = get_model_config(model_type, model_dir, mobileclip_dir)
    
    # Use provided arch or default from config
    if arch is None:
        arch = config['arch']
    
    print(f"  Type: {config['type']}")
    print(f"  Architecture: {arch}")
    print(f"  Description: {config['description']}")
    
    # Get prompter classes
    NullPrompterClass, TokenPrompterClass = _get_prompter_classes()
    
    # Create prompters (same for ALL models during evaluation)
    prompter = NullPrompterClass().to(device)
    add_prompter = TokenPrompterClass(0).to(device)  # No token prompting for evaluation
    prompt_learner = None
    
    print(f"  ✓ Created prompters (NullPrompter, TokenPrompter(0))")
    
    # Load model based on type
    if config['is_openclip']:
        # MobileCLIP / MobileCLIP2
        model, preprocess, tokenizer = load_mobileclip_model(
            arch=arch,
            model_path=config['path'],
            device=device
        )
    else:
        # Standard CLIP or fine-tuned variants
        checkpoint_path = config['path'] if config['type'] != 'CLIP' else None
        
        model, preprocess = load_clip_model(
            arch=arch,
            device=device,
            checkpoint_path=checkpoint_path,
            wrap_model=wrap_model
        )
        
        # For AUDIENCE, create prompt learner
        if config['type'] == 'AUDIENCE' and class_names is not None:
            prompt_learner = _create_prompt_learner(model, class_names, device)
    
    print(f"{'='*70}\n")
    
    return model, preprocess, prompter, add_prompter, prompt_learner


def _create_prompt_learner(model: nn.Module, class_names: list, device: str,
                           ctx: int = 16, ctx_init: str = 'This is a photo of a',
                           position: str = 'end'):
    try:
        from models.prompters import PromptLearner
        
        # Need to access the unwrapped model for PromptLearner
        unwrapped_model = model.model if hasattr(model, 'model') else model
        
        # Create args object matching run_attack.py defaults
        class Args:
            pass
        
        args = Args()
        args.ctx = ctx
        args.ctx_init = ctx_init
        args.position = position
        
        prompt_learner = PromptLearner(args, class_names, unwrapped_model).to(device)
        print(f"  ✓ Created PromptLearner (ctx={ctx}, position='{position}')")
        
        return prompt_learner
        
    except ImportError as e:
        print(f"  ⚠ Could not import PromptLearner: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ Could not create prompt learner: {e}")
        return None


# ============================================================================
# Convenience Functions
# ============================================================================

def load_clip(arch: str = 'ViT-B/32', device: str = 'cuda') -> Tuple:
    return load_model('CLIP', arch, device)


def load_tecoa(arch: str = 'ViT-B/32', device: str = 'cuda', 
               model_dir: str = MODEL_CLIP_PATH) -> Tuple:
    return load_model('TeCoA', arch, device, model_dir)


def load_pmg(arch: str = 'ViT-B/32', device: str = 'cuda',
             model_dir: str = MODEL_CLIP_PATH) -> Tuple:
    return load_model('PMG', arch, device, model_dir)


def load_fare(arch: str = 'ViT-B/32', device: str = 'cuda',
              model_dir: str = MODEL_CLIP_PATH) -> Tuple:
    return load_model('FARE', arch, device, model_dir)


def load_audience(arch: str = 'ViT-B/32', device: str = 'cuda',
                  model_dir: str = MODEL_CLIP_PATH,
                  class_names: list = None) -> Tuple:
    if class_names is None:
        raise ValueError(
            "AUDIENCE model requires class_names. "
            "Get them from your dataset, e.g.:\n"
            "  from data.dataset_loader import get_dataset_class_names\n"
            "  class_names = get_dataset_class_names('cifar10')"
        )
    return load_model('AUDIENCE', arch, device, model_dir, class_names=class_names)


def load_trades(arch: str = 'ViT-B/32', device: str = 'cuda',
                model_dir: str = MODEL_CLIP_PATH) -> Tuple:
    return load_model('TRADES', arch, device, model_dir)


def load_mobileclip(arch: str = 'MobileCLIP-S2', device: str = 'cuda',
                    mobileclip_dir: str = MOBILECLIP_PATH) -> Tuple:
    return load_model('MobileCLIP', arch, device, mobileclip_dir=mobileclip_dir)


def load_mobileclip2(arch: str = 'MobileCLIP2-S2', device: str = 'cuda',
                     mobileclip_dir: str = MOBILECLIP_PATH) -> Tuple:
    return load_model('MobileCLIP2', arch, device, mobileclip_dir=mobileclip_dir)


# ============================================================================
# Tokenization Helper
# ============================================================================

def get_text_tokens(texts: list, model: nn.Module, device: str = 'cuda'):
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        # MobileCLIP uses OpenCLIP tokenizer
        tokens = model.tokenizer(texts).to(device)
    else:
        # Standard CLIP tokenization
        try:
            from modified_clip import clip
        except ImportError:
            import clip
        tokens = clip.tokenize(texts, truncate=True).to(device)
    
    return tokens


# ============================================================================
# Utility Functions
# ============================================================================

def print_available_models(model_dir: str = MODEL_CLIP_PATH,
                           mobileclip_dir: str = MOBILECLIP_PATH):
    
    configs = get_all_model_configs(model_dir, mobileclip_dir)
    
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    print(f"CLIP checkpoint directory: {model_dir}")
    print(f"MobileCLIP directory: {mobileclip_dir}\n")
    
    for config in configs:
        status = "✓" if config.get('available', False) else "✗"
        print(f"{status} {config['type']:12s} - {config['description']}")
        if config['is_openclip'] and 'available_archs' in config:
            print(f"    Architectures: {', '.join(config['available_archs'])}")
        if not config.get('available', False) and config['path']:
            print(f"    ⚠ Missing: {config['path']}")
    
    print("="*70 + "\n")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Print available models
    print_available_models()
    
    # Example 1: Load standard CLIP
    print("\n" + "="*70)
    print("Example 1: Loading Standard CLIP")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_clip('ViT-B/32', device)
        print(f"Model type: {type(model).__name__}")
        print(f"Prompter: {type(prompter).__name__}")
        print(f"Add prompter: {type(add_prompter).__name__}")
        print(f"Prompt learner: {prompt_learner}")
        print("✓ Success!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 2: Load MobileCLIP2-S2
    print("="*70)
    print("Example 2: Loading MobileCLIP2-S2")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip2(
            'MobileCLIP2-S2', device
        )
        print(f"Model type: {type(model).__name__}")
        print(f"Has tokenizer: {model.tokenizer is not None}")
        print("✓ Success!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 3: Load MobileCLIP-S2
    print("="*70)
    print("Example 3: Loading MobileCLIP-S2")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip(
            'MobileCLIP-S2', device
        )
        print(f"Model type: {type(model).__name__}")
        print("✓ Success!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 4: Load TeCoA
    print("="*70)
    print("Example 4: Loading TeCoA")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_tecoa('ViT-B/32', device)
        print(f"Model type: {type(model).__name__}")
        print("✓ Success!\n")
    except FileNotFoundError as e:
        print(f"⚠ Checkpoint not found: {e}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 5: Load FARE
    print("="*70)
    print("Example 5: Loading FARE")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_fare('ViT-B/32', device)
        print(f"Model type: {type(model).__name__}")
        print("✓ Success!\n")
    except FileNotFoundError as e:
        print(f"⚠ Checkpoint not found: {e}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 6: Load PMG
    print("="*70)
    print("Example 6: Loading PMG")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_pmg('ViT-B/32', device)
        print(f"Model type: {type(model).__name__}")
        print("✓ Success!\n")
    except FileNotFoundError as e:
        print(f"⚠ Checkpoint not found: {e}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 7: Load TRADES
    print("="*70)
    print("Example 7: Loading TRADES")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, prompt_learner = load_trades('ViT-B/32', device)
        print(f"Model type: {type(model).__name__}")
        print("✓ Success!\n")
    except FileNotFoundError as e:
        print(f"⚠ Checkpoint not found: {e}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 8: Load AUDIENCE with class names from dataset
    print("="*70)
    print("Example 8: Loading AUDIENCE (requires class_names from dataset)")
    print("="*70)
    try:
        # Try to get class names from dataset loader
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        print(f"  Using hardcoded CIFAR-10 class names: {len(class_names)} classes")
        
        model, preprocess, prompter, add_prompter, prompt_learner = load_audience(
            'ViT-B/32', device, class_names=class_names
        )
        print(f"Model type: {type(model).__name__}")
        print(f"Prompt learner: {type(prompt_learner).__name__ if prompt_learner else 'None'}")
        print("✓ Success!\n")
    except FileNotFoundError as e:
        print(f"⚠ Checkpoint not found: {e}\n")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        print()
    
    # Example 9: Test tokenization
    print("="*70)
    print("Example 9: Test Tokenization (CLIP vs MobileCLIP)")
    print("="*70)
    try:
        texts = ["a photo of a cat", "a photo of a dog", "a red car"]
        
        # Test with CLIP
        clip_model, _, _, _, _ = load_clip('ViT-B/32', device)
        clip_tokens = get_text_tokens(texts, clip_model, device)
        print(f"CLIP tokens shape: {clip_tokens.shape}")
        
        # Test with MobileCLIP2
        mobile_model, _, _, _, _ = load_mobileclip2('MobileCLIP2-S2', device)
        mobile_tokens = get_text_tokens(texts, mobile_model, device)
        print(f"MobileCLIP2 tokens shape: {mobile_tokens.shape}")
        
        print("✓ Success!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 10: Test forward pass
    print("="*70)
    print("Example 10: Test Forward Pass")
    print("="*70)
    try:
        model, preprocess, prompter, add_prompter, _ = load_mobileclip2('MobileCLIP2-S2', device)
        
        # Create dummy input
        dummy_images = torch.randn(4, 3, 224, 224).to(device)
        texts = ["cat", "dog", "bird", "fish"]
        text_tokens = get_text_tokens(texts, model, device)
        
        # Test forward()
        with torch.no_grad():
            logits = model(dummy_images, text_tokens)
        print(f"Forward output shape: {logits.shape}")
        
        # Test ind_prompt()
        with torch.no_grad():
            logits_prompt = model.ind_prompt(dummy_images, text_tokens, prompter, add_prompter)
        print(f"ind_prompt output shape: {logits_prompt.shape}")
        
        print("✓ Success!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Example 11: Load all MobileCLIP2 variants
    print("="*70)
    print("Example 11: Loading All MobileCLIP2 Variants")
    print("="*70)
    mobileclip2_archs = ['MobileCLIP2-S0', 'MobileCLIP2-S2', 'MobileCLIP2-S3', 'MobileCLIP2-S4']
    for arch in mobileclip2_archs:
        try:
            model, _, _, _, _ = load_mobileclip2(arch, device)
            print(f"  ✓ {arch}: Loaded successfully")
        except FileNotFoundError:
            print(f"  ⚠ {arch}: Weights not found")
        except Exception as e:
            print(f"  ✗ {arch}: Error - {e}")
    print()
    
    # Example 12: Generic load_model interface
    print("="*70)
    print("Example 12: Using Generic load_model() Interface")
    print("="*70)
    model_types_to_test = [
        ('CLIP', 'ViT-B/32', None),
        ('MobileCLIP', 'MobileCLIP-S2', None),
        ('MobileCLIP2', 'MobileCLIP2-S2', None),
        ('TeCoA', 'ViT-B/32', None),
        ('FARE', 'ViT-B/32', None),
        ('PMG', 'ViT-B/32', None),
        ('TRADES', 'ViT-B/32', None),
        # AUDIENCE requires class_names
        ('AUDIENCE', 'ViT-B/32', ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                   'dog', 'frog', 'horse', 'ship', 'truck']),
    ]
    
    for model_type, arch, class_names in model_types_to_test:
        try:
            model, _, prompter, add_prompter, prompt_learner = load_model(
                model_type=model_type,
                arch=arch,
                device=device,
                class_names=class_names
            )
            is_openclip = getattr(model, 'is_openclip', False)
            has_pl = prompt_learner is not None
            print(f"  ✓ {model_type:12s} ({arch}): OpenCLIP={is_openclip}, PromptLearner={has_pl}")
        except FileNotFoundError:
            print(f"  ⚠ {model_type:12s} ({arch}): Weights not found")
        except Exception as e:
            print(f"  ✗ {model_type:12s} ({arch}): {str(e)[:50]}")
    print()
    
    # Summary
    print("="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("""
Usage Examples:
    
    # Load standard CLIP
    model, preprocess, prompter, add_prompter, _ = load_clip('ViT-B/32', 'cuda')
    
    # Load MobileCLIP2
    model, preprocess, prompter, add_prompter, _ = load_mobileclip2('MobileCLIP2-S4', 'cuda')
    
    # Load robust models (no class_names needed)
    model, preprocess, prompter, add_prompter, _ = load_tecoa('ViT-B/32', 'cuda')
    model, preprocess, prompter, add_prompter, _ = load_fare('ViT-B/32', 'cuda')
    model, preprocess, prompter, add_prompter, _ = load_pmg('ViT-B/32', 'cuda')
    model, preprocess, prompter, add_prompter, _ = load_trades('ViT-B/32', 'cuda')
    
    # Load AUDIENCE - REQUIRES class_names from dataset!
    from data.dataset_loader import get_dataset_class_names
    class_names = get_dataset_class_names('cifar10')  # or 'cifar100', 'ImageNet', etc.
    model, preprocess, prompter, add_prompter, prompt_learner = load_audience(
        'ViT-B/32', 'cuda', class_names=class_names
    )
    
    # Generic interface
    model, preprocess, prompter, add_prompter, prompt_learner = load_model(
        model_type='AUDIENCE',
        arch='ViT-B/32',
        device='cuda',
        class_names=class_names  # Required for AUDIENCE
    )
    
    # Tokenize text
    tokens = get_text_tokens(['a photo of a cat'], model, 'cuda')
    
    # Forward pass
    logits = model(images, tokens)
    
    # With prompters (for attacks)
    logits = model.ind_prompt(images, tokens, prompter, add_prompter)
    
    # With prompt_learner (AUDIENCE only)
    logits = model.ind_prompt(images, tokens, prompter, add_prompter, prompt_learner)
    """)