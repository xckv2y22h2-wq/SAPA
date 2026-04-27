# model_compatibility.py

import torch
import torch.nn as nn

class CLIPCompatibilityWrapper(nn.Module):
    """
    Wrapper to make OpenCLIP models compatible with modified CLIP interface
    """
    def __init__(self, model, source='openclip'):
        super().__init__()
        self.model = model
        self.source = source  # 'openclip' or 'modified_clip'
    
    def forward(self, image, text, ind_prompt=None, prompts=None, tokenized_prompts=None, forward_type="Origin"):
        if self.source == 'modified_clip':
            # Your modified CLIP's forward - pass all arguments
            return self.model(image, text, ind_prompt, prompts, tokenized_prompts, forward_type)
        
        elif self.source == 'openclip':
            # OpenCLIP - encode image and text, return in modified CLIP format
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Apply logit scale to text features (matching modified CLIP format)
            logit_scale = self.model.logit_scale.exp()
            return image_features, logit_scale * text_features
    
    def encode_image(self, images, ind_prompt=None):
        if self.source == 'modified_clip':
            # Your modified CLIP supports ind_prompt
            return self.model.encode_image(images, ind_prompt=ind_prompt)
        
        elif self.source == 'openclip':
            # OpenCLIP/MobileCLIP doesn't support ind_prompt
            # Just ignore ind_prompt and use standard interface
            return self.model.encode_image(images)
    
    def encode_text(self, text):
        return self.model.encode_text(text)
    
    @property
    def visual(self):
        return self.model.visual
    
    @property
    def transformer(self):
        if hasattr(self.model, 'transformer'):
            return self.model.transformer
        elif hasattr(self.model, 'text'):
            return self.model.text
        return None
    
    @property  
    def logit_scale(self):
        return self.model.logit_scale
    
    @property
    def dtype(self):
        if hasattr(self.model, 'dtype'):
            return self.model.dtype
        return torch.float32
    
    def float(self):
        self.model.float()
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode=True):
        self.model.train(mode)
        return self
    
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)