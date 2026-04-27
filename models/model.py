import torch

from modified_clip import clip

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
# Convert normalization stats into tensors shaped to broadcast across image tensors of shape [3,H,W].


# preprocessing and normalization  
def normalize(X):
    return (X - mu.to(X.device)) / std.to(X.device)

def clip_img_preprocessing(X, model=None):
    
    img_size = 224  # Default
    normalize_for_model = True  # Default: normalize for other architectures
    # Get image size from model if OpenCLIP
    if model is not None and hasattr(model, 'is_openclip') and model.is_openclip:
        normalize_for_model = False  # OpenCLIP models do their own normalization
        # ✅ Correct size determination based on architecture
        if hasattr(model, 'arch'):
            arch = model.arch
            # S0, S2, S3, S4 use 256; B, L14 use 224
            if any(x in arch for x in ['S0', 'S2', 'S3', 'S4']):
                img_size = 256
            else:
                img_size = 224
        elif hasattr(model, 'visual') and hasattr(model.visual, 'image_size'):
            # Fallback: read from model
            raw_size = model.visual.image_size
            while isinstance(raw_size, (tuple, list)):
                raw_size = raw_size[0]
            img_size = int(raw_size)
    
    # Ensure img_size is a plain integer
    img_size = int(img_size)
    
    # Only resize if needed
    if X.size(-1) != img_size or X.size(-2) != img_size:
        X = torch.nn.functional.interpolate(
            X, 
            size=(img_size, img_size), 
            mode='bicubic',
            align_corners=False
        )

    # Normalize for other architectures
    if normalize_for_model:
        X = normalize(X)
    return X

# creating cosine similarity logits
# - normalize the input tensors to have unit norm (l2 norm)
# - compute the cosine similarity between the two tensors
# - scale the cosine similarity by a logit scale factor
# - return the logits for both input tensors
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

'''
Several functions handle distributed processing across multiple GPUs:

- multiGPU_CLIP_image_logits(): Prepares images and computes logits with optional prompting
- multiGPU_CLIP(): Core function that handles distributing the workload and processing inputs
- multiGPU_CLIP_Text_Prompt_Tuning(): Specialized version that supports text prompt tuning

'''
def multiGPU_CLIP_image_logits(images, model, text_tokens, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(model, image_tokens, text_tokens, prompt_token=prompt_token)[0]


def multiGPU_CLIP(model, images, text_tokens, prompt_token=None, is_embedding=False):
    # print("text_token shape", text_tokens.shape)
    # add model check 
    if hasattr(model, 'is_openclip') and model.is_openclip:
        if images.size(0) == 1:     # single image with 2 GPUs
            images = images.repeat(2,1,1,1) # the single image is duplicated to create a batch of 2 identical images 
            img_embed = model.encode_image(images) 
            img_embed = img_embed[0].unsqueeze(0) # only the first embedding is kept, since the second one is just a duplicate 
            scale_text_embed = model.encode_text(text_tokens)
        elif text_tokens.size(0) == 2:  # two text tokens with 4 GPUs
            text_tokens = text_tokens.repeat(2,1) # the text tokens are duplicated to create a batch of 4 text tokens
            img_embed = model.encode_image(images) 
            scale_text_embed = model.encode_text(text_tokens)  # model can unilize 4 GPUs 
            text_tokens = text_tokens[0:2]  # after processing, the text tokens are restored to their original size
        else:
            print ("Normal OpenCLIP processing - single GPU")
            img_embed = model.encode_image(images) 
            scale_text_embed = model.encode_text(text_tokens)
            print(f"\n" + "="*60)
            print("FIRST BATCH IMAGE ENCODING DEBUG")
            print("="*60)
            print(f"Image batch shape: {images.shape}")
            print(f"Image features shape: {img_embed.shape}")
            print(f"Image feature norms (before normalization): {img_embed.norm(dim=-1)[:5]}")
        # normalize the embeddings
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        print(f"Image feature norms (after normalization): {img_embed.norm(dim=-1)[:5]}")
        print("="*60 + "\n")
    else: 
        # modified CLIP
        if prompt_token is not None:
            bs = images.size(0)
            prompt_token = prompt_token.repeat(bs, 1, 1)
        if images.size(0) == 1:     # single image with 2 GPUs
            images = images.repeat(2,1,1,1) # the single image is duplicated to create a batch of 2 identical images 
            img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
            img_embed = img_embed[0].unsqueeze(0) # only the first embedding is kept, since the second one is just a duplicate 
            # print("images_shape", images.shape)
            # print("scale_text_embed_shape", scale_text_embed.shape)
        elif text_tokens.size(0) == 2:  # two text tokens with 4 GPUs
            text_tokens = text_tokens.repeat(2,1) # the text tokens are duplicated to create a batch of 4 text tokens
            img_embed, scale_text_embed = model(images, text_tokens, prompt_token)  # model can unilize 4 GPUs 
            text_tokens = text_tokens[0:2]  # after processing, the text tokens are restored to their original size
        else:
            img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
        # print("img_embed_shape", img_embed.shape, "scale_text_embed_shape", scale_text_embed.shape)
    
            
    logits_per_image = img_embed @ scale_text_embed.t() 
    logits_per_text = scale_text_embed @ img_embed.t()
 
    # print("img_emb_size", img_embed.shape)
    # print("logits_size", logits_per_image.shape)

    if is_embedding: # if is_embedding is True, return the embeddings as well
        return logits_per_image, logits_per_text, img_embed, scale_text_embed
    else:
        return logits_per_image, logits_per_text

# prompt tuning - perform inference with learned text prompts 
def multiGPU_CLIP_Text_Prompt_Tuning(model, images, text_tokens, prompt_token=None, prompt_learner=None, is_embedding=False): 
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    prompts = prompt_learner()
    # Fix: Check if prompt_learner is wrapped in DataParallel
    if hasattr(prompt_learner, 'module'):
        tokenized_prompts = prompt_learner.module.tokenized_prompts
    else:
        tokenized_prompts = prompt_learner.tokenized_prompts
    img_embed, scale_text_embed = model(images, text_tokens, prompt_token, prompts, tokenized_prompts, forward_type='Text_Prompt_Tuning')

    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()

    if is_embedding:
        return logits_per_image, logits_per_text, img_embed, scale_text_embed
    else:
        return logits_per_image, logits_per_text

##############################  Noise Modulated CLIP  ###########################################

def apply_multiplicative_noise(signal, beta=0.0):
    m, d = signal.shape
    noise = torch.normal(mean=1.0, std=beta, size=(1, d)).cuda()
    noisy_signal = signal * noise
    return noisy_signal

def multiGPU_CLIP_multiply_noise(model, images, text_tokens, prompt_token=None, is_embedding=False, beta=0.0):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)

    ### Noise modulate ###
    img_embed = apply_multiplicative_noise(img_embed, beta)
    scale_text_embed = apply_multiplicative_noise(scale_text_embed, beta)
    ### Noise modulate ###

    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()

    if is_embedding:
        return logits_per_image, logits_per_text, img_embed, scale_text_embed
    else:
        return logits_per_image, logits_per_text



















