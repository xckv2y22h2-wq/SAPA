import torch
from modified_clip import clip
import warnings

class LLaVATextAdaptation:
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load LLaVA
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        print("Loading LLaVA model...")
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.llava_model.eval()
        
        # Set pad_token_id to suppress "Setting pad_token_id to eos_token_id" warning
        if self.llava_model.generation_config.pad_token_id is None:
            self.llava_model.generation_config.pad_token_id = self.llava_model.generation_config.eos_token_id
        
        # CLIP for text encoding
        # import clip - comment out, we need to load modified_clip
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
    
    def generate_adaptive_caption(self, adv_image_tensor, semantic_anchor_word=None, debug=False):
        # Convert to PIL
        adv_image_pil = self._tensor_to_pil(adv_image_tensor)

        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Describe this image in detail in one sentence. Focus on the main object and its visual characteristics.<|im_end|>
<|im_start|>assistant
"""

        inputs = self.processor(
            text=prompt,
            images=adv_image_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Suppress generation warnings unless in debug mode
            with warnings.catch_warnings():
                if not debug:
                    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                    warnings.filterwarnings("ignore", message=".*do_sample.*")
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.llava_model.generation_config.eos_token_id
                )

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        caption = caption.split("assistant\n")[-1].strip()

        if debug and semantic_anchor_word:
            print(f"  LLaVA caption: '{caption}'")

        return caption
    
    def _tensor_to_pil(self, tensor):
        from PIL import Image
        import numpy as np
        
        # Handle batch dimension - squeeze if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def get_caption_embedding(self, caption):
        with torch.no_grad():
            tokens = clip.tokenize([caption]).to(self.device)
            embed = self.clip_model.encode_text(tokens)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            # Ensure float32 for consistency with main CLIP model
            embed = embed.float()
        return embed

    def compute_semantic_similarity(self, embed1, embed2):
        with torch.no_grad():
            # Ensure both embeddings have the same dtype (float32)
            embed1 = embed1.float()
            embed2 = embed2.float()
            similarity = (embed1 @ embed2.T).squeeze().item()
        return similarity

    def ask_what_image_looks_like(self, adv_image_tensor, candidate_classes, semantic_anchor_word, true_class_name, debug=False):
        # Convert to PIL
        adv_image_pil = self._tensor_to_pil(adv_image_tensor)

        prompt = f"""<|im_start|>system
You are an image classification assistant. You must choose one option.<|im_end|>
<|im_start|>user
<image>
Based on visual appearance, does this image look more like a {true_class_name} or a {semantic_anchor_word}?
Answer with ONLY the name of the breed you think it is.<|im_end|>
<|im_start|>assistant
"""

        inputs = self.processor(
            text=prompt,
            images=adv_image_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            with warnings.catch_warnings():
                if not debug:
                    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                    warnings.filterwarnings("ignore", message=".*do_sample.*")
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.llava_model.generation_config.eos_token_id
                )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()

        llava_response_embed = self.get_caption_embedding(f"a photo of a {response}")

        # Target: what we want it to look like
        target_embed = self.get_caption_embedding(f"a photo of a {semantic_anchor_word}")

        # True class: what it actually is
        true_embed = self.get_caption_embedding(f"a photo of a {true_class_name}")

        # Compute similarities
        sim_to_target = self.compute_semantic_similarity(llava_response_embed, target_embed)
        sim_to_true = self.compute_semantic_similarity(llava_response_embed, true_embed)

        target_alignment = sim_to_target - sim_to_true

        if debug:
            print(f"  [Reverse TTA] LLaVA says: '{response}'")
            print(f"  [Reverse TTA] Similarity to target ({semantic_anchor_word}): {sim_to_target:.3f}")
            print(f"  [Reverse TTA] Similarity to true ({true_class_name}): {sim_to_true:.3f}")
            print(f"  [Reverse TTA] Target alignment: {target_alignment:.3f}")

        return {
            'caption': response,
            'looks_like_target': sim_to_target,
            'looks_like_true': sim_to_true,
            'target_alignment': target_alignment,
            'llava_embed': llava_response_embed
        }

    def generate_target_reinforcing_caption(self, adv_image_tensor, semantic_anchor_word, debug=False):
        adv_image_pil = self._tensor_to_pil(adv_image_tensor)

        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Describe the visual features that suggest this image could be a {semantic_anchor_word}.
What characteristics resemble a {semantic_anchor_word}? Be specific.<|im_end|>
<|im_start|>assistant
"""

        inputs = self.processor(
            text=prompt,
            images=adv_image_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            with warnings.catch_warnings():
                if not debug:
                    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                    warnings.filterwarnings("ignore", message=".*do_sample.*")
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=self.llava_model.generation_config.eos_token_id
                )

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        caption = caption.split("assistant\n")[-1].strip()

        if debug:
            print(f"  [Target-Reinforcing] Caption: '{caption}'")

        return caption
