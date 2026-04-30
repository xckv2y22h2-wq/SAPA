"""
Adversarial Text Generation for Multi-Modal SAPA Attack

"""

import torch
from modified_clip import clip


class AdversarialTextGenerator:
    def __init__(self, device='cuda', use_llava=True):
        self.device = device
        self.use_llava = use_llava
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        # Optional: Load LLaVA for rich text generation
        if self.use_llava:
            try:
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                print("Loading LLaVA for adversarial text generation...")
                self.processor = LlavaNextProcessor.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf"
                )
                self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.llava_model.eval()
                self.has_llava = True
                print("✓ LLaVA loaded for adversarial text")
            except Exception as e:
                print(f"⚠ Could not load LLaVA: {e}")
                print("Falling back to template-based text generation")
                self.has_llava = False
        else:
            self.has_llava = False

    def generate_target_description(self, target_class, use_llava=True, debug=False):
        if self.has_llava and use_llava:
            return self._generate_with_llava(target_class, debug)
        else:
            return self._generate_template(target_class)

    def _generate_with_llava(self, target_class, debug=False):
        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Describe the visual appearance of a {target_class} in detail for an image classification task.

Focus on these specific features:
1. Body shape and size
2. Coat texture and length
3. Ear shape and position
4. Snout/muzzle characteristics
5. Any distinctive breed features

Be specific and concise (1-2 sentences).<|im_end|>
<|im_start|>assistant
"""

        if debug:
            print(f"[AdvText] Generating description for '{target_class}'...")

        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.llava_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.llava_model.generation_config.eos_token_id
            )

        description = self.processor.decode(output[0], skip_special_tokens=True)
        description = description.split("assistant\n")[-1].strip()

        if debug:
            print(f"[AdvText] LLaVA generated: '{description}'")

        return description

    def _generate_template(self, target_class):
        templates = [
            f"a photo of a {target_class}",
            f"a {target_class} in its typical appearance",
            f"a clear view of a {target_class} showing characteristic features",
        ]
        return templates[0]

    def get_text_embedding(self, text):
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            embed = self.clip_model.encode_text(tokens)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            embed = embed.float()
        return embed

    def generate_adversarial_text_embed(self, target_class, use_llava=True, debug=False):
        description = self.generate_target_description(target_class, use_llava, debug)
        embed = self.get_text_embedding(description)

        if debug:
            print(f"[AdvText] Target: {target_class}")
            print(f"[AdvText] Description: '{description}'")
            print(f"[AdvText] Embedding shape: {embed.shape}")

        return embed, description


class MultiModalSemanticAttack:
    def __init__(self, clip_model, device, class_names, use_llava_text=True):
        self.clip_model = clip_model
        self.device = device
        self.class_names = class_names

        self.text_generator = AdversarialTextGenerator(
            device=device,
            use_llava=use_llava_text
        )
        self.class_embeddings = self._compute_class_embeddings()
        self.features = {}
        self._register_hooks()
        intermediate_dim = 768
        target_dim = 512

        import torch.nn as nn
        self.layer_projections = nn.ModuleDict({
            'layer_3': nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, target_dim)
            ),
            'layer_6': nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, target_dim)
            ),
            'layer_9': nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, target_dim)
            ),
        }).to(device)

        # Initialize projections
        for proj in self.layer_projections.values():
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.xavier_uniform_(proj[2].weight)
            proj[0].bias.data.zero_()
            proj[2].bias.data.zero_()

    def _compute_class_embeddings(self):
        with torch.no_grad():
            texts = [f"a photo of a {name}" for name in self.class_names]
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.float()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        self.clip_model.visual.transformer.resblocks[3].register_forward_hook(
            get_activation('layer_3')
        )
        self.clip_model.visual.transformer.resblocks[6].register_forward_hook(
            get_activation('layer_6')
        )
        self.clip_model.visual.transformer.resblocks[9].register_forward_hook(
            get_activation('layer_9')
        )

    def attack(
        self,
        clean_image,
        true_label,
        semantic_anchor_word,
        epsilon=0.031,
        attack_iters=30,
        alpha=0.01,
        text_weight=0.5,  # Weight for adversarial text guidance
        use_llava_text=True,
        debug=False
    ):
        if clean_image.dim() == 4:
            clean_image = clean_image.squeeze(0)

        if debug:
            print(f"\n{'='*70}")
            print(f"Multi-Modal Attack: {self.class_names[true_label]} → {semantic_anchor_word}")
            print(f"{'='*70}\n")

        # Step 1: Generate ADVERSARIAL TEXT for target class
        if debug:
            print("[Text Modality] Generating adversarial text...")
        adv_text_embed, adv_description = self.text_generator.generate_adversarial_text_embed(
            semantic_anchor_word,
            use_llava=use_llava_text,
            debug=debug
        )
        if debug:
            print(f"[Text Modality] Adversarial text: '{adv_description}'")
        wordnet_text = f"a photo of a {semantic_anchor_word}"
        wordnet_tokens = clip.tokenize([wordnet_text]).to(self.device)
        with torch.no_grad():
            wordnet_embed = self.clip_model.encode_text(wordnet_tokens)
            wordnet_embed = wordnet_embed / wordnet_embed.norm(dim=-1, keepdim=True)
            wordnet_embed = wordnet_embed.float()

        # Step 2: Semantic-guided initialization using adversarial text
        if debug:
            print("\n[Visual Modality] Semantic-guided initialization...")
        semantic_target_embed = (1 - text_weight) * wordnet_embed + text_weight * adv_text_embed
        semantic_target_embed = semantic_target_embed / semantic_target_embed.norm(dim=-1, keepdim=True)

        delta = self._semantic_guided_initialization(
            clean_image,
            semantic_target_embed,
            epsilon,
            debug=debug
        )
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], lr=alpha)

        # Step 3: Joint multi-modal optimization
        if debug:
            print("\n[Multi-Modal] Joint optimization loop...")

        pred_idx = -1
        for iteration in range(attack_iters):
            adv_image = torch.clamp(clean_image + delta, 0, 1)
            adv_img_embed = self.clip_model.encode_image(adv_image.unsqueeze(0))
            adv_img_embed = adv_img_embed / adv_img_embed.norm(dim=-1, keepdim=True)
            adv_img_embed = adv_img_embed.float()

            # Multi-Modal Loss
            total_loss = 0
            # Loss 1: Alignment with adversarial text
            text_alignment = (adv_img_embed @ semantic_target_embed.T).squeeze()
            L_text = -text_alignment
            total_loss += 0.4 * L_text

            # Loss 2: Multi-level feature alignment
            feat_3 = self.features['layer_3']
            feat_6 = self.features['layer_6']
            feat_9 = self.features['layer_9']
            L_layer3 = self._compute_layer_alignment_loss(feat_3, semantic_target_embed, 'layer_3')
            L_layer6 = self._compute_layer_alignment_loss(feat_6, semantic_target_embed, 'layer_6')
            L_layer9 = self._compute_layer_alignment_loss(feat_9, semantic_target_embed, 'layer_9')
            total_loss += 0.15 * L_layer3
            total_loss += 0.2 * L_layer6
            total_loss += 0.25 * L_layer9

            # Loss 3: Adversarial loss (push away from true class)
            true_class_embed = self.class_embeddings[true_label]
            true_similarity = (adv_img_embed @ true_class_embed.unsqueeze(0).T).squeeze()
            L_adversarial = true_similarity
            if pred_idx == true_label:
                total_loss += 0.5 * L_adversarial
            else:
                total_loss += 0.1 * L_adversarial

            # Optimization 
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(clean_image + delta.data, 0, 1) - clean_image
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)

            # Logging
            if iteration % 5 == 0:
                with torch.no_grad():
                    logits = adv_img_embed @ self.class_embeddings.T
                    pred_idx = logits.argmax().item()

                    if debug:
                        print(f"Iter {iteration:3d} | Pred: {self.class_names[pred_idx]:20s} | "
                              f"Text: {-L_text.item():.3f} | "
                              f"L3: {-L_layer3.item():.3f} | "
                              f"L6: {-L_layer6.item():.3f} | "
                              f"L9: {-L_layer9.item():.3f} | "
                              f"Loss: {total_loss.item():.3f}")

        # Final adversarial image
        final_adv_image = torch.clamp(clean_image + delta.detach(), 0, 1)

        # Compute metrics
        actual_perturbation = final_adv_image - clean_image
        l_inf_norm = actual_perturbation.abs().max().item()
        l2_norm = actual_perturbation.norm().item()

        # Evaluate
        with torch.no_grad():
            final_embed = self.clip_model.encode_image(final_adv_image.unsqueeze(0))
            final_embed = final_embed / final_embed.norm(dim=-1, keepdim=True)

            logits = final_embed @ self.class_embeddings.T
            pred_idx = logits.argmax().item()

            attack_info = {
                'success': pred_idx != true_label,
                'pred_class': self.class_names[pred_idx],
                'pred_class_idx': pred_idx,
                'semantic_alignment': (final_embed @ semantic_target_embed.T).item(),
                'adversarial_description': adv_description,
                'text_weight': text_weight,
                'l_inf_norm': l_inf_norm,
                'l2_norm': l2_norm
            }

        return final_adv_image.unsqueeze(0), attack_info

    def _semantic_guided_initialization(self, clean_image, semantic_target, epsilon, debug=False):
        clean_image_copy = clean_image.clone().requires_grad_(True)

        clean_embed = self.clip_model.encode_image(clean_image_copy.unsqueeze(0))
        clean_embed = clean_embed / clean_embed.norm(dim=-1, keepdim=True)

        clean_embed = clean_embed.float()
        semantic_target = semantic_target.float()

        init_loss = -(clean_embed @ semantic_target.T).sum()
        init_loss.backward()

        with torch.no_grad():
            init_delta = 0.02 * epsilon * clean_image_copy.grad.sign()
            init_delta = torch.clamp(init_delta, -epsilon, epsilon)

        if debug:
            print(f"  Init L2 norm: {init_delta.norm().item():.4f}")

        return init_delta

    def _compute_layer_alignment_loss(self, layer_features, target_embed, layer_name):
        if layer_features.dim() > 2:
            class_token = layer_features[:, 0, :]
        else:
            class_token = layer_features

        class_token = class_token.float()

        projection = self.layer_projections[layer_name]
        projected = projection(class_token)
        projected_norm = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)

        projected_norm = projected_norm.float()
        target_embed = target_embed.float()

        alignment = (projected_norm @ target_embed.T).mean()
        return -alignment
