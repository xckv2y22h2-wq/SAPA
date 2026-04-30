import torch
from modified_clip import clip

class SemanticFeatureSpacePerturbationAdaptive:
    def __init__(self, clip_model, llava_adapter, device, class_names,
                 layer_weights=None, adaptive=False):
        self.clip_model = clip_model
        self.llava_adapter = llava_adapter
        self.device = device
        self.class_names = class_names
        self.adaptive = adaptive
        self.clip_model.float()
        self.class_embeddings = self._compute_class_embeddings()
        if layer_weights is None:
            self.layer_weights = [0.15, 0.2, 0.25, 0.4]
        else:
            self.layer_weights = layer_weights
            total = sum(self.layer_weights)
            self.layer_weights = [w / total for w in self.layer_weights]
        self.features = {}
        self._register_hooks()

        print(f"SemanticFeatureSpacePerturbationAdaptive initialized")
        print(f"  Layer weights: L3={self.layer_weights[0]:.2f}, "
              f"L6={self.layer_weights[1]:.2f}, L9={self.layer_weights[2]:.2f}, "
              f"Final={self.layer_weights[3]:.2f}")
        print(f"  Adaptive mode: {self.adaptive}")

    def _compute_class_embeddings(self):
        with torch.no_grad():
            texts = [f"a photo of a {name}" for name in self.class_names]
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

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
        tta_frequency=10,
        adversarial_loss_weight=0.1,
        text_weight=None,
        use_llava_text=False,
        debug=False
    ):

        if clean_image.dim() == 4:
            clean_image = clean_image.squeeze(0)

        if debug:
            print(f"\n{'='*70}")
            print(f"Attacking: {self.class_names[true_label]} → {semantic_anchor_word}")
            print(f"{'='*70}\n")

        wordnet_text = f"a photo of a {semantic_anchor_word}"
        wordnet_tokens = clip.tokenize([wordnet_text]).to(self.device)

        with torch.no_grad():
            wordnet_embed = self.clip_model.encode_text(wordnet_tokens)
            wordnet_embed = wordnet_embed / wordnet_embed.norm(dim=-1, keepdim=True)
            wordnet_embed = wordnet_embed.float()

        semantic_target_embed = wordnet_embed

        if debug:
            print("\n[Phase 2 - Initialization] Semantic-guided initialization...")

        delta = self._semantic_guided_initialization(
            clean_image,
            semantic_target_embed,
            epsilon,
            debug=debug
        )
        delta.requires_grad = True

        optimizer = torch.optim.Adam([delta], lr=alpha)
        current_text_embed = semantic_target_embed.clone()
        new_caption = None

        if debug:
            print(f"\n[Phase 2 - Optimization] Adaptive multi-layer optimization...")

        pred_idx = -1
        for iteration in range(attack_iters):
            llava_target_alignment = 0.0
            if self.llava_adapter is not None and tta_frequency > 0 and iteration % tta_frequency == 0 and iteration > 0:
                if debug:
                    print(f"\n[Phase 3 - Reverse TTA] Iteration {iteration}: Asking LLaVA what image looks like...")

                with torch.no_grad():
                    adv_image_temp = torch.clamp(clean_image + delta, 0, 1)

                true_class_name = self.class_names[true_label]
                llava_perception = self.llava_adapter.ask_what_image_looks_like(
                    adv_image_temp.detach(),
                    candidate_classes=[true_class_name, semantic_anchor_word],
                    semantic_anchor_word=semantic_anchor_word,
                    true_class_name=true_class_name,
                    debug=debug
                )
                llava_target_alignment = llava_perception['target_alignment']
                new_caption = llava_perception['caption']
            adv_image = torch.clamp(clean_image + delta, 0, 1)
            adv_img_embed = self.clip_model.encode_image(adv_image.unsqueeze(0))
            adv_img_embed = adv_img_embed / adv_img_embed.norm(dim=-1, keepdim=True)
            adv_img_embed = adv_img_embed.float()
            feat_3 = self.features['layer_3']
            feat_6 = self.features['layer_6']
            feat_9 = self.features['layer_9']
            current_weights = self.layer_weights.copy()
            if self.adaptive and iteration > 0:
                if pred_idx == true_label:
                    shift = 0.05
                    current_weights[0] = min(0.3, current_weights[0] + shift)
                    current_weights[1] = min(0.3, current_weights[1] + shift)
                    current_weights[2] = min(0.3, current_weights[2] + shift)
                    current_weights[3] = max(0.1, current_weights[3] - 3 * shift)
                else:
                    shift = 0.03
                    current_weights[3] = min(0.7, current_weights[3] + shift)
                    current_weights[0] = max(0.05, current_weights[0] - shift/3)
                    current_weights[1] = max(0.05, current_weights[1] - shift/3)
                    current_weights[2] = max(0.05, current_weights[2] - shift/3)
            # Loss computation with adaptive weights
            total_loss = 0
            # Loss 1: Final embedding alignment
            final_alignment = (adv_img_embed @ current_text_embed.T).squeeze()
            L_final = -final_alignment
            total_loss += current_weights[3] * L_final
            # Loss 2: Multi-level feature alignment
            L_layer3 = self._compute_layer_alignment_loss(feat_3, current_text_embed)
            L_layer6 = self._compute_layer_alignment_loss(feat_6, current_text_embed)
            L_layer9 = self._compute_layer_alignment_loss(feat_9, current_text_embed)
            total_loss += current_weights[0] * L_layer3
            total_loss += current_weights[1] * L_layer6
            total_loss += current_weights[2] * L_layer9
            # Loss 3: Adversarial loss (push away from true class)
            true_class_embed = self.class_embeddings[true_label]
            true_similarity = (adv_img_embed @ true_class_embed.unsqueeze(0).T).squeeze()
            L_adversarial = true_similarity
            # Adaptive lambda based on prediction status
            if pred_idx == true_label:
                lambda_adv = 0.5
            else:
                lambda_adv = 0.1
            # LLaVA adjustment
            llava_adjustment = 1.0 - 0.5 * llava_target_alignment
            final_adv_weight = lambda_adv * llava_adjustment
            final_adv_weight = max(0.01, final_adv_weight)
            if debug and iteration % 10 == 0:
                print(f"Iter {iteration:3d} | Weights: [{current_weights[0]:.2f}, {current_weights[1]:.2f}, {current_weights[2]:.2f}, {current_weights[3]:.2f}] | "
                      f"Pred: {self.class_names[pred_idx] if pred_idx >= 0 else '?'}")
            total_loss += final_adv_weight * L_adversarial
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
        # Final adversarial image
        final_adv_image = torch.clamp(clean_image + delta.detach(), 0, 1)
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
                'semantic_alignment': (final_embed @ current_text_embed.T).item(),
                'final_caption': new_caption,
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
            print(f"  Initialized with L2 norm: {init_delta.norm().item():.4f}")

        return init_delta

    def _compute_layer_alignment_loss(self, layer_features, target_embed):
        if layer_features.dim() > 2:
            pooled = layer_features.mean(dim=1)
        else:
            pooled = layer_features
        pooled = pooled.float()
        pooled_norm = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        target_dim = pooled_norm.shape[-1]
        target_projected = self._project_embedding(target_embed, target_dim)
        target_projected = target_projected.float()
        alignment = (pooled_norm @ target_projected.T).mean()

        return -alignment

    def _project_embedding(self, embed, target_dim):
        embed = embed.float()
        current_dim = embed.shape[-1]

        if current_dim == target_dim:
            return embed
        elif current_dim < target_dim:
            padding = torch.zeros(embed.shape[0], target_dim - current_dim).to(self.device)
            return torch.cat([embed, padding], dim=1)
        else:
            return embed[:, :target_dim]
