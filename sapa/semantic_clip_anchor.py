"""
CLIP-based Semantic Anchor Selection

Instead of relying on WordNet (which has low coverage for many visual concepts),
this module uses CLIP embeddings to find semantically related classes that can
serve as effective attack targets.

Strategy:
1. Compute CLIP embeddings for all class names
2. For each source class, find other classes with moderate similarity
3. Select anchors that are similar enough to be confusable but different enough to be misclassified
"""

import torch
from modified_clip import clip
from typing import List, Optional, Tuple, Dict


class CLIPSemanticAnchor:
    """
    CLIP-based Semantic Anchor Word Generation
    
    Uses CLIP text embeddings to find semantically related classes
    that can serve as effective adversarial targets.
    """
    
    def __init__(
        self,
        clip_model,
        device,
        class_names: List[str],
        prompt_templates: Optional[List[str]] = None,
        debug: bool = False
    ):
        self.clip_model = clip_model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.debug = debug
        
        # Default prompt templates
        if prompt_templates is None:
            self.prompt_templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}",
                "a photograph of a {}",
            ]
        else:
            self.prompt_templates = prompt_templates
        
        # Pre-compute class embeddings
        self._precompute_class_embeddings()
        
        # Pre-compute similarity matrix
        self._precompute_similarity_matrix()
        
        # Cache for anchor selections
        self.anchor_cache: Dict[int, Tuple[str, int, float]] = {}
    
    def _precompute_class_embeddings(self):
        print("  Pre-computing CLIP class embeddings...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for class_name in self.class_names:
                # Ensemble over multiple prompts
                class_embeds = []
                for template in self.prompt_templates:
                    text = template.format(class_name)
                    tokens = clip.tokenize([text]).to(self.device)
                    embed = self.clip_model.encode_text(tokens)
                    embed = embed / embed.norm(dim=-1, keepdim=True)
                    class_embeds.append(embed)
                
                # Average embeddings from all templates
                avg_embed = torch.stack(class_embeds).mean(dim=0)
                avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)
                all_embeddings.append(avg_embed)
            
            self.class_embeddings = torch.cat(all_embeddings, dim=0).float()
        
        print(f"  ✓ Computed embeddings for {self.num_classes} classes")
    
    def _precompute_similarity_matrix(self):
        print("  Pre-computing class similarity matrix...")
        
        with torch.no_grad():
            # Compute cosine similarity matrix
            self.similarity_matrix = self.class_embeddings @ self.class_embeddings.T
            
            # Set diagonal to -inf to exclude self-similarity
            self.similarity_matrix.fill_diagonal_(-float('inf'))
        
        print(f"  ✓ Computed {self.num_classes}x{self.num_classes} similarity matrix")
    
    def find_semantic_anchor(
        self,
        class_idx: int,
        strategy: str = "similar",
        similarity_range: Tuple[float, float] = (0.5, 0.85),
        top_k: int = 5
    ) -> Optional[str]:
        # Check cache
        if class_idx in self.anchor_cache:
            anchor_name, anchor_idx, similarity = self.anchor_cache[class_idx]
            return anchor_name
        
        class_name = self.class_names[class_idx]
        min_sim, max_sim = similarity_range
        
        # Get similarities to all other classes
        similarities = self.similarity_matrix[class_idx].clone()
        
        # Find valid candidates within similarity range
        valid_mask = (similarities >= min_sim) & (similarities <= max_sim)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Fallback: expand range
            if self.debug:
                print(f"    No anchors in range [{min_sim:.2f}, {max_sim:.2f}], expanding...")
            
            # Try progressively wider ranges
            for new_min, new_max in [(0.3, 0.9), (0.2, 0.95), (0.1, 0.99)]:
                valid_mask = (similarities >= new_min) & (similarities <= new_max)
                valid_indices = torch.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    min_sim, max_sim = new_min, new_max
                    break
        
        if len(valid_indices) == 0:
            # Ultimate fallback: just pick the most similar class
            best_idx = similarities.argmax().item()
            anchor_name = self.class_names[best_idx]
            best_sim = similarities[best_idx].item()
            if self.debug:
                print(f"    ⚠ Fallback anchor: '{anchor_name}' (sim: {best_sim:.3f})")
            self.anchor_cache[class_idx] = (anchor_name, best_idx, best_sim)
            return anchor_name
        
        valid_sims = similarities[valid_indices]
        
        # Select based on strategy
        if strategy == "similar":
            # Pick the most similar valid class
            best_local_idx = valid_sims.argmax().item()
            anchor_idx = valid_indices[best_local_idx].item()

        elif strategy == "random_similar":
            # Randomly select from top-k most similar
            k = min(top_k, len(valid_indices))
            top_k_local = valid_sims.topk(k).indices
            rand_idx = torch.randint(0, k, (1,)).item()
            anchor_idx = valid_indices[top_k_local[rand_idx]].item()

        elif strategy == "boundary":
            # Pick class closest to decision boundary (similarity ~0.5)
            boundary_dist = (valid_sims - 0.5).abs()
            best_local_idx = boundary_dist.argmin().item()
            anchor_idx = valid_indices[best_local_idx].item()

        elif strategy == "dissimilar":
            # Pick the least similar valid class (for diversity)
            best_local_idx = valid_sims.argmin().item()
            anchor_idx = valid_indices[best_local_idx].item()

        elif strategy == "distant":
            # DISTANT: Pick the LOWEST similarity valid class
            # This selects a class far from true class in embedding space
            # For better adversarial success on robust models
            best_local_idx = valid_sims.argmin().item()
            anchor_idx = valid_indices[best_local_idx].item()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        anchor_name = self.class_names[anchor_idx]
        anchor_sim = similarities[anchor_idx].item()
        
        if self.debug:
            print(f"    ✓ CLIP anchor: '{anchor_name}' (sim: {anchor_sim:.3f})")
        
        # Cache result
        self.anchor_cache[class_idx] = (anchor_name, anchor_idx, anchor_sim)
        
        return anchor_name
    
    def find_top_k_anchors(
        self,
        class_idx: int,
        k: int = 5,
        similarity_range: Tuple[float, float] = (0.4, 0.9)
    ) -> List[Tuple[str, int, float]]:
        min_sim, max_sim = similarity_range
        similarities = self.similarity_matrix[class_idx].clone()
        
        # Mask out classes outside range
        mask = (similarities >= min_sim) & (similarities <= max_sim)
        similarities[~mask] = -float('inf')
        
        # Get top-k
        top_k_sims, top_k_indices = similarities.topk(min(k, mask.sum().item()))
        
        results = []
        for sim, idx in zip(top_k_sims.tolist(), top_k_indices.tolist()):
            if sim > -float('inf'):
                results.append((self.class_names[idx], idx, sim))
        
        return results
    
    def get_anchor_embedding(self, anchor_word: str) -> torch.Tensor:
        # Check if it's a class name (use pre-computed embedding)
        if anchor_word in self.class_names:
            idx = self.class_names.index(anchor_word)
            return self.class_embeddings[idx:idx+1]
        
        # Otherwise compute embedding on the fly
        with torch.no_grad():
            embeds = []
            for template in self.prompt_templates:
                text = template.format(anchor_word)
                tokens = clip.tokenize([text]).to(self.device)
                embed = self.clip_model.encode_text(tokens)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                embeds.append(embed)
            
            avg_embed = torch.stack(embeds).mean(dim=0)
            avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)
        
        return avg_embed.float()
    
    def get_similarity(self, class_idx1: int, class_idx2: int) -> float:
        return self.similarity_matrix[class_idx1, class_idx2].item()
    
    def get_class_embedding(self, class_idx: int) -> torch.Tensor:
        return self.class_embeddings[class_idx:class_idx+1]
    
    def print_similarity_stats(self):
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(self.similarity_matrix), diagonal=1).bool()
        valid_sims = self.similarity_matrix.clone()
        valid_sims.fill_diagonal_(0)
        pairwise_sims = valid_sims[mask]
        
        print("\nClass Similarity Statistics:")
        print(f"  Min similarity:  {pairwise_sims.min():.3f}")
        print(f"  Max similarity:  {pairwise_sims.max():.3f}")
        print(f"  Mean similarity: {pairwise_sims.mean():.3f}")
        print(f"  Std similarity:  {pairwise_sims.std():.3f}")
        
        # Count pairs in different ranges
        ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        print("\n  Similarity distribution:")
        for low, high in ranges:
            count = ((pairwise_sims >= low) & (pairwise_sims < high)).sum().item()
            pct = count / len(pairwise_sims) * 100
            print(f"    [{low:.1f}, {high:.1f}): {count:6d} ({pct:5.1f}%)")
    
    def find_most_confusable_pairs(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        # Get upper triangle
        mask = torch.triu(torch.ones_like(self.similarity_matrix), diagonal=1).bool()
        valid_sims = self.similarity_matrix.clone()
        valid_sims[~mask] = -float('inf')
        
        # Flatten and get top-n
        flat_sims = valid_sims.flatten()
        top_vals, top_indices = flat_sims.topk(top_n)
        
        pairs = []
        for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
            i = idx // self.num_classes
            j = idx % self.num_classes
            pairs.append((self.class_names[i], self.class_names[j], val))
        
        return pairs