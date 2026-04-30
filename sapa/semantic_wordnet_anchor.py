import torch
from modified_clip import clip
from nltk.corpus import wordnet as wn


class WordNetSemanticAnchor:
    def __init__(self, clip_model, device, class_names):
        self.clip_model = clip_model
        self.device = device
        self.class_names = class_names
        self.clip_model.float()
        self.class_embeddings = self._compute_class_embeddings()
    
    def _compute_class_embeddings(self):
        with torch.no_grad():
            texts = [f"a photo of a {name}" for name in self.class_names]
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.float()
            
        return embeddings
    
    def _preprocess_class_name(self, class_name):
        parts = [p.strip() for p in class_name.split(',')]
        
        for part in parts[:]:
            words = part.split()
            if len(words) > 1:
                parts.append(words[-1])
        
        return parts
    
    def get_filtered_anchor(self, true_label, strategy='similar', fallback=True):
        class_name = self.class_names[true_label]
        class_name_variants = self._preprocess_class_name(class_name)

        print(f"  Searching WordNet for: {class_name_variants}")

        # Get synsets
        all_synsets = []
        for variant in class_name_variants:
            synsets = wn.synsets(variant, pos='n')
            all_synsets.extend(synsets)
            if synsets:
                print(f"    Found {len(synsets)} synsets for '{variant}'")

        if not all_synsets:
            print(f"  ✗ No WordNet synsets found")
            return None

        all_synsets = list(set(all_synsets))
        print(f"  Total unique synsets: {len(all_synsets)}")

        # Generate candidates based on strategy
        candidates = set()

        if strategy == 'distant':
            print(f"  Using DISTANT strategy (high-level hypernyms)...")
            for synset in all_synsets:
                # Get hypernyms at multiple levels
                hypernyms = synset.hypernyms()
                for level, hyp in enumerate(hypernyms):
                    for lemma in hyp.lemmas():
                        word = lemma.name().replace('_', ' ')
                        candidates.add(word)
                        print(f"    Level {level+1} hypernym: {word}")
                    if level < 2:  # Limit depth
                        for hyp2 in hyp.hypernyms():
                            for lemma in hyp2.lemmas():
                                word = lemma.name().replace('_', ' ')
                                candidates.add(word)

        else:
            for synset in all_synsets:
                for lemma in synset.lemmas():
                    word = lemma.name().replace('_', ' ')
                    candidates.add(word)

                if strategy in ['similar', 'related']:
                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas():
                            word = lemma.name().replace('_', ' ')
                            candidates.add(word)

                if strategy == 'related':
                    for hyponym in synset.hyponyms()[:3]:
                        for lemma in hyponym.lemmas():
                            word = lemma.name().replace('_', ' ')
                            candidates.add(word)

                    for similar in synset.similar_tos():
                        for lemma in similar.lemmas():
                            word = lemma.name().replace('_', ' ')
                            candidates.add(word)
        for variant in class_name_variants:
            candidates.discard(variant)
        candidates.discard(class_name)
        
        if not candidates:
            print(f"  ✗ No candidate words generated")
            return None
        
        print(f"  Generated {len(candidates)} candidate words")
        print(f"    Sample: {list(candidates)[:5]}...")
        strict_anchors = []

        for word in candidates:
            text = f"a photo of a {word}"
            tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                embed = self.clip_model.encode_text(tokens)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                embed = embed.float()

                similarities = embed @ self.class_embeddings.T
                nearest_idx = similarities.argmax().item()
                nearest_class = self.class_names[nearest_idx]

                if nearest_idx != true_label:
                    true_class_sim = similarities[0, true_label].item()
                    strict_anchors.append({
                        'word': word,
                        'target_class': nearest_class,
                        'target_class_idx': nearest_idx,
                        'similarity': similarities[0, nearest_idx].item(),
                        'true_class_sim': true_class_sim,
                        'distance_from_true': 1.0 - true_class_sim
                    })

        if strict_anchors:
            if strategy == 'distant':
                strict_anchors.sort(key=lambda x: x['distance_from_true'], reverse=True)
                best = strict_anchors[0]
                print(f"  ✓ DISTANT anchor: '{best['word']}' → {best['target_class']} "
                      f"(sim: {best['similarity']:.3f}, dist_from_true: {best['distance_from_true']:.3f})")
            else:
                strict_anchors.sort(key=lambda x: x['similarity'], reverse=True)
                best = strict_anchors[0]
                print(f"  ✓ Strict anchor: '{best['word']}' → {best['target_class']} "
                      f"(sim: {best['similarity']:.3f})")
            return best['word']
        
        # ===== FALLBACK =====
        if fallback:
            print(f"  ⚠ No strict anchors, using fallback...")
            different_anchors = []
            
            for word in candidates:
                text = f"a photo of a {word}"
                tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    embed = self.clip_model.encode_text(tokens)
                    embed = embed / embed.norm(dim=-1, keepdim=True)
                    embed = embed.float()
                    
                    true_class_embed = self.class_embeddings[true_label].unsqueeze(0)
                    true_class_sim = (embed @ true_class_embed.T).item()
                    
                    different_anchors.append({
                        'word': word,
                        'distance': 1.0 - true_class_sim
                    })
            
            if different_anchors:
                different_anchors.sort(key=lambda x: x['distance'], reverse=True)
                fallback = different_anchors[0]
                print(f"  ✓ Fallback anchor: '{fallback['word']}' "
                      f"(distance: {fallback['distance']:.3f})")
                return fallback['word']
        
        print(f"  ✗ No anchor found")
        return None
