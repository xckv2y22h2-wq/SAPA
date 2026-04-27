import torch
import numpy as np
from collections import defaultdict, Counter
import json
import os

class DatasetAdaptiveSemanticTargeter:
    """
    Generalized semantic targeting system that adapts to different datasets
    """
    
    def __init__(self, dataset_name, class_names, text_embeddings=None, cache_dir="./semantic_cache"):
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.text_embeddings = text_embeddings
        self.cache_dir = cache_dir

        print(f"Dataset name: {self.dataset_name}")
        # Dataset-specific configurations
        self.config = self._get_dataset_config()
        
        # Initialize semantic analyzer
        self.semantic_analyzer = None
        self._initialize_semantic_analyzer()
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_dataset_config(self):
        
        configs = {
            # Fine-grained datasets need tighter thresholds
            "cifar100": {
                "similar_threshold": 0.12,
                "confusing_range": (0.15, 0.35),
                "opposite_threshold": 0.6,
                "use_wordnet": True,
                "use_clustering": True,
                "cluster_count": 20
            },
            
            "caltech101": {
                "similar_threshold": 0.15,
                "confusing_range": (0.2, 0.4),
                "opposite_threshold": 0.65,
                "use_wordnet": True,
                "use_clustering": True,
                "cluster_count": 15
            },
            
            # Coarse-grained datasets can use looser thresholds
            "cifar10": {
                "similar_threshold": 0.2,
                "confusing_range": (0.25, 0.5),
                "opposite_threshold": 0.7,
                "use_wordnet": True,
                "use_clustering": False,
                "cluster_count": 5
            },
            
            "imagenet": {
                "similar_threshold": 0.1,
                "confusing_range": (0.15, 0.3),
                "opposite_threshold": 0.5,
                "use_wordnet": True,
                "use_clustering": True,
                "cluster_count": 50
            },
            
            # Default configuration
            "default": {
                "similar_threshold": 0.15,
                "confusing_range": (0.2, 0.4),
                "opposite_threshold": 0.6,
                "use_wordnet": True,
                "use_clustering": True,
                "cluster_count": min(len(self.class_names) // 5, 20)
            }
        }
        
        return configs.get(self.dataset_name.lower(), configs["default"])
    
    def _initialize_semantic_analyzer(self):
        cache_file = os.path.join(self.cache_dir, f"{self.dataset_name}_semantic_analysis.json")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                self.semantic_analyzer = cached_data
                print(f"Loaded cached semantic analysis for {self.dataset_name}")
                return
            except:
                print("Cache corrupted, rebuilding semantic analysis")
        
        print(f"Building semantic analysis for {self.dataset_name} with {len(self.class_names)} classes...")
        
        # Build components in correct order (avoid circular dependencies)
        self.semantic_analyzer = {}
        
        # Step 1: Build basic components
        self.semantic_analyzer["wordnet_relationships"] = self._build_wordnet_relationships()
        self.semantic_analyzer["clip_distances"] = self._build_clip_distances()  
        self.semantic_analyzer["semantic_clusters"] = self._build_semantic_clusters()
        
        # Step 2: Build components that depend on the above
        self.semantic_analyzer["forbidden_pairs"] = self._build_forbidden_pairs()
        self.semantic_analyzer["validated_pairs"] = self._build_validated_pairs()
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.semantic_analyzer, f, indent=2)
            print(f"Cached semantic analysis for future use")
        except Exception as e:
            print(f"Could not cache results: {e}")
    
    def _build_wordnet_relationships(self):
        if not self.config["use_wordnet"]:
            return {}
        
        try:
            from nltk.corpus import wordnet as wn
            relationships = {}
            
            for i, class1 in enumerate(self.class_names):
                relationships[i] = {}
                synsets1 = self._get_synsets(class1)
                
                for j, class2 in enumerate(self.class_names):
                    if i != j:
                        synsets2 = self._get_synsets(class2)
                        rel_info = self._analyze_wordnet_relationship(synsets1, synsets2)
                        if rel_info:
                            relationships[i][j] = rel_info
            
            return relationships
        except ImportError:
            print("WordNet not available")
            return {}
    
    def _build_clip_distances(self):
        if self.text_embeddings is None:
            return {}
        
        distances = {}
        similarity_matrix = torch.matmul(self.text_embeddings, self.text_embeddings.T)
        distance_matrix = 1.0 - similarity_matrix
        
        for i in range(len(self.class_names)):
            distances[i] = {}
            for j in range(len(self.class_names)):
                if i != j and i < distance_matrix.size(0) and j < distance_matrix.size(1):
                    distances[i][j] = distance_matrix[i, j].item()
        
        return distances
    
    def _build_semantic_clusters(self):
        if not self.config["use_clustering"] or self.text_embeddings is None:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            
            embeddings_np = self.text_embeddings.cpu().numpy()
            n_clusters = min(self.config["cluster_count"], len(self.class_names) // 2)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_np)
            
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[int(label)].append(i)
            
            return dict(clusters)
        except ImportError:
            print("scikit-learn not available for clustering")
            return {}
    
    def _build_validated_pairs(self):
        validated = {"similar": [], "confusing": [], "opposite": []}
        
        for i in range(len(self.class_names)):
            for j in range(i + 1, len(self.class_names)):
                validation_score, relationship_type = self._validate_pair(i, j)
                
                if validation_score > 0.7:  # High confidence
                    validated[relationship_type].append((i, j, validation_score))
        
        # Sort by confidence
        for rel_type in validated:
            validated[rel_type].sort(key=lambda x: x[2], reverse=True)
        
        return validated
    
    def _build_forbidden_pairs(self):
        forbidden = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and self._is_forbidden_pair(i, j):
                    forbidden.append((i, j))
        
        return forbidden
    
    def _get_synsets(self, class_name):
        try:
            from nltk.corpus import wordnet as wn
            
            synsets = []
            clean_name = self._clean_class_name(class_name)
            
            # Direct lookup
            synsets.extend(wn.synsets(clean_name, pos=wn.NOUN))
            
            # Individual words
            if not synsets:
                words = clean_name.split()
                for word in words:
                    synsets.extend(wn.synsets(word, pos=wn.NOUN))
            
            # Common variations
            if not synsets:
                variations = [
                    clean_name.replace('_', ''),
                    clean_name.replace(' ', ''),
                    clean_name.replace('cell', '').replace('phone', 'telephone').strip()
                ]
                for variation in variations:
                    if variation and variation != clean_name:
                        synsets.extend(wn.synsets(variation, pos=wn.NOUN))
            
            return synsets[:3]  # Limit to top 3 synsets
        except:
            return []
    
    def _analyze_wordnet_relationship(self, synsets1, synsets2):
        if not synsets1 or not synsets2:
            return None
        
        best_relationship = None
        best_similarity = 0
        
        for s1 in synsets1:
            for s2 in synsets2:
                similarity = s1.wup_similarity(s2)
                if similarity and similarity > best_similarity:
                    best_similarity = similarity
                    best_relationship = self._classify_wordnet_relationship(s1, s2, similarity)
        
        if best_relationship:
            return {
                "type": best_relationship,
                "similarity": best_similarity,
                "confidence": min(best_similarity * 2, 1.0)  # Scale confidence
            }
        
        return None
    
    def _classify_wordnet_relationship(self, synset1, synset2, similarity):
        
        # Check direct relationships
        if synset2 in synset1.hyponyms():
            return "hyponym"
        elif synset2 in synset1.hypernyms():
            return "hypernym"
        elif any(synset2 == ant.synset() for lemma in synset1.lemmas() for ant in lemma.antonyms()):
            return "antonym"
        
        # Check sibling relationship
        for hypernym in synset1.hypernyms():
            if synset2 in hypernym.hyponyms():
                return "sibling"
        
        # Classify by similarity level
        if similarity > 0.8:
            return "very_similar"
        elif similarity > 0.6:
            return "similar"
        elif similarity > 0.4:
            return "related"
        elif similarity < 0.2:
            return "opposite"
        else:
            return "distant"
    
    def _validate_pair(self, i, j):
        class1, class2 = self.class_names[i], self.class_names[j]
        
        # WordNet validation
        wordnet_score = 0
        wordnet_type = "unknown"
        if i in self.semantic_analyzer.get("wordnet_relationships", {}):
            if j in self.semantic_analyzer["wordnet_relationships"][i]:
                rel_info = self.semantic_analyzer["wordnet_relationships"][i][j]
                wordnet_score = rel_info["confidence"]
                wordnet_type = rel_info["type"]
        
        # CLIP validation
        clip_score = 0
        clip_type = "unknown"
        if i in self.semantic_analyzer.get("clip_distances", {}):
            if j in self.semantic_analyzer["clip_distances"][i]:
                distance = self.semantic_analyzer["clip_distances"][i][j]
                clip_score, clip_type = self._classify_clip_distance(distance)
        
        # Cluster validation
        cluster_score = 0
        if self.semantic_analyzer.get("semantic_clusters"):
            same_cluster = self._in_same_cluster(i, j)
            cluster_score = 0.8 if same_cluster else 0.2
        
        # Manual validation for known cases
        manual_score, manual_type = self._manual_validation(class1, class2)
        
        # Combine scores with weights
        total_score = (
            0.4 * wordnet_score +
            0.3 * clip_score +
            0.2 * cluster_score +
            0.1 * manual_score
        )
        
        # Determine relationship type
        if wordnet_type in ["similar", "sibling", "hyponym", "hypernym"]:
            relationship_type = "similar"
        elif wordnet_type in ["antonym", "opposite"]:
            relationship_type = "opposite"
        elif clip_type in ["similar"]:
            relationship_type = "similar"
        elif clip_type in ["opposite"]:
            relationship_type = "opposite"
        else:
            relationship_type = "confusing"
        
        return total_score, relationship_type
    
    def _classify_clip_distance(self, distance):
        if distance < self.config["similar_threshold"]:
            return 0.9, "similar"
        elif distance > self.config["opposite_threshold"]:
            return 0.8, "opposite"
        elif self.config["confusing_range"][0] <= distance <= self.config["confusing_range"][1]:
            return 0.7, "confusing"
        else:
            return 0.3, "distant"
    
    def _in_same_cluster(self, i, j):
        clusters = self.semantic_analyzer.get("semantic_clusters", {})
        for cluster_classes in clusters.values():
            if i in cluster_classes and j in cluster_classes:
                return True
        return False
    
    def _manual_validation(self, class1, class2):
        # Common similar pairs
        similar_patterns = [
            (["tiger", "lion", "leopard"], ["cat", "wild cat", "feline"]),
            (["cup", "mug", "glass"], ["bowl", "container", "vessel"]),
            (["car", "automobile"], ["truck", "bus", "vehicle"]),
            (["dog", "puppy"], ["wolf", "canine"]),
            (["tree", "oak", "pine"], ["plant", "maple", "birch"]),
        ]
        
        # Common opposite pairs
        opposite_patterns = [
            (["animal", "cat", "dog"], ["vehicle", "car", "plane"]),
            (["living", "animal", "plant"], ["object", "tool", "machine"]),
            (["natural", "tree", "rock"], ["artificial", "computer", "phone"]),
        ]
        
        class1_lower = class1.lower()
        class2_lower = class2.lower()
        
        # Check similar patterns
        for group1, group2 in similar_patterns:
            if (any(p in class1_lower for p in group1) and any(p in class2_lower for p in group2)) or \
               (any(p in class1_lower for p in group2) and any(p in class2_lower for p in group1)):
                return 0.8, "similar"
        
        # Check opposite patterns
        for group1, group2 in opposite_patterns:
            if (any(p in class1_lower for p in group1) and any(p in class2_lower for p in group2)) or \
               (any(p in class1_lower for p in group2) and any(p in class2_lower for p in group1)):
                return 0.8, "opposite"
        
        return 0.1, "unknown"
    
    def _is_forbidden_pair(self, i, j):
        class1, class2 = self.class_names[i], self.class_names[j]
        
        # Define clearly incompatible pairs
        incompatible_groups = [
            (["cloud", "sky", "weather"], ["fish", "animal", "living"]),
            (["tiger", "lion", "bear", "wolf"], ["fish", "aquarium", "water"]),
            (["abstract", "concept"], ["concrete", "object"]),
        ]
        
        class1_lower = class1.lower()
        class2_lower = class2.lower()
        
        for group1, group2 in incompatible_groups:
            if (any(p in class1_lower for p in group1) and any(p in class2_lower for p in group2)) or \
               (any(p in class1_lower for p in group2) and any(p in class2_lower for p in group1)):
                return True
        
        return False
    
    def _clean_class_name(self, class_name):
        clean = class_name.replace('_', ' ').replace('-', ' ').strip().lower()
        return clean
    
    def select_semantic_targets(self, targets, strategy, used_targets=None):
        if used_targets is None:
            used_targets = set()
        
        semantic_targets = []
        
        for i, target_idx in enumerate(targets):
            true_idx = target_idx.item()
            
            # Find best semantic target
            semantic_target = self._find_best_semantic_target(
                true_idx, strategy, used_targets
            )
            
            if semantic_target is None:
                # Final fallback
                available = [j for j in range(len(self.class_names)) 
                           if j != true_idx and j not in used_targets]
                semantic_target = np.random.choice(available) if available else (true_idx + 1) % len(self.class_names)
            
            semantic_targets.append(semantic_target)
            used_targets.add(semantic_target)
            
            # Debug output
            if i < 8:
                true_class = self.class_names[true_idx]
                sem_class = self.class_names[semantic_target]
                reasoning = self._explain_choice(true_idx, semantic_target, strategy)
                print(f"  {true_class} -> {sem_class} ({reasoning})")
        
        return torch.tensor(semantic_targets, device=targets.device)
    
    def _find_best_semantic_target(self, true_idx, strategy, used_targets):
        
        # Get validated pairs for this strategy
        validated_pairs = self.semantic_analyzer.get("validated_pairs", {}).get(strategy, [])
        
        # Priority 1: Use validated pairs
        for i, j, confidence in validated_pairs:
            if i == true_idx and j not in used_targets:
                return j
            elif j == true_idx and i not in used_targets:
                return i
        
        # Priority 2: Use WordNet relationships
        if strategy == "similar":
            return self._find_similar_target(true_idx, used_targets)
        elif strategy == "opposite":
            return self._find_opposite_target(true_idx, used_targets)
        elif strategy == "confusing":
            return self._find_confusing_target(true_idx, used_targets)
        
        return None
    
    def _find_similar_target(self, true_idx, used_targets):
        candidates = []
        
        # WordNet-based candidates
        wordnet_rels = self.semantic_analyzer.get("wordnet_relationships", {}).get(true_idx, {})
        for target_idx, rel_info in wordnet_rels.items():
            if target_idx not in used_targets and rel_info["type"] in ["similar", "sibling", "hyponym", "hypernym"]:
                candidates.append((target_idx, rel_info["confidence"], "wordnet"))
        
        # CLIP-based candidates
        clip_distances = self.semantic_analyzer.get("clip_distances", {}).get(true_idx, {})
        for target_idx, distance in clip_distances.items():
            if target_idx not in used_targets and distance < self.config["similar_threshold"]:
                # Check if not forbidden
                if (true_idx, target_idx) not in self.semantic_analyzer.get("forbidden_pairs", []):
                    score = 1.0 - distance
                    candidates.append((target_idx, score, "clip"))
        
        # Cluster-based candidates
        if self.semantic_analyzer.get("semantic_clusters"):
            for cluster_classes in self.semantic_analyzer["semantic_clusters"].values():
                if true_idx in cluster_classes:
                    for target_idx in cluster_classes:
                        if target_idx != true_idx and target_idx not in used_targets:
                            candidates.append((target_idx, 0.7, "cluster"))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _find_opposite_target(self, true_idx, used_targets):
        candidates = []
        
        # WordNet antonyms
        wordnet_rels = self.semantic_analyzer.get("wordnet_relationships", {}).get(true_idx, {})
        for target_idx, rel_info in wordnet_rels.items():
            if target_idx not in used_targets and rel_info["type"] in ["antonym", "opposite"]:
                candidates.append((target_idx, rel_info["confidence"], "wordnet"))
        
        # CLIP distance-based
        clip_distances = self.semantic_analyzer.get("clip_distances", {}).get(true_idx, {})
        for target_idx, distance in clip_distances.items():
            if target_idx not in used_targets and distance > self.config["opposite_threshold"]:
                candidates.append((target_idx, distance, "clip"))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _find_confusing_target(self, true_idx, used_targets):
        candidates = []
        
        # WordNet related concepts
        wordnet_rels = self.semantic_analyzer.get("wordnet_relationships", {}).get(true_idx, {})
        for target_idx, rel_info in wordnet_rels.items():
            if target_idx not in used_targets and rel_info["type"] in ["related", "distant"]:
                candidates.append((target_idx, rel_info["confidence"], "wordnet"))
        
        # CLIP moderate distances
        clip_distances = self.semantic_analyzer.get("clip_distances", {}).get(true_idx, {})
        for target_idx, distance in clip_distances.items():
            if (target_idx not in used_targets and 
                self.config["confusing_range"][0] <= distance <= self.config["confusing_range"][1]):
                if (true_idx, target_idx) not in self.semantic_analyzer.get("forbidden_pairs", []):
                    score = 1.0 - abs(distance - np.mean(self.config["confusing_range"]))
                    candidates.append((target_idx, score, "clip"))
        
        if candidates:
            # Add randomness for confusing strategy
            top_candidates = [c for c in candidates if c[1] > 0.5]
            if top_candidates:
                return np.random.choice([c[0] for c in top_candidates])
            else:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        
        return None
    
    def _explain_choice(self, true_idx, semantic_idx, strategy):
        # Check validation source
        if true_idx in self.semantic_analyzer.get("wordnet_relationships", {}):
            if semantic_idx in self.semantic_analyzer["wordnet_relationships"][true_idx]:
                rel_info = self.semantic_analyzer["wordnet_relationships"][true_idx][semantic_idx]
                return f"WordNet {rel_info['type']}, conf={rel_info['confidence']:.2f}"
        
        if true_idx in self.semantic_analyzer.get("clip_distances", {}):
            if semantic_idx in self.semantic_analyzer["clip_distances"][true_idx]:
                distance = self.semantic_analyzer["clip_distances"][true_idx][semantic_idx]
                return f"CLIP distance={distance:.3f}"
        
        return f"{strategy} fallback"


def create_semantic_targeter(dataset_name, class_names, text_embeddings=None):
    return DatasetAdaptiveSemanticTargeter(dataset_name, class_names, text_embeddings)


# Usage example:
def example_usage():
    
    # Example for CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create targeter (will automatically configure for CIFAR-10)
    targeter = create_semantic_targeter("cifar10", cifar10_classes)
    
    # Select semantic targets
    targets = torch.tensor([0, 1, 2, 3])  # airplane, automobile, bird, cat
    semantic_targets = targeter.select_semantic_targets(targets, "similar")
    
    print("CIFAR-10 similar targets:")
    for i, (true_idx, sem_idx) in enumerate(zip(targets, semantic_targets)):
        print(f"{cifar10_classes[true_idx]} -> {cifar10_classes[sem_idx]}")


# Integration function for your existing code:
def get_adaptive_semantic_targets(targets, dataset_name, class_names, text_embeddings, strategy):
    targeter = create_semantic_targeter(dataset_name, class_names, text_embeddings)
    return targeter.select_semantic_targets(targets, strategy)