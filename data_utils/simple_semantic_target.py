import torch
import numpy as np
from collections import defaultdict, Counter
import json
import os

class SimpleSemanticTargeter:
    def __init__(self, dataset_name, class_names, text_embeddings=None, debug=False):
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.text_embeddings = text_embeddings
        self.debug = debug
        
        # Build semantic distance matrix once
        self.distance_matrix = self._build_distance_matrix()
        
    def _build_distance_matrix(self):
        if self.text_embeddings is None:
            return None
            
        # Compute cosine distances - this is from Dataset itself, no wordnet reference 
        similarity_matrix = torch.matmul(self.text_embeddings, self.text_embeddings.T)
        distance_matrix = 1.0 - similarity_matrix
        
        return distance_matrix
    
    def select_semantic_targets(self, targets, strategy):
        semantic_targets = []
        
        for target_idx in targets:
            true_idx = target_idx.item()
            
            if strategy == "opposite":
                sem_idx = self._find_most_distant_class(true_idx)
            elif strategy == "similar":
                sem_idx = self._find_most_similar_class(true_idx)
            elif strategy == "confusing":
                sem_idx = self._find_moderately_distant_class(true_idx)
            else:
                sem_idx = self._find_random_different_class(true_idx)
            
            semantic_targets.append(sem_idx)
            
            # Debug output (only when debug mode is enabled)
            if self.debug:
                print(f"  {self.class_names[true_idx]} -> {self.class_names[sem_idx]} ({strategy})")
        
        return torch.tensor(semantic_targets, device=targets.device)
    
    def _find_most_distant_class(self, true_idx):
        if self.distance_matrix is None:
            return self._find_random_different_class(true_idx)
        
        distances = self.distance_matrix[true_idx]
        distances[true_idx] = -1  # Exclude self
        
        most_distant_idx = torch.argmax(distances).item()
        return most_distant_idx
    
    def _find_most_similar_class(self, true_idx):
        if self.distance_matrix is None:
            return self._find_random_different_class(true_idx)
        
        distances = self.distance_matrix[true_idx]
        distances[true_idx] = float('inf')  # Exclude self
        
        most_similar_idx = torch.argmin(distances).item()
        return most_similar_idx
    
    def _find_moderately_distant_class(self, true_idx):
        if self.distance_matrix is None:
            return self._find_random_different_class(true_idx)
        
        distances = self.distance_matrix[true_idx]
        distances[true_idx] = -1  # Exclude self
        
        # Find median distance class
        sorted_distances, sorted_indices = torch.sort(distances, descending=False)
        median_idx = len(sorted_distances) // 2
        
        return sorted_indices[median_idx].item()
    
    def _find_random_different_class(self, true_idx):
        available_classes = [i for i in range(len(self.class_names)) if i != true_idx]
        return np.random.choice(available_classes)

def create_semantic_targeter(dataset_name, class_names, text_embeddings=None, debug=False):
    return SimpleSemanticTargeter(dataset_name, class_names, text_embeddings, debug=debug)