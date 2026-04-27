import torch
from utils import one_hot_embedding
from models.model import *
import torch.nn.functional as F
import functools
from autoattack import AutoAttack
import numpy as np
from scipy.stats import spearmanr
from collections import Counter
import json 
from scipy.stats import entropy
import copy
from typing import List, Optional, Tuple, Dict, Any, Union
import difflib
from models.model import multiGPU_CLIP_image_logits
import gc 


# Make sure these imports are correct for your setup
try:
    from utils import refine_classname
    from data_utils.dataset_adaptive_semantic_target import DatasetAdaptiveSemanticTargeter
    from data_utils.simple_semantic_target import SimpleSemanticTargeter
    from rich_description_generator import RichDescriptionGenerator, generate_rich_target_prompt_enhanced
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the required modules are available in your Python path.")
    exit(1)
try:
    # Detectron2 imports
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    import cv2
    DETECTRON2_AVAILABLE = True
    print("Detectron2 available")
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Detectron2 not available - using fallback semantic selection")

try:
    import nltk
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. WordNet semantic features will be disabled.")


lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
_description_generator = None

def get_description_generator():
    global _description_generator
    if _description_generator is None:
        _description_generator = RichDescriptionGenerator()
    return _description_generator

class SemanticControlEvaluator:
    """
    Evaluates semantic control capabilities of adversarial attacks using three metrics:
    - SCI: Semantic Control Index - measures variance in achieving different semantic goals
    - SDS: Semantic Directability Score - measures precision in hitting intended targets
    - SPI: Semantic Predictability Index - measures predictability of semantic patterns
    """
    
    def __init__(self, model, text_tokens, class_names, device='cuda'):
        self.model = model
        self.text_tokens = text_tokens
        self.class_names = class_names
        self.device = device
        self.text_features = self._extract_text_features()
        
    def _extract_text_features(self):
        try:
            with torch.no_grad():
                if hasattr(self.model, 'encode_text'):
                    text_features = self.model.encode_text(self.text_tokens)
                    return F.normalize(text_features, dim=-1)
                else:
                    # Manual extraction for models without direct encode_text
                    x = self.text_tokens
                    x = self.model.token_embedding(x)
                    x = x + self.model.positional_embedding
                    x = x.permute(1, 0, 2)
                    x = self.model.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = self.model.ln_final(x)
                    text_features = x[torch.arange(x.shape[0]), self.text_tokens.argmax(dim=-1)] @ self.model.text_projection
                    return F.normalize(text_features, dim=-1)
        except Exception as e:
            print(f"Warning: Could not extract text features: {e}")
            return None
    
    def categorize_semantic_relationship(self, true_class_idx, predicted_class_idx, 
                                       similar_threshold=0.6, opposite_threshold=0.3):
        if self.text_features is None:
            return 'unknown'
            
        if (true_class_idx >= self.text_features.size(0) or 
            predicted_class_idx >= self.text_features.size(0)):
            return 'unknown'
            
        similarity = torch.cosine_similarity(
            self.text_features[true_class_idx:true_class_idx+1],
            self.text_features[predicted_class_idx:predicted_class_idx+1]
        ).item()
        
        if similarity > similar_threshold:
            return 'similar'
        elif similarity < opposite_threshold:
            return 'opposite'
        else:
            return 'confusing'
    
    def run_multi_strategy_attack(self, attack_fn, attack_kwargs_template, test_loader, 
                                 strategies=['similar', 'opposite', 'confusing'], max_batches=5):
        multi_strategy_results = {}
        
        # Check if this attack supports semantic strategies
        attack_name = attack_fn.__name__ if hasattr(attack_fn, '__name__') else str(attack_fn)
        supports_semantic_strategy = 'sapa' in attack_name.lower() or 'semantic' in attack_name.lower()
        
        if not supports_semantic_strategy:
            print(f"Attack {attack_name} doesn't support semantic strategies. Running same attack multiple times to measure natural semantic patterns...")
        
        for strategy in strategies:
            print(f"Running attack with '{strategy}' strategy...")
            
            all_true_labels = []
            all_predicted_labels = []
            
            self.model.eval()
            
            for batch_idx, (images, targets) in enumerate(test_loader):
                if batch_idx >= max_batches:
                    break
                    
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Update attack kwargs for current strategy and batch
                attack_kwargs = copy.deepcopy(attack_kwargs_template)
                attack_kwargs.update({
                    'X': images,
                    'target': targets,
                    'text_tokens': self.text_tokens,
                })
                
                # Only add semantic_strategy for attacks that support it
                if supports_semantic_strategy:
                    attack_kwargs['semantic_strategy'] = strategy
                
                try:
                    # Run attack - handle different return formats
                    attack_result = attack_fn(**attack_kwargs)
                    if isinstance(attack_result, tuple):
                        delta, _ = attack_result  # Some attacks return (delta, semantic_targets)
                    else:
                        delta = attack_result
                    
                    # Generate adversarial examples
                    adv_images = torch.clamp(images + delta, 0, 1)
                    
                    # Get predictions on adversarial examples
                    with torch.no_grad():
                        processed_adv = clip_img_preprocessing(adv_images)
                        
                        # Handle different model interfaces
                        if 'prompter' in attack_kwargs and attack_kwargs['prompter'] is not None:
                            prompted_adv = attack_kwargs['prompter'](processed_adv)
                        else:
                            prompted_adv = processed_adv
                            
                        prompt_token = None
                        if 'add_prompter' in attack_kwargs and attack_kwargs['add_prompter'] is not None:
                            prompt_token = attack_kwargs['add_prompter']()
                        
                        if 'prompt_learner' in attack_kwargs and attack_kwargs['prompt_learner'] is not None:
                            from models.model import multiGPU_CLIP_Text_Prompt_Tuning
                            output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                                self.model, prompted_adv, self.text_tokens, prompt_token, 
                                attack_kwargs['prompt_learner'])
                        else:
                            from models.model import multiGPU_CLIP
                            output, _ = multiGPU_CLIP(self.model, prompted_adv, self.text_tokens, prompt_token)
                        
                        predicted_labels = output.argmax(dim=1)
                        
                        all_true_labels.extend(targets.cpu())
                        all_predicted_labels.extend(predicted_labels.cpu())
                
                except Exception as e:
                    print(f"Error in batch {batch_idx} with strategy {strategy}: {e}")
                    continue
            
            if all_true_labels and all_predicted_labels:
                multi_strategy_results[strategy] = (
                    torch.stack(all_true_labels),
                    torch.stack(all_predicted_labels)
                )
                print(f"Strategy '{strategy}': Collected {len(all_true_labels)} samples")
            else:
                print(f"Warning: No valid results for strategy '{strategy}'")
        
        return multi_strategy_results
    
    def compute_semantic_control_index(self, multi_strategy_results):
        if not multi_strategy_results or len(multi_strategy_results) < 2:
            return 0.0
        
        strategy_success_rates = {}
        
        for strategy, (true_labels, predicted_labels) in multi_strategy_results.items():
            if strategy not in ['similar', 'opposite', 'confusing']:
                continue
                
            successful_attacks = 0
            successful_targeting = 0
            
            for true_label, pred_label in zip(true_labels, predicted_labels):
                if pred_label != true_label:  # Attack succeeded
                    successful_attacks += 1
                    
                    # Check if attack achieved intended semantic relationship
                    actual_relationship = self.categorize_semantic_relationship(
                        true_label.item(), pred_label.item()
                    )
                    
                    if actual_relationship == strategy:
                        successful_targeting += 1
            
            # Success rate: fraction of successful attacks that hit intended semantic relationship
            if successful_attacks > 0:
                strategy_success_rates[strategy] = successful_targeting / successful_attacks
            else:
                strategy_success_rates[strategy] = 0.0
        
        if len(strategy_success_rates) < 2:
            return 0.0
        
        # SCI is variance in success rates across strategies
        # High variance = good control (can achieve different semantic goals)
        # Low variance = poor control (same behavior regardless of intended goal)
        success_values = list(strategy_success_rates.values())
        sci_score = np.var(success_values)
        
        return sci_score
    
    def generate_intended_targets(self, true_labels, strategy, k=3):
        if self.text_features is None:
            # Random fallback
            num_classes = len(self.class_names)
            intended_targets = []
            for true_label in true_labels:
                candidates = list(range(num_classes))
                candidates.remove(true_label.item())
                intended_targets.append(np.random.choice(candidates))
            return torch.tensor(intended_targets, device=true_labels.device)
        
        intended_targets = []
        
        for true_label in true_labels:
            true_idx = true_label.item()
            
            if true_idx >= self.text_features.size(0):
                intended_targets.append((true_idx + 1) % len(self.class_names))
                continue
            
            # Calculate similarities to all other classes
            similarities = torch.cosine_similarity(
                self.text_features[true_idx:true_idx+1],
                self.text_features
            )
            similarities[true_idx] = -2.0  # Mask out true class
            
            if strategy == 'similar':
                # Select from most similar classes
                _, top_indices = torch.topk(similarities, k=min(k, len(similarities)-1))
                intended_target = top_indices[np.random.randint(0, len(top_indices))].item()
                
            elif strategy == 'opposite':
                # Select from least similar classes
                _, bottom_indices = torch.topk(similarities, k=min(k, len(similarities)-1), largest=False)
                intended_target = bottom_indices[np.random.randint(0, len(bottom_indices))].item()
                
            elif strategy == 'confusing':
                # Select from middle similarity range
                sorted_sims, sorted_indices = torch.sort(similarities, descending=True)
                mid_start = len(sorted_indices) // 3
                mid_end = 2 * len(sorted_indices) // 3
                mid_candidates = sorted_indices[mid_start:mid_end]
                
                if len(mid_candidates) > 0:
                    intended_target = mid_candidates[np.random.randint(0, len(mid_candidates))].item()
                else:
                    intended_target = sorted_indices[len(sorted_indices)//2].item()
            else:
                # Random fallback
                candidates = list(range(len(self.class_names)))
                candidates.remove(true_idx)
                intended_target = np.random.choice(candidates)
            
            intended_targets.append(intended_target)
        
        return torch.tensor(intended_targets, device=true_labels.device)
    
    def compute_semantic_directability_score(self, true_labels, predicted_labels, 
                                           intended_targets, strategy):
        total_attacks = 0
        precise_hits = 0
        
        for true_label, pred_label, intended_target in zip(true_labels, predicted_labels, intended_targets):
            if pred_label != true_label:  # Successful attack
                total_attacks += 1
                
                # Check if hit the specific intended target
                if pred_label.item() == intended_target.item():
                    precise_hits += 1
                else:
                    # Partial credit: check if achieved intended semantic relationship type
                    actual_relationship = self.categorize_semantic_relationship(
                        true_label.item(), pred_label.item()
                    )
                    if actual_relationship == strategy:
                        precise_hits += 0.5  # Partial credit for right type, wrong specific target
        
        return precise_hits / total_attacks if total_attacks > 0 else 0.0
    
    def compute_semantic_predictability_index(self, true_labels, predicted_labels):
        semantic_patterns = []
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if pred_label != true_label:  # Successful attack
                relationship = self.categorize_semantic_relationship(
                    true_label.item(), pred_label.item()
                )
                if relationship != 'unknown':
                    semantic_patterns.append(relationship)
        
        if len(semantic_patterns) == 0:
            return 0.0
        
        # Calculate entropy of semantic patterns
        pattern_counts = Counter(semantic_patterns)
        total_patterns = len(semantic_patterns)
        probabilities = [count / total_patterns for count in pattern_counts.values()]
        
        # Compute entropy (lower entropy = more predictable)
        if len(probabilities) == 1:
            # Perfect predictability
            return 1.0
        
        pattern_entropy = entropy(probabilities, base=2)
        
        # Convert to predictability index (inverse of entropy, normalized)
        max_entropy = np.log2(3)  # Maximum entropy for 3 categories
        predictability = 1 - (pattern_entropy / max_entropy)
        
        return max(0.0, predictability)
    
    def evaluate_semantic_control(self, attack_fn, attack_kwargs_template, test_loader, 
                                 strategies=['similar', 'opposite', 'confusing'], max_batches=5):
        print("Starting semantic control evaluation...")
        
        # Step 1: Run multi-strategy attack
        multi_strategy_results = self.run_multi_strategy_attack(
            attack_fn, attack_kwargs_template, test_loader, strategies, max_batches
        )
        # ADD DIAGNOSTIC STEP HERE
        if multi_strategy_results:
            print("\n" + "="*60)
            print("RUNNING DIAGNOSTIC ON SEMANTIC CATEGORIZATION")
            print("="*60)
            
            # Use the first available strategy's results for diagnosis
            for strategy, (true_labels, pred_labels) in multi_strategy_results.items():
                print(f"\nDiagnosing strategy: {strategy}")
                relationships, similarities = self.diagnose_semantic_categorization(true_labels, pred_labels)
                break  # Only diagnose the first strategy to avoid repetition
            if not multi_strategy_results:
                print("Error: No multi-strategy results obtained")
                return {'SCI': 0.0, 'SDS': 0.0, 'SPI': 0.0, 'error': 'No results'}
            
        # Step 2: Compute SCI
        sci_score = self.compute_semantic_control_index(multi_strategy_results)
        
        # Step 3: Compute SDS and SPI for each strategy
        sds_scores = {}
        spi_scores = {}
        
        for strategy, (true_labels, predicted_labels) in multi_strategy_results.items():
            # Generate intended targets for this strategy
            intended_targets = self.generate_intended_targets(true_labels, strategy)
            
            # Compute SDS
            sds_score = self.compute_semantic_directability_score(
                true_labels, predicted_labels, intended_targets, strategy
            )
            sds_scores[strategy] = sds_score
            
            # Compute SPI
            spi_score = self.compute_semantic_predictability_index(true_labels, predicted_labels)
            spi_scores[strategy] = spi_score
        
        # Average SDS and SPI across strategies
        avg_sds = np.mean(list(sds_scores.values())) if sds_scores else 0.0
        avg_spi = np.mean(list(spi_scores.values())) if spi_scores else 0.0
        
        metrics = {
            'SCI': sci_score,
            'SDS': avg_sds,
            'SPI': avg_spi,
            'detailed': {
                'strategy_results': multi_strategy_results,
                'sds_by_strategy': sds_scores,
                'spi_by_strategy': spi_scores,
                'total_samples': sum(len(labels[0]) for labels in multi_strategy_results.values())
            }
        }

        # Print results
        print(f"\nSemantic Control Evaluation Results:")
        print(f"SCI (Semantic Control Index): {sci_score:.4f}")
        print(f"SDS (Semantic Directability Score): {avg_sds:.4f}")
        print(f"SPI (Semantic Predictability Index): {avg_spi:.4f}")
        print(f"Total samples evaluated: {metrics['detailed']['total_samples']}")
        
        return metrics
    
    def diagnose_semantic_categorization(self, true_labels, predicted_labels):
        print("=== SEMANTIC RELATIONSHIP DIAGNOSTIC ===")
        relationships = []
        similarities = []
        
        for i, (true_label, pred_label) in enumerate(zip(true_labels[:20], predicted_labels[:20])):
            if pred_label != true_label:  # Only check misclassified examples
                true_idx = true_label.item()
                pred_idx = pred_label.item()
                
                if (true_idx < self.text_features.size(0) and 
                    pred_idx < self.text_features.size(0)):
                    
                    sim = torch.cosine_similarity(
                        self.text_features[true_idx:true_idx+1],
                        self.text_features[pred_idx:pred_idx+1]
                    ).item()
                    
                    rel = self.categorize_semantic_relationship(true_idx, pred_idx)
                    relationships.append(rel)
                    similarities.append(sim)
                    
                    # Show class names if available
                    true_name = self.class_names[true_idx] if true_idx < len(self.class_names) else f"class_{true_idx}"
                    pred_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"class_{pred_idx}"
                    
                    print(f"  {true_name} -> {pred_name}: sim={sim:.3f}, rel={rel}")
        
        if similarities:
            print(f"\nSimilarity range: {min(similarities):.3f} to {max(similarities):.3f}")
            print(f"Mean similarity: {np.mean(similarities):.3f}")
            print(f"Relationships found: {set(relationships)}")
            print(f"Relationship counts: {Counter(relationships)}")
        else:
            print("No misclassified examples found for analysis")
        print("=" * 45)
        
        return relationships, similarities
    
# Integration function for your existing codebase
def evaluate_attack_semantic_control(attack_name, attack_fn, model, test_loader, texts, 
                                   prompter=None, add_prompter=None, prompt_learner=None,
                                   class_names=None, device='cuda', max_batches=5):
    print(f"Evaluating semantic control for {attack_name}")
    
    # Convert texts to tensor if needed
    if not torch.is_tensor(texts):
        if isinstance(texts[0], str):
            from modified_clip import clip
            text_tokens = torch.stack([clip.tokenize(text)[0] for text in texts]).to(device)
        else:
            text_tokens = torch.stack([torch.from_numpy(np.array(text)) for text in texts]).to(device)
    else:
        text_tokens = texts
    
    # Default class names if not provided
    if class_names is None:
        num_classes = text_tokens.shape[0] if text_tokens.dim() > 0 else len(texts)
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Initialize evaluator
    evaluator = SemanticControlEvaluator(model, text_tokens, class_names, device)
    
    # Prepare attack kwargs template
    attack_kwargs_template = {
        'prompter': prompter,
        'model': model,
        'add_prompter': add_prompter,
        'criterion': torch.nn.CrossEntropyLoss(),
        'alpha': 0.01,  # Default step size
        'attack_iters': 20,
        'norm': 'l_inf',
        'prompt_learner': prompt_learner,
        'restarts': 1,
        'epsilon': 0.031,
        # Add other attack-specific parameters as needed
    }
    
    # Add SAPA-specific parameters if needed
    if 'sapa' in attack_name.lower():
        attack_kwargs_template.update({
            'text_perb_stepsize': 0.01,
            'embedding_loss_weight': 0.3,
            'semantic_anchor_weight': 0.2,
            'contrastive_loss_weight': 0.1,
            'dataset_classes': class_names,
            'original_dataset_classes': class_names,
            'model_type': 'CLIP'
        })
    
    # Run evaluation
    metrics = evaluator.evaluate_semantic_control(
        attack_fn, attack_kwargs_template, test_loader, max_batches=max_batches
    )
    
    return metrics


# Example usage for comparing attacks
def compare_attacks_semantic_control(attacks_dict, model, test_loader, texts, **kwargs):
    results = {}
    
    for attack_name, attack_fn in attacks_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {attack_name}")
        print(f"{'='*50}")
        
        try:
            metrics = evaluate_attack_semantic_control(
                attack_name, attack_fn, model, test_loader, texts, **kwargs
            )
            results[attack_name] = metrics
        except Exception as e:
            print(f"Error evaluating {attack_name}: {e}")
            results[attack_name] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("SEMANTIC CONTROL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Attack':<20} {'SCI':<8} {'SDS':<8} {'SPI':<8}")
    print("-" * 60)
    
    for attack_name, metrics in results.items():
        if 'error' in metrics:
            print(f"{attack_name:<20} {'ERROR':<24}")
        else:
            print(f"{attack_name:<20} {metrics['SCI']:<8.4f} {metrics['SDS']:<8.4f} {metrics['SPI']:<8.4f}")
    
    return results

# Semantic Coherence Metrics Class - for SAPA attack and evaluation 
class SemanticCoherenceMetrics:
    def __init__(self, model, text_tokens, class_names):
        self.model = model
        self.text_tokens = text_tokens
        self.class_names = class_names
        self.text_features = self._extract_text_features()
        
    def _extract_text_features(self):
        try:
            with torch.no_grad():
                if hasattr(self.model, 'encode_text'):
                    text_features = self.model.encode_text(self.text_tokens)
                    return F.normalize(text_features, dim=-1)
                else:
                    # Fallback manual extraction
                    x = self.text_tokens
                    x = self.model.token_embedding(x)
                    x = x + self.model.positional_embedding
                    x = x.permute(1, 0, 2)
                    x = self.model.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = self.model.ln_final(x)
                    text_features = x[torch.arange(x.shape[0]), self.text_tokens.argmax(dim=-1)] @ self.model.text_projection
                    return F.normalize(text_features, dim=-1)
        except Exception as e:
            print(f"Warning: Could not extract text features: {e}")
            return None
    
    def compute_semantic_similarity_preservation(self, true_labels, predicted_labels):
        if self.text_features is None:
            return 0.0
            
        similarities = []
        for true_label, pred_label in zip(true_labels, predicted_labels):
            true_idx = true_label.item()
            pred_idx = pred_label.item()
            
            if true_idx < self.text_features.size(0) and pred_idx < self.text_features.size(0):
                sim = torch.cosine_similarity(
                    self.text_features[true_idx:true_idx+1], 
                    self.text_features[pred_idx:pred_idx+1]
                ).item()
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def compute_semantic_rank_correlation(self, true_labels, predicted_labels):
        if self.text_features is None:
            return 0.0
        
        # Count prediction frequencies for each true class
        confusion_counts = {}
        for true_label, pred_label in zip(true_labels, predicted_labels):
            true_idx = true_label.item()
            pred_idx = pred_label.item()
            
            if true_idx not in confusion_counts:
                confusion_counts[true_idx] = Counter()
            confusion_counts[true_idx][pred_idx] += 1
        
        correlations = []
        
        for true_class, pred_counter in confusion_counts.items():
            if true_class >= self.text_features.size(0):
                continue
                
            # Get semantic similarities for this true class
            similarities = torch.cosine_similarity(
                self.text_features[true_class:true_class+1], 
                self.text_features
            ).cpu().numpy()
            
            # Get prediction frequencies
            pred_classes = list(range(len(self.class_names)))
            pred_freqs = [pred_counter.get(i, 0) for i in pred_classes]
            
            # Only compute correlation if we have multiple predictions
            if len(set(pred_freqs)) > 1:
                try:
                    corr, _ = spearmanr(similarities, pred_freqs)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def compute_interpretability_score(self, true_labels, predicted_labels):
        ssp = self.compute_semantic_similarity_preservation(true_labels, predicted_labels)
        src = self.compute_semantic_rank_correlation(true_labels, predicted_labels)
        return (ssp + src) / 2.0
    
    def compute_semantic_targeting_accuracy(self, true_labels, predicted_labels, semantic_targets, strategy):
        if self.text_features is None:
            return 0.0
        
        successful_attacks = (predicted_labels != true_labels)
        if successful_attacks.sum() == 0:
            return 0.0
        
        intended_successes = 0
        total_successes = successful_attacks.sum().item()
        
        for i, (true_label, pred_label, semantic_target, is_success) in enumerate(
            zip(true_labels, predicted_labels, semantic_targets, successful_attacks)):
            
            if not is_success:
                continue
                
            pred_idx = pred_label.item()
            target_idx = semantic_target.item()
            
            # Check if prediction matches intended semantic relationship
            if strategy == 'similar' or strategy == 'opposite':
                # Direct targeting - check if we hit the intended target
                if pred_idx == target_idx:
                    intended_successes += 1
            elif strategy == 'confusing':
                # For confusing strategy, check if prediction is in moderate similarity range
                true_idx = true_label.item()
                if (true_idx < self.text_features.size(0) and 
                    pred_idx < self.text_features.size(0)):
                    
                    true_feat = self.text_features[true_idx]
                    pred_feat = self.text_features[pred_idx]
                    similarity = torch.cosine_similarity(true_feat.unsqueeze(0), pred_feat.unsqueeze(0)).item()
                    
                    # Consider it successful if similarity is in moderate range (0.2 to 0.7)
                    if 0.2 <= similarity <= 0.7:
                        intended_successes += 1
        
        return intended_successes / total_successes if total_successes > 0 else 0.0
    
    def compute_semantic_targeting_alignment(
        self,
        adv_images: torch.Tensor,
        target_classes: Union[List[int], torch.Tensor],
        device: torch.device = None
    ) -> float:
        if device is None:
            device = next(self.model.parameters()).device
            
        self.model.eval()
        
        with torch.no_grad():
            # 1. Get Image Embeddings f_I(x_adv)
            # Note: Ensure images are preprocessed/normalized correctly for CLIP
            if hasattr(self.model, 'encode_image'):
                img_embeds = self.model.encode_image(adv_images.to(device))
            else:
                # Fallback for models without encode_image
                img_embeds = self.model.visual(adv_images.to(device))
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            img_embeds = img_embeds.float()
            
            # 2. Get Target Semantic Anchor Embeddings e_a
            # Convert target_classes to list if tensor
            if torch.is_tensor(target_classes):
                target_indices = target_classes.cpu().tolist()
            else:
                target_indices = list(target_classes)
            
            # Build target text prompts using class names
            target_texts = [f"a photo of a {self.class_names[t_idx]}" for t_idx in target_indices]
            
            # Tokenize and encode target texts
            from modified_clip import clip
            text_tokens = clip.tokenize(target_texts).to(device)
            
            if hasattr(self.model, 'encode_text'):
                text_embeds = self.model.encode_text(text_tokens)
            else:
                # Fallback manual text encoding
                x = text_tokens
                x = self.model.token_embedding(x)
                x = x + self.model.positional_embedding
                x = x.permute(1, 0, 2)
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)
                x = self.model.ln_final(x)
                text_embeds = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.model.text_projection
            
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds.float()
            
            # 3. Compute Cosine Similarity (STA)
            # Element-wise dot product between image [B, D] and text [B, D]
            # Equation: STA = (z_adv · e_a) for each pair
            sta_scores = (img_embeds * text_embeds).sum(dim=-1)
            
            # Return average STA
            return sta_scores.mean().item()
    
    def compute_all_metrics(self, true_labels, predicted_labels, semantic_targets=None, strategy=None,
                             adv_images=None, target_classes=None, device=None):
        metrics = {}
        
        if semantic_targets is not None and strategy is not None:
            metrics['STAccuracy'] = self.compute_semantic_targeting_accuracy(
                true_labels, predicted_labels, semantic_targets, strategy
            )
        
        # Compute STA (Semantic Target Alignment) if adversarial images and targets provided
        if adv_images is not None and target_classes is not None:
            metrics['STA'] = self.compute_semantic_targeting_alignment(
                adv_images, target_classes, device
            )
        
        return metrics

def attack_pgd(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0, dataset_name=None,dataset_classes=None, debug=False):
    delta = torch.zeros_like(X).cuda()     # detach needs to be deleted after motivation
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    delta.requires_grad = True
    if debug:
        print("📊 Extracting model's text embeddings...")
    text_embeddings = None
    try:
        with torch.no_grad():
            if hasattr(model, 'encode_text'):
                text_embeddings = model.encode_text(text_tokens)
                text_embeddings = F.normalize(text_embeddings, dim=-1)
            else:
                # Manual extraction for older CLIP versions
                x = text_tokens
                x = model.token_embedding(x)
                x = x + model.positional_embedding
                x = x.permute(1, 0, 2)
                x = model.transformer(x)
                x = x.permute(1, 0, 2)
                x = model.ln_final(x)
                text_embeddings = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
                text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        if debug:
            print(f"✅ Text embeddings extracted: {text_embeddings.shape}")
    except Exception as e:
        if debug:
            print(f"⚠️ Could not extract text embeddings: {e}")
            print("🔄 Will use fallback semantic targeting")
    targeter = create_semantic_targeter(dataset_name, dataset_classes, text_embeddings, debug=debug)
    # For untargeted attacks (PGD/CW), use TRUE CLASS for alignment measurement
    # This shows how much adversarial examples drift away from original class as epsilon increases
    semantic_targets = target  # Use true class to measure degradation
    # debug_semantic_selection(target, semantic_targets, dataset_classes, strategy="similar")

    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta,model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()  # Get gradients, detach from computation graph
        d = delta[:, :, :, :] # Copy of current perturbation
        g = grad[:, :, :, :] # Copy of gradients  
        x = X[:, :, :, :]   # Copy of original images
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)  # Keep perturbed image in valid pixel range [0,1]
        delta.data[:, :, :, :] = d   # Update delta with new perturbation
        delta.grad.zero_()  # Clear gradients for next iteration

    # Clear cache after attack completes 
    torch.cuda.empty_cache() 

    return delta,semantic_targets

def attack_pgd_targeted_semantic(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0,
               semantic_target_class=None, text_embeddings=None, debug=False):
    if semantic_target_class is None:
        # Fallback to random target if no semantic target provided
        num_classes = text_embeddings.shape[0] if text_embeddings is not None else 1000
        semantic_target_class = torch.randint(0, num_classes, (X.shape[0],), device=X.device)

    if isinstance(semantic_target_class, int):
        semantic_target_class = torch.tensor([semantic_target_class] * X.shape[0], device=X.device)

    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError(f"Unknown norm: {norm}")

    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    delta.requires_grad = True

    if debug:
        print(f"🎯 Targeted PGD: Attacking toward semantic class {semantic_target_class[0].item()}")
        print(f"✓ Original target class: {target[0].item()}")

    for t in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)

        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images

        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        # TARGETED: Minimize loss toward semantic target class
        # (instead of maximizing loss w.r.t. true class)
        loss = criterion(output, semantic_target_class)

        loss.backward()
        grad = delta.grad.detach()

        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            # For targeted: move TOWARD target, so SUBTRACT gradient
            d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            # For targeted: subtract gradient
            d = (d - scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

        if debug and t % 10 == 0:
            with torch.no_grad():
                pred = output.argmax(dim=1)
                success_rate = (pred == semantic_target_class).float().mean().item()
                print(f"  Iter {t}: Loss={loss.item():.4f}, Target Success Rate={success_rate:.2%}")

    torch.cuda.empty_cache()

    return delta, semantic_target_class


def attack_CW_targeted_semantic(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0,
               semantic_target_class=None, text_embeddings=None, kappa=0, debug=False):
    if semantic_target_class is None:
        num_classes = text_embeddings.shape[0] if text_embeddings is not None else 1000
        semantic_target_class = torch.randint(0, num_classes, (X.shape[0],), device=X.device)

    if isinstance(semantic_target_class, int):
        semantic_target_class = torch.tensor([semantic_target_class] * X.shape[0], device=X.device)

    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    delta.requires_grad = True

    # C&W uses Adam optimizer for smoother convergence
    from torch.optim import Adam
    optimizer = Adam([delta], lr=0.01)

    if debug:
        print(f"🎯 Targeted C&W: Attacking toward semantic class {semantic_target_class[0].item()}")

    for t in range(attack_iters):
        optimizer.zero_grad()

        _images = clip_img_preprocessing(X + delta, model=model)

        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images

        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        # C&W targeted loss: minimize perturbation + maximize target logit
        # loss = ||delta||^2 + c * max(Z_target - Z_max_other, -kappa)

        # Perturbation magnitude (L2)
        perturbation_loss = delta.view(delta.shape[0], -1).norm(p=2, dim=1).mean()

        # Targeted classification loss
        # We want Z[target] - Z[max_other] > kappa
        # So loss = max(Z[max_other] - Z[target] + kappa, 0)

        # Get target logits
        target_logits = output[torch.arange(output.shape[0]), semantic_target_class]

        # Get max of other logits
        mask = torch.ones_like(output, dtype=torch.bool)
        mask[torch.arange(output.shape[0]), semantic_target_class] = False
        other_logits = output[mask].view(output.shape[0], -1)
        max_other_logits = other_logits.max(dim=1)[0]

        # C&W loss
        f_loss = torch.clamp(max_other_logits - target_logits + kappa, min=0.0)
        classification_loss = f_loss.mean()

        # Combined loss (c is implicitly balanced via the loss magnitude)
        loss = perturbation_loss + 10.0 * classification_loss

        loss.backward()

        # Manually update delta with projection
        with torch.no_grad():
            grad = delta.grad
            delta.data = delta.data - 0.01 * grad

            # Project onto epsilon ball
            if norm == "l_inf":
                delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
            elif norm == "l_2":
                d_flat = delta.data.view(delta.shape[0], -1)
                d_norm = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
                scale = torch.clamp(d_norm / epsilon, max=1.0)
                delta.data = delta.data / (scale + 1e-10)

            # Project onto valid image range
            delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)

        if debug and t % 10 == 0:
            with torch.no_grad():
                pred = output.argmax(dim=1)
                success_rate = (pred == semantic_target_class).float().mean().item()
                print(f"  Iter {t}: Loss={loss.item():.4f}, Pert={perturbation_loss.item():.4f}, "
                      f"Class={classification_loss.item():.4f}, Target Success={success_rate:.2%}")

    torch.cuda.empty_cache()

    return delta, semantic_target_class


def attack_pgd_motivation(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X.detach()).cuda()     # detach needs to be deleted for motivation
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    delta.requires_grad = True
    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def high_curv_point(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X.detach()).cuda()     
    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    delta.requires_grad = True
    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def attack_pgd_nuc(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, ori_nat_logits, W_CE, W_reg, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = W_CE * criterion(output, target) + W_reg * torch.norm(ori_nat_logits - output, 'nuc')/_images.size(0)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def attack_TRADES_KL(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, ori_nat_logits, restarts=1, early_stop=True, epsilon=0):
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = criterion_KL(F.log_softmax(output, dim=1), F.softmax(ori_nat_logits, dim=1))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def criterion_L2(out, targets, reduction='mean'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    return squared_error_batch

def attack_FARE_Emb_L2(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, ori_nat_emb, restarts=1, early_stop=True, epsilon=0):
    # criterion_L2 = torch.nn.MSELoss().cuda()
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _, output_emb, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token, is_embedding=True)

        loss = criterion_L2(output_emb, ori_nat_emb)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def attack_CW(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0,dataset_name=None,dataset_classes=None, debug=False):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    if debug:
        print("📊 Extracting model's text embeddings...")
    text_embeddings = None
    try:
        with torch.no_grad():
            if hasattr(model, 'encode_text'):
                text_embeddings = model.encode_text(text_tokens)
                text_embeddings = F.normalize(text_embeddings, dim=-1)
            else:
                # Manual extraction for older CLIP versions
                x = text_tokens
                x = model.token_embedding(x)
                x = x + model.positional_embedding
                x = x.permute(1, 0, 2)
                x = model.transformer(x)
                x = x.permute(1, 0, 2)
                x = model.ln_final(x)
                text_embeddings = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
                text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        if debug:
            print(f"✅ Text embeddings extracted: {text_embeddings.shape}")
    except Exception as e:
        if debug:
            print(f"⚠️ Could not extract text embeddings: {e}")
            print("🔄 Will use fallback semantic targeting")
    targeter = create_semantic_targeter(dataset_name, dataset_classes, text_embeddings, debug=debug)
    # For untargeted attacks (PGD/CW), use TRUE CLASS for alignment measurement
    # This shows how much adversarial examples drift away from original class as epsilon increases
    semantic_targets = target  # Use true class to measure degradation
    # debug_semantic_selection(target, semantic_targets, dataset_classes, strategy="similar")

    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images

        prompt_token = add_prompter() if add_prompter is not None else None

        output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask*output, dim=1)
        wrong_logit, _ = torch.max((1-label_mask)*output - 1e4*label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta,semantic_targets

def attack_auto(model, images, target, text_tokens, prompter, add_prompter,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens,
        prompter=prompter, add_prompter=add_prompter
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv

def attack_auto_new(model, images, target, text_tokens, prompter, add_prompter,
                             attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    def model_fn(x):
        if prompter is not None:
            prompted_images = prompter(clip_img_preprocessing(x))
        else:
            prompted_images = clip_img_preprocessing(x)
        prompt_token = add_prompter() if add_prompter is not None else None
        output_a, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
        # print("img_shape", prompted_images.shape, "text_shape", text_tokens.shape, "output_shape", output_a.shape)
        return output_a.to(torch.float32)

    adversary = AutoAttack(model_fn, norm='Linf', eps=epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv

def attack_pgd_adv_prompt(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, prompt_learner, text_perb_stepsize, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    ################## text prompt optimizer ##################
    prompter_optim = torch.optim.SGD(prompt_learner.parameters(),
                                    lr=text_perb_stepsize,
                                    momentum=0,
                                    weight_decay=0)
    # prompter_state = copy.deepcopy(prompt_learner.state_dict())
    ################## text prompt optimizer ##################

    ### Simulate the adversarial token embedding ###
    # prompt_output = prompt_learner()
    # print("Prompt learner output stats:")
    # print(f"Mean: {prompt_output.mean().item()}, StdDev: {prompt_output.std().item()}")
    # print(prompt_learner().shape)
    ### Simulate the adversarial token embedding ###

    for _ in range(attack_iters):
        
        _images = clip_img_preprocessing(X + delta, model=model)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        
        output, _ = multiGPU_CLIP_Text_Prompt_Tuning(model, prompted_images, text_tokens, prompt_token, prompt_learner)

        loss = criterion(output, target)

        prompter_optim.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

        for param in prompt_learner.parameters():
            if param.grad is not None:
                # Gradient sign reverse
                param.grad.data = -1.0 * param.grad.data
        prompter_optim.step()

        ### Simulate the adversarial token embedding ###
        # prompt_output = prompt_learner()
        # print("Prompt learner output stats:")
        # print(f"Mean: {prompt_output.mean().item()}, StdDev: {prompt_output.std().item()}")
        # print(prompt_learner().shape)
        # print("loss", loss)
        ### Simulate the adversarial token embedding ###

    ## Reset Prompt Learner
    # prompt_learner.load_state_dict(prompter_state)
    prompter_optim.zero_grad()

    return delta

# PGD only disrupts the text branch
def attack_pgd_adv_promptONLY(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, prompt_learner, text_perb_stepsize, restarts=1, early_stop=True, epsilon=0):
    
    delta = torch.zeros_like(X).cuda()
    
    ################## text prompt optimizer ##################
    prompter_optim = torch.optim.SGD(prompt_learner.parameters(),
                                    lr=text_perb_stepsize,
                                    momentum=0,
                                    weight_decay=0)
    # prompter_state = copy.deepcopy(prompt_learner.state_dict())
    ################## text prompt optimizer ##################

    ### Simulate the adversarial token embedding ###
    # prompt_output = prompt_learner()
    # print("Prompt learner output stats:")
    # print(f"Mean: {prompt_output.mean().item()}, StdDev: {prompt_output.std().item()}")
    # print(prompt_learner().shape)
    ### Simulate the adversarial token embedding ###

    for _ in range(attack_iters):
        
        _images = clip_img_preprocessing(X)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        
        output, _ = multiGPU_CLIP_Text_Prompt_Tuning(model, prompted_images, text_tokens, prompt_token, prompt_learner)

        loss = criterion(output, target)

        prompter_optim.zero_grad()
        loss.backward()

        for param in prompt_learner.parameters():
            if param.grad is not None:
                # Gradient sign reverse
                param.grad.data = -1.0 * param.grad.data
        prompter_optim.step()

        ### Simulate the adversarial token embedding ###
        # prompt_output = prompt_learner()
        # print("Prompt learner output stats:")
        # print(f"Mean: {prompt_output.mean().item()}, StdDev: {prompt_output.std().item()}")
        # print(prompt_learner().shape)
        # print("loss", loss)
        ### Simulate the adversarial token embedding ###

    ## Reset Prompt Learner
    # prompt_learner.load_state_dict(prompter_state)
    prompter_optim.zero_grad()
    return delta

# 2025-09-07 Semantic flow attack - good for CLIP, when wordnet not use  
def attack_semantic_flow_basic(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, prompt_learner=None, text_perb_stepsize=0.01,
                        restarts=1, early_stop=True, epsilon=0, semantic_strategy="confusing",
                        embedding_loss_weight=0.3, semantic_anchor_weight=0.2, 
                        contrastive_loss_weight=0.1, dataset_classes=None, 
                        original_dataset_classes=None, momentum=0.9, 
                        temperature=0.1, flow_strength=0.5):
    
    lower_limit, upper_limit = 0, 1
    
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    # Extract text embeddings for semantic analysis
    def extract_text_embeddings(model, text_tokens):
        try:
            with torch.no_grad():
                if hasattr(model, 'encode_text'):
                    text_features = model.encode_text(text_tokens)
                    return F.normalize(text_features, dim=-1)
                else:
                    # Manual extraction
                    x = text_tokens
                    x = model.token_embedding(x)
                    x = x + model.positional_embedding
                    x = x.permute(1, 0, 2)
                    x = model.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = model.ln_final(x)
                    text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
                    return F.normalize(text_features, dim=-1)
        except Exception as e:
            print(f"Warning: Could not extract text embeddings: {e}")
            return None
    
    # Semantic guidance loss computation
    def compute_semantic_guidance_loss(output, image_embeddings, text_embeddings, targets, semantic_targets, flow_strength):
        if image_embeddings is None or text_embeddings is None:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        batch_size = output.size(0)
        device = output.device
        
        # 1. Embedding space semantic guidance
        semantic_embedding_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for i in range(batch_size):
            true_idx = targets[i].item()
            sem_idx = semantic_targets[i].item()
            
            if true_idx < text_embeddings.size(0) and sem_idx < text_embeddings.size(0):
                true_text_emb = text_embeddings[true_idx].to(device)
                sem_text_emb = text_embeddings[sem_idx].to(device)
                img_emb = image_embeddings[i]
                
                # Push image embedding away from true class, toward semantic class
                true_similarity = torch.cosine_similarity(img_emb.unsqueeze(0), true_text_emb.unsqueeze(0))
                sem_similarity = torch.cosine_similarity(img_emb.unsqueeze(0), sem_text_emb.unsqueeze(0))
                
                # Semantic flow loss: minimize true similarity, maximize semantic similarity
                semantic_embedding_loss = semantic_embedding_loss + true_similarity - sem_similarity
        
        semantic_embedding_loss = semantic_embedding_loss / batch_size
        
        # 2. Output space semantic guidance (logits)
        correct_logits = output.gather(1, targets.unsqueeze(1)).squeeze(1)
        semantic_logits = output.gather(1, semantic_targets.unsqueeze(1)).squeeze(1)
        
        # Semantic flow in logit space: decrease correct, increase semantic
        logit_flow_loss = correct_logits.mean() - semantic_logits.mean()
        
        # 3. Combine embedding and logit guidance
        total_semantic_loss = flow_strength * semantic_embedding_loss + (1 - flow_strength) * logit_flow_loss
        
        return total_semantic_loss
    
    # Adaptive step sizing based on gradient consistency
    def compute_adaptive_step_size(current_delta, current_grad, base_alpha, epsilon_val):
        # Compute perturbation magnitude
        delta_magnitude = torch.norm(current_delta.view(current_delta.size(0), -1), dim=1).mean()
        
        # Compute gradient magnitude
        grad_magnitude = torch.norm(current_grad.view(current_grad.size(0), -1), dim=1).mean()
        
        # Adaptive scaling based on gradient and perturbation consistency
        if delta_magnitude < epsilon_val * 0.5:
            scale_factor = 1.2  # Increase step size when perturbation is small
        elif delta_magnitude > epsilon_val * 0.9:
            scale_factor = 0.8  # Decrease step size when near boundary
        else:
            scale_factor = 1.0  # Maintain step size
            
        # Additional scaling based on gradient magnitude
        if grad_magnitude > 1.0:
            scale_factor *= 0.9  # Reduce step size for large gradients
        elif grad_magnitude < 0.1:
            scale_factor *= 1.1  # Increase step size for small gradients
            
        return base_alpha * scale_factor
    
    # Select semantic targets
    def select_semantic_targets(targets, text_embeddings, strategy, num_classes):
        if text_embeddings is None:
            # Fallback: random different classes
            target_classes = []
            for i in range(targets.size(0)):
                true_class = targets[i].item()
                wrong_classes = [j for j in range(num_classes) if j != true_class]
                semantic_target = np.random.choice(wrong_classes) if wrong_classes else (true_class + 1) % num_classes
                target_classes.append(semantic_target)
            return torch.tensor(target_classes, device=targets.device)
        
        # Compute similarity matrix for semantic targeting
        similarity_matrix = torch.matmul(text_embeddings, text_embeddings.T)
        target_classes = []
        
        for i in range(targets.size(0)):
            true_class = targets[i].item()
            
            if true_class >= similarity_matrix.size(0):
                semantic_target = (true_class + 1) % num_classes
            else:
                similarities = similarity_matrix[true_class]
                similarities[true_class] = -2.0  # Mask out true class
                
                if strategy == "confusing":
                    # Choose moderately similar classes
                    sorted_sims, sorted_indices = torch.sort(similarities, descending=True)
                    mid_range = len(sorted_indices) // 3
                    start_idx = mid_range
                    end_idx = min(mid_range * 2, len(sorted_indices) - 1)
                    choice_idx = start_idx + np.random.randint(0, max(1, end_idx - start_idx))
                    semantic_target = sorted_indices[choice_idx].item()
                elif strategy == "similar":
                    semantic_target = torch.argmax(similarities).item()
                elif strategy == "opposite":
                    semantic_target = torch.argmin(similarities).item()
                else:  # random
                    valid_indices = [j for j in range(num_classes) if j != true_class]
                    semantic_target = np.random.choice(valid_indices) if valid_indices else (true_class + 1) % num_classes
            
            target_classes.append(semantic_target)
        
        return torch.tensor(target_classes, device=targets.device)
    
    # Main attack implementation
    print(f"Starting Semantic Flow Attack with strategy: {semantic_strategy}")
    
    # Extract text embeddings
    text_embeddings = extract_text_embeddings(model, text_tokens)
    num_classes = text_tokens.size(0)
    
    best_delta = None
    best_attack_rate = -1
    final_semantic_targets = None
    
    for restart_idx in range(restarts):
        print(f"Semantic Flow Restart {restart_idx + 1}/{restarts}")
        
        # Initialize perturbation and momentum
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        
        # Initialize momentum buffer
        momentum_buffer = torch.zeros_like(delta)
        
        # Text prompt optimizer
        if prompt_learner is not None:
            prompter_optim = torch.optim.SGD(prompt_learner.parameters(),
                                          lr=text_perb_stepsize, momentum=0, weight_decay=0)
            prompt_original_state = {k: v.clone() for k, v in prompt_learner.state_dict().items()}
        
        # Select semantic targets
        semantic_targets = select_semantic_targets(target, text_embeddings, semantic_strategy, num_classes)
        if restart_idx == 0:
            final_semantic_targets = semantic_targets.clone()
        
        print(f"  Targets: {target[:3].cpu().numpy()} -> Semantic: {semantic_targets[:3].cpu().numpy()}")
        
        # Attack iterations with semantic flow
        for iter_idx in range(attack_iters):
            # Forward pass to get embeddings
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            # Get output and embeddings
            try:
                if prompt_learner is not None:
                    output, _, image_embeddings, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                        model, prompted_images, text_tokens, prompt_token, prompt_learner, is_embedding=True)
                    if prompt_learner is not None:
                        prompter_optim.zero_grad()
                else:
                    output, _, image_embeddings, _ = multiGPU_CLIP(
                        model, prompted_images, text_tokens, prompt_token, is_embedding=True)
                
                if image_embeddings is not None:
                    image_embeddings = F.normalize(image_embeddings, dim=-1)
            except Exception:
                # Fallback without embeddings
                if prompt_learner is not None:
                    output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                        model, prompted_images, text_tokens, prompt_token, prompt_learner)
                    if prompt_learner is not None:
                        prompter_optim.zero_grad()
                else:
                    output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
                image_embeddings = None
            
            # Compute semantic guidance loss
            semantic_guidance_loss = compute_semantic_guidance_loss(
                output, image_embeddings, text_embeddings, target, semantic_targets, flow_strength)
            
            # Compute main losses
            adversarial_loss = criterion(output, target)
            semantic_targeting_loss = -criterion(output, semantic_targets)  # Negative to maximize
            
            # Combined loss with semantic guidance
            total_loss = (embedding_loss_weight * adversarial_loss + 
                         semantic_anchor_weight * semantic_targeting_loss +
                         contrastive_loss_weight * semantic_guidance_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Get gradient
            grad = delta.grad.detach()
            
            # Compute adaptive step size
            adaptive_alpha = compute_adaptive_step_size(delta, grad, alpha, epsilon)
            
            # Momentum update (Nesterov-style)
            momentum_buffer = momentum * momentum_buffer + grad
            
            # Update perturbation with momentum and adaptive step size
            d = delta[:, :, :, :]
            x = X[:, :, :, :]
            
            if norm == "l_inf":
                d = torch.clamp(d + adaptive_alpha * torch.sign(momentum_buffer), 
                               min=-epsilon, max=epsilon)
            elif norm == "l_2":
                momentum_norm = torch.norm(momentum_buffer.view(momentum_buffer.shape[0], -1), 
                                         dim=1).view(-1, 1, 1, 1)
                scaled_momentum = momentum_buffer / (momentum_norm + 1e-10)
                d = (d + scaled_momentum * adaptive_alpha).view(d.size(0), -1).renorm(
                    p=2, dim=0, maxnorm=epsilon).view_as(d)
            
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
            
            # Update text perturbation
            if prompt_learner is not None:
                for param in prompt_learner.parameters():
                    if param.grad is not None:
                        param.grad.data = -1.0 * param.grad.data
                prompter_optim.step()
            
            # Debug output
            if iter_idx % 10 == 0:
                with torch.no_grad():
                    pred_classes = output.argmax(dim=1)
                    correct_rate = (pred_classes == target).float().mean()
                    semantic_rate = (pred_classes == semantic_targets).float().mean()
                    print(f"    Iter {iter_idx}: Loss={total_loss.item():.3f}, "
                          f"Correct={correct_rate:.3f}, Semantic={semantic_rate:.3f}, "
                          f"Alpha={adaptive_alpha:.4f}")
        
        # Evaluate restart
        with torch.no_grad():
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            if prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
            
            pred_classes = output.argmax(dim=1)
            attack_rate = (pred_classes != target).float().mean()
            
            if attack_rate > best_attack_rate:
                best_attack_rate = attack_rate
                best_delta = delta.clone()
                print(f"    New best attack rate: {attack_rate:.3f}")
        
        # Reset prompt learner
        if prompt_learner is not None:
            prompt_learner.load_state_dict(prompt_original_state)
    
    print(f"Final Semantic Flow attack success rate: {best_attack_rate:.3f}")
    
    return best_delta, final_semantic_targets

# 2025-09-08 Semantic flow attack with wordnet 
def attack_semantic_flow(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, prompt_learner=None, text_perb_stepsize=0.01,
                        restarts=1, early_stop=True, epsilon=0, semantic_strategy="confusing",
                        embedding_loss_weight=0.3, semantic_anchor_weight=0.2, 
                        contrastive_loss_weight=0.1, dataset_classes=None, 
                        original_dataset_classes=None, momentum=0.9, 
                        temperature=0.1, flow_strength=0.5, dataset_name=""):
    
    if NLTK_AVAILABLE and dataset_classes is not None:
        return attack_semantic_flow_enhanced(
            prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
            attack_iters, norm, prompt_learner, text_perb_stepsize,
            restarts, early_stop, epsilon, semantic_strategy,
            embedding_loss_weight, semantic_anchor_weight, contrastive_loss_weight,
            dataset_classes, original_dataset_classes, momentum, 
            temperature, flow_strength, dataset_name
        )
    else:
        # Fallback to original implementation if WordNet not available
        print("WordNet not available, using basic semantic flow")
        return attack_semantic_flow_basic(
            prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
            attack_iters, norm, prompt_learner, text_perb_stepsize,
            restarts, early_stop, epsilon, semantic_strategy,
            embedding_loss_weight, semantic_anchor_weight, contrastive_loss_weight,
            dataset_classes, original_dataset_classes, momentum, 
            temperature, flow_strength
        )


def attack_semantic_flow_enhanced(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
                                 attack_iters, norm, prompt_learner=None, text_perb_stepsize=0.01,
                                 restarts=1, early_stop=True, epsilon=0, semantic_strategy="confusing",
                                 embedding_loss_weight=0.3, semantic_anchor_weight=0.2, 
                                 contrastive_loss_weight=0.1, dataset_classes=None, 
                                 original_dataset_classes=None, momentum=0.9, 
                                 temperature=0.1, flow_strength=0.5, dataset_name=""):
    
    lower_limit, upper_limit = 0, 1
    device = X.device
    
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    # ========================================================================
    # WORDNET SEMANTIC HIERARCHY CONSTRUCTION
    # ========================================================================
    
    def build_wordnet_semantic_graph(class_names):
        try:
            import nltk
            from nltk.corpus import wordnet as wn
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading WordNet...")
                nltk.download('wordnet')
                nltk.download('omw-1.4')  # For better coverage
                
            semantic_graph = {}
            
            for i, class_name in enumerate(class_names):
                # Clean class name for WordNet lookup
                clean_name = class_name.lower().replace('_', ' ').replace('-', ' ')
                
                try:
                    # Get synsets for the class
                    synsets = wn.synsets(clean_name, pos=wn.NOUN)
                    if not synsets:
                        # Try without POS restriction
                        synsets = wn.synsets(clean_name)
                    
                    if synsets:
                        primary_synset = synsets[0]  # Use most common synset
                        
                        # Extract various semantic relationships
                        relationships = {
                            'hypernyms': [],      # More general concepts (animal -> mammal)
                            'hyponyms': [],       # More specific concepts (mammal -> dog)
                            'meronyms': [],       # Part-of relationships (car -> wheel)
                            'holonyms': [],       # Whole-of relationships (wheel -> car)
                            'similar_to': [],     # Similar concepts
                            'entailments': [],    # What this concept entails
                            'antonyms': [],       # Opposite concepts
                            'coordinates': []     # Sibling concepts (same hypernym)
                        }
                        
                        # Get hypernyms (more general)
                        for hypernym in primary_synset.hypernyms():
                            relationships['hypernyms'].extend([lemma.name().replace('_', ' ') 
                                                             for lemma in hypernym.lemmas()])
                        
                        # Get hyponyms (more specific)
                        for hyponym in primary_synset.hyponyms():
                            relationships['hyponyms'].extend([lemma.name().replace('_', ' ') 
                                                            for lemma in hyponym.lemmas()])
                        
                        # Get meronyms (parts)
                        for meronym in primary_synset.part_meronyms():
                            relationships['meronyms'].extend([lemma.name().replace('_', ' ') 
                                                            for lemma in meronym.lemmas()])
                        
                        # Get holonyms (wholes)
                        for holonym in primary_synset.part_holonyms():
                            relationships['holonyms'].extend([lemma.name().replace('_', ' ') 
                                                            for lemma in holonym.lemmas()])
                        
                        # Get similar concepts
                        for similar in primary_synset.similar_tos():
                            relationships['similar_to'].extend([lemma.name().replace('_', ' ') 
                                                              for lemma in similar.lemmas()])
                        
                        # Get antonyms from lemmas
                        for lemma in primary_synset.lemmas():
                            for antonym in lemma.antonyms():
                                relationships['antonyms'].append(antonym.name().replace('_', ' '))
                        
                        # Get coordinate terms (siblings in taxonomy)
                        for hypernym in primary_synset.hypernyms():
                            for coordinate in hypernym.hyponyms():
                                if coordinate != primary_synset:
                                    relationships['coordinates'].extend([lemma.name().replace('_', ' ') 
                                                                       for lemma in coordinate.lemmas()])
                        
                        # Calculate semantic distances to other classes
                        semantic_distances = {}
                        conceptual_paths = {}
                        
                        for j, other_class in enumerate(class_names):
                            if i != j:
                                other_clean = other_class.lower().replace('_', ' ').replace('-', ' ')
                                other_synsets = wn.synsets(other_clean, pos=wn.NOUN)
                                if not other_synsets:
                                    other_synsets = wn.synsets(other_clean)
                                
                                if other_synsets:
                                    other_synset = other_synsets[0]
                                    
                                    # Multiple distance metrics
                                    path_similarity = primary_synset.path_similarity(other_synset)
                                    wup_similarity = primary_synset.wup_similarity(other_synset)
                                    lch_similarity = None
                                    
                                    try:
                                        lch_similarity = primary_synset.lch_similarity(other_synset)
                                    except:
                                        pass
                                    
                                    # Combine similarities into distance
                                    if path_similarity is not None:
                                        path_distance = 1.0 - path_similarity
                                    else:
                                        path_distance = 1.0
                                    
                                    if wup_similarity is not None:
                                        wup_distance = 1.0 - wup_similarity
                                    else:
                                        wup_distance = 1.0
                                    
                                    # Combined semantic distance
                                    semantic_distance = 0.6 * path_distance + 0.4 * wup_distance
                                    semantic_distances[j] = semantic_distance
                                    
                                    # Find conceptual path
                                    try:
                                        lcs = primary_synset.lowest_common_hypernyms(other_synset)
                                        if lcs:
                                            conceptual_paths[j] = {
                                                'common_ancestor': lcs[0].name(),
                                                'path_length': primary_synset.shortest_path_distance(other_synset)
                                            }
                                    except:
                                        conceptual_paths[j] = {'common_ancestor': None, 'path_length': None}
                        
                        semantic_graph[i] = {
                            'synset': primary_synset,
                            'relationships': relationships,
                            'semantic_distances': semantic_distances,
                            'conceptual_paths': conceptual_paths,
                            'definition': primary_synset.definition(),
                            'examples': primary_synset.examples(),
                            'class_name': class_name
                        }
                    else:
                        # Fallback for classes not in WordNet
                        semantic_graph[i] = {
                            'synset': None,
                            'relationships': {k: [] for k in ['hypernyms', 'hyponyms', 'meronyms', 
                                                            'holonyms', 'similar_to', 'entailments', 
                                                            'antonyms', 'coordinates']},
                            'semantic_distances': {},
                            'conceptual_paths': {},
                            'definition': None,
                            'examples': [],
                            'class_name': class_name
                        }
                        
                except Exception as e:
                    print(f"Error processing {class_name}: {e}")
                    semantic_graph[i] = {
                        'synset': None,
                        'relationships': {k: [] for k in ['hypernyms', 'hyponyms', 'meronyms', 
                                                        'holonyms', 'similar_to', 'entailments', 
                                                        'antonyms', 'coordinates']},
                        'semantic_distances': {},
                        'conceptual_paths': {},
                        'definition': None,
                        'examples': [],
                        'class_name': class_name
                    }
            
            return semantic_graph
            
        except ImportError:
            print("Warning: NLTK not available. Using basic semantic graph.")
            return {i: {'synset': None, 'relationships': {}, 'semantic_distances': {}, 
                       'conceptual_paths': {}, 'definition': None, 'examples': [], 
                       'class_name': class_names[i]} for i in range(len(class_names))}
    
    # ========================================================================
    # ENHANCED TEXT EMBEDDING EXTRACTION WITH SEMANTIC CONTEXT
    # ========================================================================
    
    def extract_semantic_embeddings(model, text_tokens, semantic_graph):
        try:
            with torch.no_grad():
                if hasattr(model, 'encode_text'):
                    text_features = model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, dim=-1)
                else:
                    # Manual extraction
                    x = text_tokens
                    x = model.token_embedding(x)
                    x = x + model.positional_embedding
                    x = x.permute(1, 0, 2)
                    x = model.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = model.ln_final(x)
                    text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
                    text_features = F.normalize(text_features, dim=-1)
                
                # Enhance embeddings with semantic structure
                enhanced_features = text_features.clone()
                
                # Create semantic similarity matrix
                semantic_similarity_matrix = torch.zeros(len(dataset_classes), len(dataset_classes))
                
                for i in range(len(dataset_classes)):
                    for j in range(len(dataset_classes)):
                        if i != j and i in semantic_graph and j in semantic_graph[i]['semantic_distances']:
                            # Convert semantic distance to similarity
                            distance = semantic_graph[i]['semantic_distances'][j]
                            similarity = max(0.0, 1.0 - distance)
                            semantic_similarity_matrix[i, j] = similarity
                
                return text_features, semantic_similarity_matrix.to(device)
                
        except Exception as e:
            print(f"Warning: Could not extract semantic embeddings: {e}")
            return None, None
    
    # ========================================================================
    # SEMANTIC TARGET SELECTION WITH WORDNET RELATIONSHIPS
    # ========================================================================
    
    def select_semantic_targets_wordnet(targets, semantic_graph, text_embeddings, strategy, num_classes):
        semantic_targets = []
        
        print(f"Semantic targeting with WordNet relationships")
        print(f"Strategy: {strategy}")
        
        for i, target_idx in enumerate(targets):
            true_idx = target_idx.item()
            
            if true_idx not in semantic_graph:
                # Fallback to random
                wrong_classes = [j for j in range(num_classes) if j != true_idx]
                semantic_target = np.random.choice(wrong_classes) if wrong_classes else (true_idx + 1) % num_classes
            else:
                graph_info = semantic_graph[true_idx]
                relationships = graph_info['relationships']
                semantic_distances = graph_info['semantic_distances']
                
                candidates = []
                
                if strategy == "opposite" or strategy == "antonym":
                    # Use antonyms or most distant concepts
                    antonym_candidates = []
                    for antonym in relationships['antonyms']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and antonym.lower() in class_name.lower():
                                antonym_candidates.append(j)
                    
                    if antonym_candidates:
                        candidates = antonym_candidates
                    else:
                        # Use most semantically distant
                        if semantic_distances:
                            candidates = [j for j, dist in semantic_distances.items() if dist > 0.7]
                        
                elif strategy == "similar" or strategy == "coordinate":
                    # Use coordinate terms (siblings) or similar concepts
                    coordinate_candidates = []
                    similar_candidates = []
                    
                    for coord in relationships['coordinates']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and coord.lower() in class_name.lower():
                                coordinate_candidates.append(j)
                    
                    for sim in relationships['similar_to']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and sim.lower() in class_name.lower():
                                similar_candidates.append(j)
                    
                    candidates = coordinate_candidates + similar_candidates
                    
                    if not candidates and semantic_distances:
                        # Use semantically similar
                        candidates = [j for j, dist in semantic_distances.items() if dist < 0.4]
                        
                elif strategy == "hypernym":
                    # Use more general concepts
                    hypernym_candidates = []
                    for hypernym in relationships['hypernyms']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and hypernym.lower() in class_name.lower():
                                hypernym_candidates.append(j)
                    candidates = hypernym_candidates
                    
                elif strategy == "hyponym":
                    # Use more specific concepts
                    hyponym_candidates = []
                    for hyponym in relationships['hyponyms']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and hyponym.lower() in class_name.lower():
                                hyponym_candidates.append(j)
                    candidates = hyponym_candidates
                    
                elif strategy == "meronym":
                    # Use part-of relationships
                    meronym_candidates = []
                    for meronym in relationships['meronyms']:
                        for j, class_name in enumerate(dataset_classes):
                            if j != true_idx and meronym.lower() in class_name.lower():
                                meronym_candidates.append(j)
                    candidates = meronym_candidates
                    
                elif strategy == "confusing":
                    # Use medium semantic distance for confusion
                    if semantic_distances:
                        sorted_distances = sorted(semantic_distances.items(), key=lambda x: x[1])
                        mid_start = len(sorted_distances) // 4
                        mid_end = 3 * len(sorted_distances) // 4
                        candidates = [idx for idx, _ in sorted_distances[mid_start:mid_end]]
                    
                elif strategy == "adaptive":
                    # Dynamic strategy based on relationships available
                    if relationships['antonyms']:
                        strategy = "antonym"
                    elif relationships['coordinates']:
                        strategy = "coordinate"
                    elif semantic_distances:
                        strategy = "opposite"
                    else:
                        strategy = "random"
                    
                    # Recursive call with determined strategy
                    return select_semantic_targets_wordnet([target_idx], {true_idx: graph_info}, 
                                                         text_embeddings, strategy, num_classes)
                
                # Select from candidates
                if candidates:
                    semantic_target = np.random.choice(candidates)
                elif semantic_distances:
                    # Fallback to most distant
                    semantic_target = max(semantic_distances.keys(), key=lambda k: semantic_distances[k])
                else:
                    # Ultimate fallback
                    wrong_classes = [j for j in range(num_classes) if j != true_idx]
                    semantic_target = np.random.choice(wrong_classes) if wrong_classes else (true_idx + 1) % num_classes
            
            semantic_targets.append(semantic_target)
            
            # Show semantic relationship for first few samples
            if i < 3:
                true_class = dataset_classes[true_idx]
                sem_class = dataset_classes[semantic_target]
                
                relationship_type = "unknown"
                if true_idx in semantic_graph and semantic_target in semantic_graph[true_idx]['semantic_distances']:
                    distance = semantic_graph[true_idx]['semantic_distances'][semantic_target]
                    if distance < 0.3:
                        relationship_type = "similar"
                    elif distance > 0.7:
                        relationship_type = "opposite"
                    else:
                        relationship_type = "confusing"
                
                print(f"  {true_class} → {sem_class} ({relationship_type})")
        
        return torch.tensor(semantic_targets, device=targets.device)
    
    # ========================================================================
    # SEMANTIC FLOW GUIDANCE LOSS WITH WORDNET STRUCTURE
    # ========================================================================
    
    def compute_semantic_flow_loss(output, image_embeddings, text_embeddings, targets, semantic_targets, 
                                 semantic_graph, semantic_similarity_matrix, flow_strength, iter_idx):
        if image_embeddings is None or text_embeddings is None:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        batch_size = output.size(0)
        device = output.device
        
        # =====================================================================
        # 1. SEMANTIC EMBEDDING FLOW (WordNet-guided)
        # =====================================================================
        semantic_embedding_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for i in range(batch_size):
            true_idx = targets[i].item()
            sem_idx = semantic_targets[i].item()
            
            if (true_idx < text_embeddings.size(0) and 
                sem_idx < text_embeddings.size(0) and
                true_idx in semantic_graph):
                
                true_text_emb = text_embeddings[true_idx]
                sem_text_emb = text_embeddings[sem_idx]
                img_emb = image_embeddings[i]
                
                # Current similarities
                true_similarity = torch.cosine_similarity(img_emb.unsqueeze(0), true_text_emb.unsqueeze(0))
                sem_similarity = torch.cosine_similarity(img_emb.unsqueeze(0), sem_text_emb.unsqueeze(0))
                
                # Get WordNet semantic distance
                wordnet_distance = 1.0  # Default
                if sem_idx in semantic_graph[true_idx]['semantic_distances']:
                    wordnet_distance = semantic_graph[true_idx]['semantic_distances'][sem_idx]
                
                # Adaptive flow based on semantic distance
                # For very different concepts (high distance), strong flow
                # For similar concepts, gentler flow
                flow_intensity = min(1.0, wordnet_distance * 1.5)
                
                # Semantic flow: push away from true, pull toward semantic with intensity
                flow_component = flow_intensity * (true_similarity - sem_similarity)
                semantic_embedding_loss = semantic_embedding_loss + flow_component
        
        semantic_embedding_loss = semantic_embedding_loss / batch_size
        
        # =====================================================================
        # 2. CONCEPTUAL PATH FLOW (following WordNet paths)
        # =====================================================================
        conceptual_path_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if semantic_similarity_matrix is not None:
            # Use semantic similarity matrix to guide intermediate concepts
            for i in range(batch_size):
                true_idx = targets[i].item()
                sem_idx = semantic_targets[i].item()
                
                if (true_idx < semantic_similarity_matrix.size(0) and 
                    sem_idx < semantic_similarity_matrix.size(1)):
                    
                    # Find intermediate concepts along semantic path
                    semantic_similarities = semantic_similarity_matrix[true_idx]
                    
                    # Get top-k intermediate concepts
                    k = min(5, semantic_similarities.size(0) - 1)
                    _, intermediate_indices = torch.topk(semantic_similarities, k)
                    
                    img_emb = image_embeddings[i]
                    
                    # Flow through intermediate concepts
                    for intermediate_idx in intermediate_indices:
                        if (intermediate_idx != true_idx and 
                            intermediate_idx < text_embeddings.size(0)):
                            
                            intermediate_emb = text_embeddings[intermediate_idx]
                            intermediate_similarity = torch.cosine_similarity(
                                img_emb.unsqueeze(0), intermediate_emb.unsqueeze(0))
                            
                            # Weight by semantic path relevance
                            path_weight = semantic_similarities[intermediate_idx]
                            conceptual_path_loss = conceptual_path_loss + path_weight * intermediate_similarity
        
        # =====================================================================
        # 3. ADAPTIVE SEMANTIC CONSISTENCY
        # =====================================================================
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure semantic targeting respects linguistic relationships
        for i in range(batch_size):
            true_idx = targets[i].item()
            sem_idx = semantic_targets[i].item()
            
            if (true_idx in semantic_graph and 
                sem_idx in semantic_graph[true_idx]['semantic_distances']):
                
                expected_distance = semantic_graph[true_idx]['semantic_distances'][sem_idx]
                
                # Compute actual distance in embedding space
                if (true_idx < text_embeddings.size(0) and 
                    sem_idx < text_embeddings.size(0)):
                    
                    true_emb = text_embeddings[true_idx]
                    sem_emb = text_embeddings[sem_idx]
                    img_emb = image_embeddings[i]
                    
                    true_sim = torch.cosine_similarity(img_emb.unsqueeze(0), true_emb.unsqueeze(0))
                    sem_sim = torch.cosine_similarity(img_emb.unsqueeze(0), sem_emb.unsqueeze(0))
                    
                    # Actual distance in embedding space
                    actual_distance = torch.sigmoid(true_sim - sem_sim)
                    
                    # Consistency: actual should match expected semantic distance
                    consistency_component = (actual_distance - expected_distance).pow(2)
                    consistency_loss = consistency_loss + consistency_component
        
        consistency_loss = consistency_loss / batch_size
        
        # =====================================================================
        # 4. PROGRESSIVE FLOW STRENGTH (adaptive over iterations)
        # =====================================================================
        
        # Start with gentle flow, increase intensity
        progress = iter_idx / attack_iters
        adaptive_flow_strength = flow_strength * (0.5 + 0.5 * progress)
        
        # Combine all semantic flow components
        total_semantic_flow = (
            adaptive_flow_strength * semantic_embedding_loss +
            0.3 * conceptual_path_loss +
            0.2 * consistency_loss
        )
        
        return total_semantic_flow
    
    # ========================================================================
    # MAIN ATTACK IMPLEMENTATION
    # ========================================================================
    
    print("Enhanced Semantic Flow Attack with WordNet Integration")
    print(f"Strategy: {semantic_strategy}")
    print(f"Dataset: {dataset_name}")
    print(f"Flow strength: {flow_strength}")
    
    # Build WordNet semantic graph
    print("Building WordNet semantic graph...")
    semantic_graph = build_wordnet_semantic_graph(dataset_classes)
    
    # Extract semantic embeddings
    print("Extracting semantic embeddings...")
    text_embeddings, semantic_similarity_matrix = extract_semantic_embeddings(model, text_tokens, semantic_graph)
    
    best_delta = None
    best_attack_rate = -1
    final_semantic_targets = None
    
    for restart_idx in range(restarts):
        print(f"\nSemantic Flow Restart {restart_idx + 1}/{restarts}")
        
        # Initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        
        # Semantic flow momentum buffers
        momentum_buffer = torch.zeros_like(delta)
        semantic_velocity = torch.zeros_like(delta)  # For semantic-guided momentum
        
        # Text prompt optimizer
        if prompt_learner is not None:
            prompter_optim = torch.optim.SGD(prompt_learner.parameters(),
                                          lr=text_perb_stepsize, momentum=0, weight_decay=0)
            prompt_original_state = {k: v.clone() for k, v in prompt_learner.state_dict().items()}
        
        # Select semantic targets using WordNet
        semantic_targets = select_semantic_targets_wordnet(
            target, semantic_graph, text_embeddings, semantic_strategy, len(dataset_classes))
        
        if restart_idx == 0:
            final_semantic_targets = semantic_targets.clone()
        
        # Semantic flow iterations
        for iter_idx in range(attack_iters):
            # Forward pass
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            # Get outputs and embeddings
            try:
                if prompt_learner is not None:
                    output, _, image_embeddings, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                        model, prompted_images, text_tokens, prompt_token, prompt_learner, is_embedding=True)
                    if prompt_learner is not None:
                        prompter_optim.zero_grad()
                else:
                    output, _, image_embeddings, _ = multiGPU_CLIP(
                        model, prompted_images, text_tokens, prompt_token, is_embedding=True)
                
                if image_embeddings is not None:
                    image_embeddings = F.normalize(image_embeddings, dim=-1)
                    
            except Exception:
                if prompt_learner is not None:
                    output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                        model, prompted_images, text_tokens, prompt_token, prompt_learner)
                    if prompt_learner is not None:
                        prompter_optim.zero_grad()
                else:
                    output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
                image_embeddings = None
            
            # Compute losses
            adversarial_loss = criterion(output, target)
            semantic_targeting_loss = -criterion(output, semantic_targets)
            
            # Enhanced semantic flow loss with WordNet guidance
            semantic_flow_loss = compute_semantic_flow_loss(
                output, image_embeddings, text_embeddings, target, semantic_targets,
                semantic_graph, semantic_similarity_matrix, flow_strength, iter_idx)
            
            # Combined loss
            total_loss = (
                embedding_loss_weight * adversarial_loss +
                semantic_anchor_weight * semantic_targeting_loss +
                contrastive_loss_weight * semantic_flow_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Get gradient
            grad = delta.grad.detach()
            
            # Semantic-guided momentum update
            # Standard momentum
            momentum_buffer = momentum * momentum_buffer + grad
            
            # Semantic velocity (flows along WordNet relationships)
            semantic_gradient_weight = min(1.0, iter_idx / (attack_iters * 0.3))  # Increase over time
            semantic_velocity = momentum * semantic_velocity + semantic_gradient_weight * grad
            
            # Combine momentum and semantic velocity
            combined_momentum = 0.7 * momentum_buffer + 0.3 * semantic_velocity
            
            # Adaptive step size based on semantic flow progress
            flow_progress = semantic_flow_loss.abs().item()
            adaptive_alpha = alpha * (1.0 + 0.5 * flow_progress)  # Increase step when flow is strong
            
            # Update perturbation
            d = delta[:, :, :, :]
            x = X[:, :, :, :]
            
            if norm == "l_inf":
                d = torch.clamp(d + adaptive_alpha * torch.sign(combined_momentum), 
                               min=-epsilon, max=epsilon)
            elif norm == "l_2":
                momentum_norm = torch.norm(combined_momentum.view(combined_momentum.shape[0], -1), 
                                         dim=1).view(-1, 1, 1, 1)
                scaled_momentum = combined_momentum / (momentum_norm + 1e-10)
                d = (d + scaled_momentum * adaptive_alpha).view(d.size(0), -1).renorm(
                    p=2, dim=0, maxnorm=epsilon).view_as(d)
            
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
            
            # Update text perturbation
            if prompt_learner is not None:
                for param in prompt_learner.parameters():
                    if param.grad is not None:
                        param.grad.data = -1.0 * param.grad.data
                prompter_optim.step()
            
            # Debug output
            if iter_idx % 10 == 0:
                with torch.no_grad():
                    pred_classes = output.argmax(dim=1)
                    correct_rate = (pred_classes == target).float().mean()
                    semantic_rate = (pred_classes == semantic_targets).float().mean()
                    attack_success = (pred_classes != target).float().mean()
                    
                    print(f"    Iter {iter_idx:2d}: Loss={total_loss.item():.3f} | "
                          f"Success={attack_success:.3f} | Semantic={semantic_rate:.3f} | "
                          f"Flow={semantic_flow_loss.item():.4f} | Alpha={adaptive_alpha:.4f}")
        
        # Evaluate restart
        with torch.no_grad():
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            if prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
            
            pred_classes = output.argmax(dim=1)
            attack_rate = (pred_classes != target).float().mean()
            semantic_rate = (pred_classes == semantic_targets).float().mean()
            
            if attack_rate > best_attack_rate:
                best_attack_rate = attack_rate
                best_delta = delta.clone()
                print(f"    New best attack rate: {attack_rate:.3f} (semantic: {semantic_rate:.3f})")
        
        # Reset prompt learner
        if prompt_learner is not None:
            prompt_learner.load_state_dict(prompt_original_state)
    
    print(f"\nFinal Enhanced Semantic Flow Results:")
    print(f"Best attack success rate: {best_attack_rate:.3f}")
    print(f"Strategy: {semantic_strategy}")
    print(f"WordNet relationships utilized: {len([k for k in semantic_graph.keys() if semantic_graph[k]['synset'] is not None])}/{len(semantic_graph)}")
    
    return best_delta, final_semantic_targets

# ========================================================================
# ADDITIONAL SEMANTIC STRATEGIES FOR ROBUST MODELS
# ========================================================================

def attack_semantic_flow_robust_enhanced(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
                                        attack_iters, norm, prompt_learner=None, text_perb_stepsize=0.01,
                                        restarts=1, early_stop=True, epsilon=0, semantic_strategy="confusing",
                                        embedding_loss_weight=0.3, semantic_anchor_weight=0.2, 
                                        contrastive_loss_weight=0.1, dataset_classes=None, 
                                        original_dataset_classes=None, momentum=0.9, 
                                        temperature=0.1, flow_strength=0.5, dataset_name="",
                                        model_type="CLIP"):
    
    # Model-specific parameter adjustments
    robust_configs = {
        'TRADES': {
            'flow_strength': 0.7,
            'momentum': 0.95, 
            'strategies': ['opposite', 'confusing', 'coordinate'],
            'adaptive_alpha': True
        },
        'TeCoA': {
            'flow_strength': 0.6,
            'momentum': 0.9,
            'strategies': ['antonym', 'opposite', 'hypernym'],
            'adaptive_alpha': True
        },
        'FARE': {
            'flow_strength': 0.65,
            'momentum': 0.92,
            'strategies': ['confusing', 'meronym', 'opposite'],
            'adaptive_alpha': True
        },
        'PMG-AFT': {
            'flow_strength': 0.5,
            'momentum': 0.9,
            'strategies': ['coordinate', 'confusing'],
            'adaptive_alpha': False
        },
        'AUDIENCE': {
            'flow_strength': 0.8,  # Higher for distilled models
            'momentum': 0.95,
            'strategies': ['adaptive', 'opposite', 'antonym'],
            'adaptive_alpha': True
        }
    }
    
    config = robust_configs.get(model_type, {
        'flow_strength': flow_strength,
        'momentum': momentum,
        'strategies': [semantic_strategy],
        'adaptive_alpha': False
    })
    
    print(f"Robust Semantic Flow Attack for {model_type}")
    print(f"Config: {config}")
    
    best_delta = None
    best_attack_rate = -1
    best_semantic_targets = None
    
    # Try multiple semantic strategies for robust models
    strategies_to_try = config['strategies']
    
    for strategy_idx, strategy in enumerate(strategies_to_try):
        print(f"\nTrying semantic strategy: {strategy} ({strategy_idx + 1}/{len(strategies_to_try)})")
        
        delta, semantic_targets = attack_semantic_flow_enhanced(
            prompter=prompter,
            model=model,
            add_prompter=add_prompter,
            criterion=criterion,
            X=X,
            target=target,
            text_tokens=text_tokens,
            alpha=alpha,
            attack_iters=attack_iters,
            norm=norm,
            prompt_learner=prompt_learner,
            text_perb_stepsize=text_perb_stepsize,
            restarts=max(1, restarts // len(strategies_to_try)),  # Distribute restarts
            early_stop=early_stop,
            epsilon=epsilon,
            semantic_strategy=strategy,
            embedding_loss_weight=embedding_loss_weight,
            semantic_anchor_weight=semantic_anchor_weight,
            contrastive_loss_weight=contrastive_loss_weight,
            dataset_classes=dataset_classes,
            original_dataset_classes=original_dataset_classes,
            momentum=config['momentum'],
            temperature=temperature,
            flow_strength=config['flow_strength'],
            dataset_name=dataset_name
        )
        
        # Evaluate this strategy
        with torch.no_grad():
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            if prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
            
            pred_classes = output.argmax(dim=1)
            attack_rate = (pred_classes != target).float().mean().item()
            
            print(f"Strategy {strategy} attack rate: {attack_rate:.3f}")
            
            if attack_rate > best_attack_rate:
                best_attack_rate = attack_rate
                best_delta = delta.clone()
                best_semantic_targets = semantic_targets.clone()
                print(f"New best strategy: {strategy} with rate {attack_rate:.3f}")
    
    print(f"\nFinal Robust Semantic Flow Results for {model_type}:")
    print(f"Best attack success rate: {best_attack_rate:.3f}")
    print(f"Strategies tested: {strategies_to_try}")
    
    return best_delta, best_semantic_targets


# ========================================================================
# INTEGRATION HELPER FUNCTIONS
# ========================================================================

def create_wordnet_enhanced_targeter(dataset_name, dataset_classes):
    class WordNetTargeter:
        def __init__(self, dataset_name, dataset_classes):
            self.dataset_name = dataset_name
            self.dataset_classes = dataset_classes
            self.semantic_graph = None
            self._build_graph()
        
        def _build_graph(self):
            try:
                import nltk
                from nltk.corpus import wordnet as wn
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet')
                    nltk.download('omw-1.4')
                
                self.semantic_graph = {}
                
                for i, class_name in enumerate(self.dataset_classes):
                    clean_name = class_name.lower().replace('_', ' ').replace('-', ' ')
                    synsets = wn.synsets(clean_name, pos=wn.NOUN)
                    
                    if synsets:
                        synset = synsets[0]
                        self.semantic_graph[i] = {
                            'synset': synset,
                            'hypernyms': [h.name().split('.')[0] for h in synset.hypernyms()],
                            'hyponyms': [h.name().split('.')[0] for h in synset.hyponyms()],
                            'antonyms': [ant.name() for lemma in synset.lemmas() for ant in lemma.antonyms()],
                            'coordinates': []
                        }
                        
                        # Get coordinates (siblings)
                        for hypernym in synset.hypernyms():
                            for coord in hypernym.hyponyms():
                                if coord != synset:
                                    self.semantic_graph[i]['coordinates'].extend([l.name() for l in coord.lemmas()])
                
            except ImportError:
                print("WordNet not available, using basic targeting")
                self.semantic_graph = {}
        
        def get_semantic_target(self, true_idx, strategy):
            if true_idx not in self.semantic_graph:
                return (true_idx + 1) % len(self.dataset_classes)
            
            graph_info = self.semantic_graph[true_idx]
            
            candidates = []
            
            if strategy == "opposite" and graph_info['antonyms']:
                # Find antonyms in dataset
                for antonym in graph_info['antonyms']:
                    for j, class_name in enumerate(self.dataset_classes):
                        if j != true_idx and antonym.lower() in class_name.lower():
                            candidates.append(j)
            
            elif strategy == "coordinate" and graph_info['coordinates']:
                # Find coordinate terms in dataset
                for coord in graph_info['coordinates']:
                    for j, class_name in enumerate(self.dataset_classes):
                        if j != true_idx and coord.lower() in class_name.lower():
                            candidates.append(j)
            
            elif strategy == "hypernym" and graph_info['hypernyms']:
                # Find hypernyms in dataset
                for hypernym in graph_info['hypernyms']:
                    for j, class_name in enumerate(self.dataset_classes):
                        if j != true_idx and hypernym.lower() in class_name.lower():
                            candidates.append(j)
            
            if candidates:
                return np.random.choice(candidates)
            else:
                return (true_idx + 1) % len(self.dataset_classes)
    
    return WordNetTargeter(dataset_name, dataset_classes)

def get_optimal_semantic_strategy(model_type, dataset_name):
    strategy_map = {
        'TRADES': {
            'cifar10': 'opposite',
            'cifar100': 'confusing', 
            'ImageNet': 'antonym',
            'default': 'opposite'
        },
        'TeCoA': {
            'cifar10': 'confusing',
            'cifar100': 'coordinate',
            'ImageNet': 'hypernym',
            'default': 'confusing'
        },
        'FARE': {
            'cifar10': 'opposite',
            'cifar100': 'meronym',
            'ImageNet': 'antonym',
            'default': 'opposite'
        },
        'PMG-AFT': {
            'cifar10': 'coordinate',
            'cifar100': 'confusing',
            'ImageNet': 'hypernym',
            'default': 'confusing'
        },
        'AUDIENCE': {
            'cifar10': 'adaptive',
            'cifar100': 'antonym',
            'ImageNet': 'opposite',
            'default': 'adaptive'
        },
        'CLIP': {
            'default': 'confusing'
        }
    }
    
    model_strategies = strategy_map.get(model_type, strategy_map['CLIP'])
    return model_strategies.get(dataset_name, model_strategies['default'])


def create_semantic_targeter(dataset_name, class_names, text_embeddings=None, debug=False):
    if debug:
        print(f"\n🔧 Creating semantic targeter for dataset: {dataset_name}")
    return SimpleSemanticTargeter(dataset_name, class_names, text_embeddings, debug=debug)

def debug_semantic_selection(true_targets, semantic_targets, dataset_classes, strategy):
    print(f"\n=== SEMANTIC SELECTION DEBUG ({strategy}) ===")
    
    for i in range(min(5, len(true_targets))):
        true_idx = true_targets[i].item()
        sem_idx = semantic_targets[i].item()
        
        print(f"Image {i}:")
        print(f"  True: {dataset_classes[true_idx]}")
        print(f"  Semantic target: {dataset_classes[sem_idx]}")
        
        # Check if they're the same (common bug)
        if true_idx == sem_idx:
            print(f"  ERROR: Semantic target same as true target!")
        
        # Check if semantic target makes sense for strategy
        if strategy == "similar":
            print(f"  Should be similar to '{dataset_classes[true_idx]}'")
        elif strategy == "opposite":
            print(f"  Should be opposite to '{dataset_classes[true_idx]}'")

# this is the main function for the CLIP-Specific adversarial calibration for semantic dissonance 
def calibrate_clip_semantics(prompter, model, add_prompter, criterion, X, target, text_tokens, alpha,
                                attack_iters, norm, prompt_learner=None, text_perb_stepsize=0.01,
                                restarts=1, early_stop=True, epsilon=0, semantic_strategy="confusing",
                                embedding_loss_weight=0.3, semantic_anchor_weight=0.2, contrastive_loss_weight=0.1):
   
    lower_limit, upper_limit = 0, 1
    
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    def get_text_features_properly(model, text_tokens):
        try:
            with torch.no_grad():
                # Method 1: Direct text encoding
                if hasattr(model, 'encode_text'):
                    text_features = model.encode_text(text_tokens)
                    return F.normalize(text_features, dim=-1)
                else:
                    # Method 2: Through the model's text transformer
                    x = text_tokens
                    x = model.token_embedding(x)  # [batch_size, n_ctx, d_model]
                    x = x + model.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = model.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = model.ln_final(x)
                    # Take features from the eot embedding
                    text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
                    return F.normalize(text_features, dim=-1)
        except Exception as e:
            print(f"Could not extract text features: {e}")
            return None
    
    def select_semantic_targets_robust(targets, text_features, strategy, num_classes):
        if text_features is None or text_features.size(0) < num_classes:
            # Fallback: select random different classes
            target_classes = []
            for i in range(targets.size(0)):
                true_class = targets[i].item()
                # Simple strategy: pick a random different class
                wrong_classes = list(range(num_classes))
                wrong_classes.remove(true_class)
                semantic_target = np.random.choice(wrong_classes) if wrong_classes else (true_class + 1) % num_classes
                target_classes.append(semantic_target)
            return torch.tensor(target_classes, device=targets.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_features, text_features.T)
        target_classes = []
        
        for i in range(targets.size(0)):
            true_class = targets[i].item()
            
            if true_class >= similarity_matrix.size(0):
                # Fallback for out-of-range classes
                semantic_target = (true_class + 1) % num_classes
            else:
                similarities = similarity_matrix[true_class]
                similarities[true_class] = -2  # Mask out true class
                
                if strategy == "confusing":
                    # Choose moderately similar classes
                    sorted_sims, sorted_indices = torch.sort(similarities, descending=True)
                    # Take from middle range for confusion
                    mid_range = len(sorted_indices) // 3
                    choice_idx = mid_range + np.random.randint(0, max(1, mid_range))
                    choice_idx = min(choice_idx, len(sorted_indices) - 1)
                    semantic_target = sorted_indices[choice_idx].item()
                elif strategy == "similar":
                    semantic_target = torch.argmax(similarities).item()
                elif strategy == "opposite":
                    semantic_target = torch.argmin(similarities).item()
                else:  # random
                    valid_indices = [j for j in range(num_classes) if j != true_class]
                    semantic_target = np.random.choice(valid_indices) if valid_indices else (true_class + 1) % num_classes
            
            target_classes.append(semantic_target)
        
        return torch.tensor(target_classes, device=targets.device)
    
    # this loss greatly improve the CLIP accuracy for various datasets
    def compute_sapa_loss(output, targets, semantic_targets):
        batch_size = output.size(0)
        
        # Extract logits for correct and semantic classes
        correct_logits = output.gather(1, targets.unsqueeze(1)).squeeze(1)
        semantic_logits = output.gather(1, semantic_targets.unsqueeze(1)).squeeze(1)

        # 1. Primary adversarial loss: reduce correct class confidence
        # decrease correct logits, increase semantic logits

        # correct_probs = F.softmax(output, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        # adversarial_loss = correct_probs.mean()  # Minimize correct predictions
        # Component 1: Adversarial Classification Loss (on logits, not probabilities)
        L_adv = -correct_logits.mean() + semantic_logits.mean()
        
         # Component 2: Semantic Cross-Entropy Loss (difference of CE losses)
        ce_correct = criterion(output, targets)  # CE loss for correct class
        ce_semantic = criterion(output, semantic_targets)  # CE loss for semantic class
        L_sem = -ce_correct + ce_semantic
        
        # Component 3: Margin-Based Loss (C&W style)
        margin = 5.0
        L_margin = torch.clamp(semantic_logits - correct_logits - margin, min=0).mean()
        
        # Component 4: Confidence Reduction Loss
        correct_probs = F.softmax(output, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        L_conf = correct_probs.mean()
          
        # Combine losses with proper weighting
        total_loss = (embedding_loss_weight * L_adv + 
                     semantic_anchor_weight * L_sem + 
                     contrastive_loss_weight * L_margin +
                     0.1 * L_conf)
        
        return total_loss
    # Extract text features properly
    text_features = get_text_features_properly(model, text_tokens)
    num_classes = text_tokens.size(0)
    
    # Main attack loop
    best_attack_rate = -1
    best_delta = None
    
    for restart_idx in range(restarts):
        # Initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError("Norm must be either 'l_inf' or 'l_2'")
        
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        
        # Text prompt optimizer
        if prompt_learner is not None:
            prompter_optim = torch.optim.SGD(prompt_learner.parameters(),
                                          lr=text_perb_stepsize, momentum=0, weight_decay=0)
            prompt_original_state = {k: v.clone() for k, v in prompt_learner.state_dict().items()}
        
        # Select semantic targets for this restart
        semantic_targets = select_semantic_targets_robust(target, text_features, semantic_strategy, num_classes)
        
        print(f"SAPA Restart {restart_idx}: targets {target[:3].cpu().numpy()} -> semantic {semantic_targets[:3].cpu().numpy()}")
        
        # Attack iterations
        for iter_idx in range(attack_iters):
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            # Forward pass
            if prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
                prompter_optim.zero_grad()
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
            
            # SAPA loss computation
            loss = compute_sapa_loss(output, target, semantic_targets)
            
            # Debug output
            if iter_idx % 10 == 0:
                with torch.no_grad():
                    pred_classes = output.argmax(dim=1)
                    correct_rate = (pred_classes == target).float().mean()
                    semantic_rate = (pred_classes == semantic_targets).float().mean()
                    print(f"  Iter {iter_idx}: Loss={loss.item():.3f}, Correct={correct_rate:.3f}, Semantic={semantic_rate:.3f}")
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
            
            # Update text perturbation
            if prompt_learner is not None:
                for param in prompt_learner.parameters():
                    if param.grad is not None:
                        param.grad.data = -1.0 * param.grad.data
                prompter_optim.step()
        
        # Evaluate this restart using attack success rate
        with torch.no_grad():
            _images = clip_img_preprocessing(X + delta, model=model)
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images
            
            prompt_token = add_prompter() if add_prompter is not None else None
            
            if prompt_learner is not None:
                output, _ = multiGPU_CLIP_Text_Prompt_Tuning(
                    model, prompted_images, text_tokens, prompt_token, prompt_learner)
            else:
                output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)
            
            # Use attack success rate for restart selection
            pred_classes = output.argmax(dim=1)
            attack_rate = (pred_classes != target).float().mean()
            
            if attack_rate > best_attack_rate:
                best_attack_rate = attack_rate
                best_delta = delta.clone()
                print(f"  New best attack rate: {attack_rate:.3f}")
        
        # Reset prompt learner
        if prompt_learner is not None:
            prompt_learner.load_state_dict(prompt_original_state)
    
    print(f"Final SAPA attack success rate: {best_attack_rate:.3f}")
    return best_delta


def generate_rich_target_prompt(target_class: str, descriptions: Optional[List[str]] = None) -> str:
    if descriptions is None:
        # Rich descriptions for common classes
        rich_descriptions = {
            "butterfly": "a vibrant monarch butterfly with orange and black wings perched delicately on a purple flower",
            "dog": "a golden retriever dog sitting in a sunny meadow with its tongue out",
            "cat": "a fluffy orange tabby cat with green eyes lying on a soft blue cushion",
            "bird": "a colorful robin with a red breast perched on a cherry blossom branch",
            "car": "a sleek red sports car parked on a city street with chrome details gleaming",
            "airplane": "a large commercial airplane with white fuselage flying through blue skies",
            "ship": "a large cruise ship with white hull sailing on calm blue ocean waters",
            "truck": "a heavy-duty pickup truck with chrome bumper parked on gravel road",
            "horse": "a beautiful brown stallion galloping through an open green meadow",
            "frog": "a bright green tree frog sitting on a large lily pad in a pond",
            "deer": "a graceful white-tailed deer standing alert in a forest clearing",
            "automobile": "a classic vintage car with polished chrome details and leather interior",
        }
        
        if target_class.lower() in rich_descriptions:
            return f"This is an image of {rich_descriptions[target_class.lower()]}"
        else:
            return f"This is an image of a beautiful {target_class} in a natural setting"
    else:
        return descriptions[0] if descriptions else f"This is an image of a {target_class}"

