#!/usr/bin/env python3
"""
WGSMA (WordNet-Guided Semantic Multi-Modal Attack) on ImageNet-100

This script implements the complete WGSMA attack pipeline with configurable epsilon.
Priority 1B: Cross-Model STA Evaluation for circularity validation.

Usage:
    python run_semantic_attack_config.py --epsilon 0.031
    python run_semantic_attack_config.py --epsilon 0.062 --num_samples 200
    python run_semantic_attack_config.py --epsilon 0.015 --attack_iters 50
    python run_semantic_attack_config.py --model TeCoA --epsilon 0.031
    python run_semantic_attack_config.py --compute_cross_model_sta  # Priority 1B
"""

import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
import json
import os
import sys
import itertools

# Import your existing modules
from replace.tv_datasets.dataset_ops import get_dataset
from utils import refine_classname

# Import WGSMA components
from sapa.semantic_wordnet_anchor import WordNetSemanticAnchor
from sapa.semantic_feature_perturbation import SemanticFeatureSpacePerturbation
from sapa.llava_text_adaption import LLaVATextAdaptation

# Import model loader
from replace.model_loader import (
    load_model, load_clip, load_tecoa, load_fare, load_pmg,
    load_trades, load_audience, load_mobileclip, load_mobileclip2,
    get_text_tokens
)

# Priority 1B: Cross-Model STA
try:
    from paper.cross_model_sta_integrated import CrossModelSTACalculator
    CROSS_MODEL_STA_AVAILABLE = True
except ImportError:
    CROSS_MODEL_STA_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='SAPA Attack with Configurable Epsilon')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        help='Dataset name (default: ImageNet)')
    parser.add_argument('--data_path', type=str, default='./datasets/',
                        help='Path to dataset')
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help='Number of test samples (default: 100)')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=0.031,
                        help='L-infinity perturbation budget (default: 0.031 = 8/255)')
    parser.add_argument('--attack_iters', type=int, default=30,
                        help='Number of attack iterations (default: 30)')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Attack step size (default: 0.01)')
    parser.add_argument('--tta_frequency', type=int, default=10,
                        help='LLaVA TTA update frequency (default: 10)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='CLIP',
                        choices=['CLIP', 'TeCoA', 'FARE', 'PMG', 'TRADES', 'AUDIENCE',
                                 'MobileCLIP', 'MobileCLIP2'],
                        help='Model to attack (default: CLIP)')
    parser.add_argument('--arch', type=str, default='ViT-B/32',
                        help='Model architecture (default: ViT-B/32)')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='Directory containing model checkpoints')
    parser.add_argument('--mobileclip_dir', type=str, default='./models/mobileclip_models/',
                        help='Directory containing MobileCLIP checkpoints')
    
    # Misc parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device (default: 0)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory (default: ./results)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save adversarial images')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    # Priority 1B: Cross-Model STA
    parser.add_argument('--compute_cross_model_sta', action='store_true',
                        help='Compute cross-model STA using independent evaluation model (Priority 1B)')
    parser.add_argument('--eval_model', type=str, default='openclip_vit_l',
                        choices=['openclip_vit_l', 'openclip_vit_h', 'siglip'],
                        help='Independent model for cross-model STA evaluation')

    return parser.parse_args()


def print_attack_config(args):
    print("="*70)
    print("SAPA ATTACK CONFIGURATION")
    print("="*70)
    print(f"Dataset:              {args.dataset}")
    print(f"Test samples:         {args.num_test_samples}")
    print(f"")
    print(f"Model:")
    print(f"  Type:               {args.model}")
    print(f"  Architecture:       {args.arch}")
    print(f"")
    print(f"Attack Parameters:")
    print(f"  Epsilon (L∞):       {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"  Attack iterations:  {args.attack_iters}")
    print(f"  Step size (alpha):  {args.alpha}")
    print(f"  TTA frequency:      {args.tta_frequency}")
    print(f"")
    print(f"Device:               cuda:{args.gpu}")
    print(f"Random seed:          {args.seed}")
    print("="*70 + "\n")


def load_target_model(args, device, class_names=None):
    model_type = args.model.upper()
    
    print(f"\nLoading {args.model} model...")
    
    # Load model based on type
    if model_type == 'CLIP':
        model, preprocess, prompter, add_prompter, prompt_learner = load_clip(
            arch=args.arch,
            device=device
        )
    elif model_type == 'TECOA':
        model, preprocess, prompter, add_prompter, prompt_learner = load_tecoa(
            arch=args.arch,
            device=device,
            model_dir=args.model_dir
        )
    elif model_type == 'FARE':
        model, preprocess, prompter, add_prompter, prompt_learner = load_fare(
            arch=args.arch,
            device=device,
            model_dir=args.model_dir
        )
    elif model_type == 'PMG':
        model, preprocess, prompter, add_prompter, prompt_learner = load_pmg(
            arch=args.arch,
            device=device,
            model_dir=args.model_dir
        )
    elif model_type == 'TRADES':
        model, preprocess, prompter, add_prompter, prompt_learner = load_trades(
            arch=args.arch,
            device=device,
            model_dir=args.model_dir
        )
    elif model_type == 'AUDIENCE':
        if class_names is None:
            raise ValueError("AUDIENCE model requires class_names. Load dataset first.")
        model, preprocess, prompter, add_prompter, prompt_learner = load_audience(
            arch=args.arch,
            device=device,
            model_dir=args.model_dir,
            class_names=class_names
        )
    elif model_type == 'MOBILECLIP':
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip(
            arch=args.arch,
            device=device,
            mobileclip_dir=args.mobileclip_dir
        )
    elif model_type == 'MOBILECLIP2':
        model, preprocess, prompter, add_prompter, prompt_learner = load_mobileclip2(
            arch=args.arch,
            device=device,
            mobileclip_dir=args.mobileclip_dir
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print(f" Loaded {args.model} ({args.arch})")
    print(f"  Is OpenCLIP: {getattr(model, 'is_openclip', False)}")
    print(f"  Has PromptLearner: {prompt_learner is not None}")
    
    return model, preprocess, prompter, add_prompter, prompt_learner

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Print configuration
    print_attack_config(args)
    
    # ===== 1. Load dataset =====
    print("Loading dataset...")
    train_loader, test_loader, class_names, original_class_names = get_dataset(args)
    
    print(f"\nDataset Info:")
    print(f"  Classes: {len(class_names)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # ===== 2. Initialize SAPA components =====
    
    # Load target model using model_loader
    model, preprocess, prompter, add_prompter, prompt_learner = load_target_model(
        args, device, class_names
    )
    
    # Get the underlying CLIP model for SAPA components
    # The wrapper provides access to the original model
    clip_model = model.model if hasattr(model, 'model') else model
    
    # Load LLaVA
    print("\nLoading LLaVA...")
    try:
        llava_adapter = LLaVATextAdaptation(device=device)
    except Exception as e:
        print(f"⚠ Warning: Could not load LLaVA: {e}")
        print("Continuing without LLaVA (TTA will be skipped)...")
        llava_adapter = None
    
    # Initialize WordNet Anchor
    print("\nInitializing WordNet Semantic Anchor...")
    wordnet_anchor = WordNetSemanticAnchor(
        clip_model, 
        device, 
        class_names
    )
    
    # Initialize Semantic Anchor Finder (CLIP-based for better coverage)
    print("\nInitializing CLIP Semantic Anchor...")
    from sapa.semantic_clip_anchor import CLIPSemanticAnchor
    
    clip_anchor = CLIPSemanticAnchor(
        clip_model=clip_model,
        device=device,
        class_names=class_names
    )
    
    # Print similarity stats
    clip_anchor.print_similarity_stats()

    # Initialize SAPA Attack
    print("\nInitializing SAPA Attack...")
    attack = SemanticFeatureSpacePerturbation(
        clip_model=clip_model,
        llava_adapter=llava_adapter,
        device=device,
        class_names=class_names
    )
    
    print(f"\n{'='*70}")
    print(f"SAPA Components Initialized")
    print(f"  Target Model: {args.model} ({args.arch})")
    print(f"{'='*70}\n")

    # Priority 1B: Initialize Cross-Model STA Calculator
    cross_model_calculator = None
    if args.compute_cross_model_sta:
        if not CROSS_MODEL_STA_AVAILABLE:
            print("\n⚠ Cross-model STA requested but open_clip not available")
            print("  Install with: pip install open_clip_torch")
        else:
            print(f"\nLoading independent evaluation model for cross-model STA: {args.eval_model}...")
            try:
                cross_model_calculator = CrossModelSTACalculator(
                    eval_model_name=args.eval_model,
                    device=device
                )
                if cross_model_calculator.available:
                    print(f" Cross-model STA evaluation enabled")
                else:
                    print(f" Cross-model STA calculator not available")
                    cross_model_calculator = None
            except Exception as e:
                print(f" Failed to initialize cross-model STA calculator: {e}")
                cross_model_calculator = None

    # ===== 3. Run experiments =====
    # ===================================================================
    # Run attacks
    # ===================================================================
    results = []
    skipped_no_anchor = 0
    skipped_errors = 0
    total_samples = 0
    anchor_cache = {}
    
    # Calculate how many batches we need
    num_batches_needed = (args.num_test_samples + args.batch_size - 1) // args.batch_size
    
    print(f"\nStarting attack on {len(test_loader)} batches...")
    print(f"Batch size: {args.batch_size}")
    print(f"Expected samples: {min(args.num_test_samples, len(test_loader.dataset))}")
    
    done = False  # Flag to break outer loop
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Processing batches")):
        if done:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(len(images)):
            total_samples += 1
            
            if total_samples > args.num_test_samples:
                done = True  # Set flag to break outer loop
                break
            
            try:
                image = images[i]
                label = labels[i].item()
                
                # Get semantic anchor using CLIP
                if label in anchor_cache:
                    anchor = anchor_cache[label]
                else:
                    anchor = clip_anchor.find_semantic_anchor(
                        label,
                        strategy="similar",
                        similarity_range=(0.5, 0.85)
                    )
                    anchor_cache[label] = anchor
                
                if anchor is None:
                    skipped_no_anchor += 1
                    continue
                
                # Run attack
                adv_image, attack_info = attack.attack(
                    image,
                    label,
                    anchor,
                    epsilon=args.epsilon,
                    attack_iters=args.attack_iters,
                    alpha=args.alpha,
                    tta_frequency=args.tta_frequency
                )
                
                # Check for error
                if 'error' in attack_info:
                    print(f"    ✗ Attack error: {attack_info['error']}")
                    skipped_errors += 1
                    continue
                
                # Show attack details: true label → anchor → prediction
                true_class = class_names[label]
                pred_class = attack_info['pred_class']
                success_mark = "✓" if attack_info['success'] else "✗"
                
                print(f"  [{total_samples:3d}] {success_mark} "
                      f"True={true_class:20s} → Anchor={anchor:20s} → Pred={pred_class:20s} "
                      f"(align={attack_info['semantic_alignment']:.3f})")

                # Priority 1B: Compute Cross-Model STA
                cross_sta = None
                if cross_model_calculator is not None and attack_info['success']:
                    try:
                        # Get semantic target index (anchor class)
                        if anchor in class_names:
                            target_idx = class_names.index(anchor)
                        else:
                            # Fallback to predicted class
                            target_idx = attack_info.get('pred_class_idx', label)

                        # Debug: print tensor shape
                        if args.verbose:
                            print(f"      adv_image shape: {adv_image.shape}")

                        # Compute cross-model STA using independent model
                        # Ensure we have a 3D tensor (C, H, W) by squeezing all batch dims
                        adv_image_3d = adv_image.squeeze()
                        if args.verbose:
                            print(f"      adv_image_3d shape after squeeze: {adv_image_3d.shape}")

                        cross_sta = cross_model_calculator.compute_cross_model_sta(
                            adv_image_3d,
                            torch.tensor([target_idx], device=device),
                            class_names
                        )

                        if cross_sta is not None:
                            # Compute STA drop
                            sta_drop = attack_info['semantic_alignment'] - cross_sta
                            sta_drop_pct = (sta_drop / attack_info['semantic_alignment']) * 100 if attack_info['semantic_alignment'] > 0 else 0

                            if args.verbose:
                                print(f"      Cross-Model STA: {cross_sta:.3f} (drop: {sta_drop:.3f}, {sta_drop_pct:.1f}%)")
                    except Exception as e:
                        if args.verbose:
                            print(f"      Cross-model STA computation failed: {e}")

                # Collect results
                result_dict = {
                    'sample_idx': total_samples - 1,
                    'true_label': label,
                    'true_class': class_names[label],
                    'semantic_anchor': anchor,
                    'pred_class': attack_info['pred_class'],
                    'pred_class_idx': attack_info['pred_class_idx'],
                    'success': attack_info['success'],
                    'semantic_alignment': attack_info['semantic_alignment'],
                    'l_inf_norm': attack_info['l_inf_norm'],
                    'l2_norm': attack_info['l2_norm'],
                }

                # Priority 1B: Add cross-model STA to results
                if cross_sta is not None:
                    result_dict['cross_model_sta'] = cross_sta
                    result_dict['sta_drop'] = attack_info['semantic_alignment'] - cross_sta
                    result_dict['sta_drop_percentage'] = (result_dict['sta_drop'] / attack_info['semantic_alignment']) * 100 if attack_info['semantic_alignment'] > 0 else 0

                results.append(result_dict)
                
            except Exception as e:
                print(f"    ✗ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                skipped_errors += 1
                continue
        
        if total_samples >= args.num_test_samples:
            break
    
    # ===================================================================
    # Compute statistics (with safety checks)
    # ===================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Target Model: {args.model} ({args.arch})")
    print(f"\nAttack Configuration:")
    print(f"  Epsilon (L∞):         {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"  Attack iterations:    {args.attack_iters}")
    
    print(f"\nSamples:")
    print(f"  Total processed:      {total_samples}")
    
    if total_samples > 0:
        print(f"  Skipped (no anchor):  {skipped_no_anchor} ({skipped_no_anchor/total_samples*100:.1f}%)")
        print(f"  Skipped (errors):     {skipped_errors} ({skipped_errors/total_samples*100:.1f}%)")
        print(f"  Successfully attacked: {len(results)} ({len(results)/total_samples*100:.1f}%)")
    else:
        print(f"  Skipped (no anchor):  {skipped_no_anchor}")
        print(f"  Skipped (errors):     {skipped_errors}")
        print(f"  Successfully attacked: {len(results)}")
        print("\n⚠ WARNING: No samples were processed!")
    
    # Compute metrics only if we have results
    if len(results) > 0:
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        
        l_inf_norms = [r['l_inf_norm'] for r in results]
        l2_norms = [r['l2_norm'] for r in results]
        semantic_alignments = [r['semantic_alignment'] for r in results]
        
        avg_l_inf = np.mean(l_inf_norms)
        max_l_inf = np.max(l_inf_norms)
        avg_l2 = np.mean(l2_norms)
        avg_semantic = np.mean(semantic_alignments)

        print(f"\nAttack Success Rate: {success_rate:.2%} ({success_count}/{len(results)})")
        print(f"\nPerturbation Metrics:")
        print(f"  Avg L∞ norm:          {avg_l_inf:.6f} (budget: {args.epsilon:.6f})")
        print(f"  Max L∞ norm:          {max_l_inf:.6f}")
        print(f"  Avg L2 norm:          {avg_l2:.4f}")
        print(f"  L∞ budget usage:      {avg_l_inf/args.epsilon*100:.1f}%")
        print(f"\nSemantic Alignment:")
        print(f"  Avg alignment:        {avg_semantic:.4f}")

        # Priority 1B: Cross-Model STA Statistics
        cross_sta_values = [r.get('cross_model_sta') for r in results if r.get('cross_model_sta') is not None]
        if cross_sta_values:
            avg_cross_sta = np.mean(cross_sta_values)
            avg_sta_drop = np.mean([r.get('sta_drop', 0) for r in results if r.get('sta_drop') is not None])
            avg_sta_drop_pct = np.mean([r.get('sta_drop_percentage', 0) for r in results if r.get('sta_drop_percentage') is not None])

            # Compute correlation between within-model and cross-model STA
            within_sta = [r['semantic_alignment'] for r in results if r.get('cross_model_sta') is not None]
            if len(within_sta) > 1:
                correlation = np.corrcoef(within_sta, cross_sta_values)[0, 1]
            else:
                correlation = 0.0

            print(f"\nCross-Model STA (Priority 1B):")
            print(f"  Avg cross-model STA:  {avg_cross_sta:.4f}")
            print(f"  Avg STA drop:         {avg_sta_drop:.4f} ({avg_sta_drop_pct:.1f}%)")
            print(f"  Correlation (r):      {correlation:.3f}")
            print(f"  Num samples:          {len(cross_sta_values)}")

            # Interpretation
            if correlation > 0.7:
                print(f"  → Strong correlation indicates semantic coherence generalizes")
            elif correlation > 0.4:
                print(f"  → Moderate correlation indicates partial generalization")
            else:
                print(f"  → Weak correlation suggests limited generalization")

        # Save results
        num_successful = sum(1 for r in results if r.get('success', False))
        summary = {
            'total_samples': total_samples,
            'successfully_attacked': num_successful,
            'skipped_no_anchor': skipped_no_anchor,
            'skipped_errors': skipped_errors,
            'success_rate': success_rate,
            'avg_semantic_alignment': avg_semantic,
            'avg_l_inf_norm': avg_l_inf,
            'max_l_inf_norm': max_l_inf,
            'avg_l2_norm': avg_l2,
            'l_inf_budget_usage': avg_l_inf/args.epsilon*100,
        }

        # Priority 1B: Add cross-model STA to summary
        if cross_sta_values:
            summary['cross_model_sta'] = {
                'avg_cross_model_sta': float(avg_cross_sta),
                'avg_sta_drop': float(avg_sta_drop),
                'avg_sta_drop_percentage': float(avg_sta_drop_pct),
                'within_cross_correlation': float(correlation),
                'num_cross_model_evaluations': len(cross_sta_values),
                'evaluation_model': args.eval_model
            }
    else:
        print("\n No results to compute metrics!")
        summary = {
            'total_samples': total_samples,
            'successfully_attacked': 0,
            'skipped_no_anchor': skipped_no_anchor,
            'skipped_errors': skipped_errors,
            'success_rate': 0.0,
            'avg_semantic_alignment': 0.0,
            'avg_l_inf_norm': 0.0,
            'max_l_inf_norm': 0.0,
            'avg_l2_norm': 0.0,
            'l_inf_budget_usage': 0.0,
        }
    
    # ===================================================================
    # Print final summary
    # ===================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Target Model: {args.model} ({args.arch})")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon (L∞): {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Attack Iterations: {args.attack_iters}")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Successfully Attacked: {len(results)}")
    print(f"Skipped (No Anchor): {skipped_no_anchor}")
    print(f"Skipped (Errors): {skipped_errors}")
    
    if len(results) > 0:
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_alignment = sum(r['semantic_alignment'] for r in results) / len(results)
        avg_l_inf = np.mean([r['l_inf_norm'] for r in results])
        max_l_inf = np.max([r['l_inf_norm'] for r in results])
        avg_l2 = np.mean([r['l2_norm'] for r in results])
        
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Semantic Alignment: {avg_alignment:.3f}")
        print(f"Average L∞ Norm: {avg_l_inf:.4f} ({avg_l_inf/args.epsilon*100:.1f}% of budget)")
        print(f"Maximum L∞ Norm: {max_l_inf:.4f} ({max_l_inf/args.epsilon*100:.1f}% of budget)")
        print(f"Average L2 Norm: {avg_l2:.4f}")
    else:
        print("No results available for metrics computation.")
    
    print(f"{'='*70}\n")
    
    # ===================================================================
    # Save detailed results
    # ===================================================================
    if len(results) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(
            args.output_dir, 
            f'wgsma_{args.model}_{args.dataset}_eps{args.epsilon:.4f}_results.json'
        )
        
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'model': args.model,
                    'architecture': args.arch,
                    'epsilon': args.epsilon,
                    'attack_iters': args.attack_iters,
                    'alpha': args.alpha,
                    'tta_frequency': args.tta_frequency,
                    'num_test_samples': args.num_test_samples
                },
                'summary': summary,
                'detailed_results': results
            }, f, indent=2)
        
        print(f" Detailed results saved to: {output_file}")
    else:
        print("No detailed results to save.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
