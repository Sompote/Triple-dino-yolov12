#!/usr/bin/env python3
"""
Comprehensive model evaluation script for YOLOv12 Triple Input models.

This script evaluates trained models on train, validation, and test splits
and provides a comprehensive mAP report.

Usage:
    python evaluate_model_all_splits.py --model runs/detect/yolov12_triple_dinov3/weights/best.pt --data test_triple_dataset.yaml
    python evaluate_model_all_splits.py --model runs/detect/yolov12_triple_dinov3/weights/best.pt --data test_triple_dataset.yaml --splits val test
"""

import argparse
import torch
from pathlib import Path
import yaml
from ultralytics import YOLO

def evaluate_model_on_splits(model_path, data_config, splits=['train', 'val', 'test'], device='0'):
    """
    Evaluate a trained model on multiple data splits.
    
    Args:
        model_path: Path to trained model (.pt file)
        data_config: Path to dataset configuration (.yaml file)
        splits: List of splits to evaluate on
        device: Device to use for evaluation
        
    Returns:
        Dictionary containing results for each split
    """
    
    print("YOLOv12 Triple Input Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data Config: {data_config}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Check if data config exists
    if not Path(data_config).exists():
        print(f"❌ Data config not found: {data_config}")
        return None
    
    # Load data config to check available splits
    try:
        with open(data_config, 'r') as f:
            data_info = yaml.safe_load(f)
        print(f"✓ Data config loaded: {data_info.get('nc', 'unknown')} classes")
    except Exception as e:
        print(f"⚠️ Could not parse data config: {e}")
        data_info = {}
    
    results_summary = {}
    
    # Evaluate on each split
    for split in splits:
        print(f"\n📊 Evaluating on {split} set...")
        try:
            # Run validation on the specific split
            val_results = model.val(
                data=data_config,
                split=split,
                device=device,
                verbose=False  # Reduce output noise
            )
            
            # Extract metrics
            if hasattr(val_results, 'box'):
                map50 = val_results.box.map50
                map50_95 = val_results.box.map
                map_per_class = val_results.box.maps if hasattr(val_results.box, 'maps') else None
            else:
                # Fallback for different result formats
                map50 = getattr(val_results, 'map50', 0.0)
                map50_95 = getattr(val_results, 'map', 0.0)
                map_per_class = None
            
            # Store results
            results_summary[split] = {
                'map50': float(map50) if map50 is not None else 0.0,
                'map50_95': float(map50_95) if map50_95 is not None else 0.0,
                'map_per_class': map_per_class.tolist() if map_per_class is not None else None
            }
            
            print(f"  ✓ {split.capitalize()} mAP50: {results_summary[split]['map50']:.4f}")
            print(f"  ✓ {split.capitalize()} mAP50-95: {results_summary[split]['map50_95']:.4f}")
            
        except Exception as e:
            print(f"  ❌ {split.capitalize()} evaluation failed: {e}")
            results_summary[split] = {'error': str(e)}
    
    # Display comprehensive summary
    print(f"\n📈 Comprehensive mAP Summary")
    print("=" * 60)
    
    # Table header
    print(f"{'Split':<12} {'mAP50':<8} {'mAP50-95':<10} {'Status'}")
    print("-" * 60)
    
    # Table rows
    for split in splits:
        if split in results_summary and 'error' not in results_summary[split]:
            metrics = results_summary[split]
            print(f"{split.capitalize():<12} {metrics['map50']:<8.4f} {metrics['map50_95']:<10.4f} {'✓ Success'}")
        else:
            error_msg = results_summary.get(split, {}).get('error', 'Unknown error')
            print(f"{split.capitalize():<12} {'N/A':<8} {'N/A':<10} {'❌ Failed'}")
    
    print("=" * 60)
    
    # Performance analysis
    if len([s for s in results_summary.values() if 'error' not in s]) >= 2:
        print(f"\n🔍 Performance Analysis:")
        
        # Check for overfitting (train much higher than val/test)
        train_map = results_summary.get('train', {}).get('map50', 0)
        val_map = results_summary.get('val', {}).get('map50', 0)
        test_map = results_summary.get('test', {}).get('map50', 0)
        
        if train_map > 0 and val_map > 0:
            overfit_diff = train_map - val_map
            if overfit_diff > 0.1:  # 10% difference
                print(f"  ⚠️ Potential overfitting detected: Train mAP50 ({train_map:.4f}) >> Val mAP50 ({val_map:.4f})")
                print(f"     Difference: {overfit_diff:.4f}")
            else:
                print(f"  ✓ Good generalization: Train-Val difference: {overfit_diff:.4f}")
        
        # Check val vs test performance
        if val_map > 0 and test_map > 0:
            val_test_diff = abs(val_map - test_map)
            if val_test_diff < 0.05:  # Less than 5% difference
                print(f"  ✓ Consistent performance: Val-Test difference: {val_test_diff:.4f}")
            else:
                print(f"  ⚠️ Val-Test performance gap: {val_test_diff:.4f}")
    
    # Class-wise performance (if available)
    if data_info.get('names') and any('map_per_class' in r and r['map_per_class'] for r in results_summary.values()):
        print(f"\n📊 Class-wise Performance (mAP50):")
        class_names = data_info['names']
        
        for split, metrics in results_summary.items():
            if 'map_per_class' in metrics and metrics['map_per_class']:
                print(f"\n  {split.capitalize()} set:")
                for i, (class_name, class_map) in enumerate(zip(class_names, metrics['map_per_class'])):
                    print(f"    {class_name}: {class_map:.4f}")
    
    print(f"\n🎯 Recommendations:")
    
    # Provide recommendations based on results
    if len([s for s in results_summary.values() if 'error' not in s]) == 0:
        print("  • All evaluations failed - check model and data compatibility")
    else:
        best_split = max(
            [s for s in results_summary if 'error' not in results_summary[s]], 
            key=lambda s: results_summary[s]['map50']
        )
        best_map = results_summary[best_split]['map50']
        
        if best_map < 0.3:
            print("  • Low mAP scores - consider longer training, different model size, or data augmentation")
        elif best_map < 0.5:
            print("  • Moderate performance - consider hyperparameter tuning or model architecture changes")
        else:
            print("  • Good performance achieved!")
        
        # Model deployment recommendation
        if 'val' in results_summary and 'error' not in results_summary['val']:
            print(f"  • Use validation mAP50 ({results_summary['val']['map50']:.4f}) as primary metric for model selection")
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv12 Triple Input model on multiple data splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  # Evaluate on all splits:\n"
               "  python evaluate_model_all_splits.py --model best.pt --data dataset.yaml\n\n"
               "  # Evaluate on specific splits:\n"
               "  python evaluate_model_all_splits.py --model best.pt --data dataset.yaml --splits val test\n\n"
               "  # Use CPU:\n"
               "  python evaluate_model_all_splits.py --model best.pt --data dataset.yaml --device cpu"
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='Data splits to evaluate on (default: train val test)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (default: 0, use "cpu" for CPU)')
    parser.add_argument('--save-results', type=str,
                       help='Save results to JSON file (optional)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    # Run evaluation
    results = evaluate_model_on_splits(
        model_path=args.model,
        data_config=args.data,
        splits=args.splits,
        device=args.device
    )
    
    # Save results if requested
    if args.save_results and results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.save_results}")

if __name__ == "__main__":
    main()