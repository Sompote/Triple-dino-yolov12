#!/usr/bin/env python3
"""
Fixed training script for YOLOv12 Triple Input with DINOv3 backbone.
This version handles triple input datasets correctly.
"""

# Import from the original training script
from train_triple_dinov3 import train_triple_dinov3
import argparse
from pathlib import Path

def train_triple_dinov3_fixed(
    data_config: str,
    dinov3_size: str = "small",
    freeze_dinov3: bool = True,
    use_triple_branches: bool = False,
    pretrained_path: str = None,
    epochs: int = 100,
    batch_size: int = 8,
    imgsz: int = 224,
    patience: int = 50,
    name: str = "yolov12_triple_dinov3",
    device: str = "0",
    integrate: str = "initial",
    variant: str = "s",
    save_period: int = -1,
    **kwargs
):
    """
    Fixed training function that handles triple input datasets correctly.
    
    This version automatically detects triple input datasets and disables
    problematic augmentations that don't work with 9-channel images.
    """
    
    print("YOLOv12 Triple Input Training (Fixed for Triple Input Datasets)")
    print("=" * 70)
    
    # Check if this is a triple input dataset
    import yaml
    with open(data_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Check if dataset has triple input structure
    train_path = dataset_config.get('train', '')
    if train_path:
        base_path = Path(train_path).parent.parent
        expected_folders = ["primary", "detail1", "detail2"]
        is_triple_input = all((base_path / folder).exists() for folder in expected_folders)
        
        if is_triple_input:
            print("🔍 Triple input dataset detected!")
            print("📝 Automatically disabling problematic augmentations for 9-channel images")
            
            # Override kwargs to disable problematic augmentations
            triple_input_overrides = {
                # Disable all color-based augmentations (incompatible with 9-channel)
                'hsv_h': 0.0,
                'hsv_s': 0.0,
                'hsv_v': 0.0,
                'auto_augment': None,
                'bgr': 0.0,
                
                # Disable augmentations that may cause issues
                'mixup': 0.0,
                'copy_paste': 0.0,
                'erasing': 0.0,
                
                # Keep geometric augmentations (these should work)
                'degrees': kwargs.get('degrees', 0.0),
                'translate': kwargs.get('translate', 0.1),
                'scale': kwargs.get('scale', 0.5),
                'shear': kwargs.get('shear', 0.0),
                'perspective': kwargs.get('perspective', 0.0),
                'flipud': kwargs.get('flipud', 0.0),
                'fliplr': kwargs.get('fliplr', 0.5),
                
                # Disable mosaic for now (may need special handling)
                'mosaic': 0.0,
                'close_mosaic': 0,
                
                # Keep other settings
                'plots': False,  # Disable plots (visualization incompatible with 9-channel)
            }
            
            # Update kwargs with triple input overrides
            kwargs.update(triple_input_overrides)
            
            print("✅ Augmentation settings optimized for triple input")
        else:
            print("📷 Standard dataset detected - using normal augmentations")
    
    # Call the original training function with the potentially modified kwargs
    return train_triple_dinov3(
        data_config=data_config,
        dinov3_size=dinov3_size,
        freeze_dinov3=freeze_dinov3,
        use_triple_branches=use_triple_branches,
        pretrained_path=pretrained_path,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        patience=patience,
        name=name,
        device=device,
        integrate=integrate,
        variant=variant,
        save_period=save_period,
        **kwargs
    )

def print_dinov3_info():
    """Print detailed DINOv3 model information."""
    print("\n📊 DINOv3 Model Size Comparison:")
    print("=" * 80)
    print(f"{'Model':<12} {'Params':<8} {'Dimensions':<11} {'Memory':<10} {'Speed':<8} {'Use Case'}")
    print("-" * 80)
    print(f"{'small':<12} {'21M':<8} {'384':<11} {'~2GB':<10} {'Fast':<8} {'Quick experiments, prototyping'}")
    print(f"{'base':<12} {'86M':<8} {'768':<11} {'~4GB':<10} {'Medium':<8} {'Recommended for most tasks'}")
    print(f"{'large':<12} {'304M':<8} {'1024':<11} {'~8GB':<10} {'Slow':<8} {'High accuracy requirements'}")
    print(f"{'giant':<12} {'1.1B':<8} {'1536':<11} {'~16GB':<10} {'Slowest':<8} {'Research, maximum quality'}")
    print(f"{'sat_large':<12} {'304M':<8} {'1024':<11} {'~8GB':<10} {'Slow':<8} {'Satellite/aerial imagery'}")
    print(f"{'sat_giant':<12} {'1.1B':<8} {'1536':<11} {'~16GB':<10} {'Slowest':<8} {'Max satellite performance'}")
    print("=" * 80)
    print("\n💡 Recommendations:")
    print("  • Start with 'base' for most applications")
    print("  • Use 'small' for quick testing or limited GPU memory")
    print("  • Use 'large' for production where accuracy is critical")
    print("  • Use 'sat_*' variants specifically for satellite/aerial images")
    print("  • Consider 'giant' only if you have ample compute resources\n")

def main():
    """Main function with same argument parsing as original script."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 Triple Input with DINOv3 backbone (FIXED for Triple Input Datasets)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  # Quick training with small model:\n"
               "  python train_triple_dinov3_fixed.py --data dataset.yaml --dinov3-size small --variant s\n\n"
               "  # Production training with large model:\n"
               "  python train_triple_dinov3_fixed.py --data dataset.yaml --dinov3-size large --variant x --epochs 300\n\n"
               "  # Satellite imagery with specialized model:\n"
               "  python train_triple_dinov3_fixed.py --data dataset.yaml --dinov3-size sat_large --integrate p0p3\n\n"
               "  # CPU training for testing:\n"
               "  python train_triple_dinov3_fixed.py --data dataset.yaml --dinov3-size small --device cpu --batch 2\n\n"
               "For DINOv3 model details, run: python -c \"from train_triple_dinov3_fixed import print_dinov3_info; print_dinov3_info()\""
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--dinov3-size', type=str, 
                       choices=['small', 'base', 'large', 'giant', 'sat_large', 'sat_giant'], 
                       default='small',
                       help='DINOv3 model size - impacts accuracy vs speed/memory:\n'
                            'small: 21M params, fastest, lowest memory (384 dim) - good for quick experiments\n'
                            'base: 86M params, balanced performance (768 dim) - recommended for most use cases\n'
                            'large: 304M params, high accuracy (1024 dim) - best for quality results\n'
                            'giant: 1.1B params, maximum accuracy (1536 dim) - research/high-end only\n'
                            'sat_large: 304M params, satellite-optimized (1024 dim) - for aerial/satellite imagery\n'
                            'sat_giant: 1.1B params, satellite-optimized (1536 dim) - maximum satellite performance\n'
                            '(default: small)')
    parser.add_argument('--freeze-dinov3', action='store_true', default=True,
                       help='Freeze DINOv3 backbone during training (default: True)')
    parser.add_argument('--unfreeze-dinov3', action='store_true',
                       help='Unfreeze DINOv3 backbone for fine-tuning')
    parser.add_argument('--triple-branches', action='store_true',
                       help='Use separate DINOv3 branches for each input')
    parser.add_argument('--pretrained', type=str,
                       help='Path to pretrained YOLOv12 model (.pt file)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size (default: 8, reduced for DINOv3)')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Image size (default: 224, DINOv3 optimized)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--name', type=str, default='yolov12_triple_dinov3_fixed',
                       help='Experiment name (default: yolov12_triple_dinov3_fixed)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (default: 0, use "cpu" for CPU)')
    parser.add_argument('--variant', type=str, choices=['n', 's', 'm', 'l', 'x'], default='s',
                       help='YOLOv12 model variant (default: s)')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save weights every N epochs (-1 = only best/last, saves disk space)')
    parser.add_argument('--integrate', type=str, choices=['initial', 'nodino', 'p3', 'p0p3'], 
                       default='initial', 
                       help='DINOv3 integration strategy')
    
    args = parser.parse_args()
    
    # Handle freeze/unfreeze logic
    freeze_dinov3 = args.freeze_dinov3 and not args.unfreeze_dinov3
    
    # Validate input files
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    if args.pretrained and not Path(args.pretrained).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained}")
    
    # Run fixed training
    train_triple_dinov3_fixed(
        data_config=args.data,
        dinov3_size=args.dinov3_size,
        freeze_dinov3=freeze_dinov3,
        use_triple_branches=args.triple_branches,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name,
        device=args.device,
        integrate=args.integrate,
        variant=args.variant,
        save_period=args.save_period
    )

if __name__ == "__main__":
    main()