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

def main():
    """Main function with same argument parsing as original script."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 Triple Input with DINOv3 backbone (FIXED)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--dinov3-size', type=str, choices=['small', 'base', 'large', 'giant', 'sat_large', 'sat_giant'], 
                       default='small', help='DINOv3 model size (default: small)')
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