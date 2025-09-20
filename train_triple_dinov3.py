#!/usr/bin/env python3
"""
Training script for YOLOv12 Triple Input with DINOv3 backbone.

This script trains YOLOv12 Triple Input models with DINOv3 feature extraction
for enhanced performance in civil engineering applications.

Usage:
    python train_triple_dinov3.py --data test_triple_dataset.yaml
    python train_triple_dinov3.py --data test_triple_dataset.yaml --dinov3-size base --freeze-dinov3
    python train_triple_dinov3.py --data test_triple_dataset.yaml --pretrained yolov12n.pt --dinov3-size small
"""

import argparse
import torch
from pathlib import Path
import warnings
from ultralytics import YOLO

def train_triple_dinov3(
    data_config: str,
    dinov3_size: str = "small",
    freeze_dinov3: bool = True,
    use_triple_branches: bool = False,
    pretrained_path: str = None,
    epochs: int = 100,
    batch_size: int = 8,  # Smaller default due to DINOv3 memory usage
    imgsz: int = 224,     # DINOv3 default size
    patience: int = 50,
    name: str = "yolov12_triple_dinov3",
    device: str = "0",
    **kwargs
):
    """
    Train YOLOv12 Triple Input with DINOv3 backbone.
    
    Args:
        data_config: Path to dataset configuration
        dinov3_size: DINOv3 model size (small, base, large, giant)
        freeze_dinov3: Whether to freeze DINOv3 backbone
        use_triple_branches: Whether to use separate DINOv3 branches
        pretrained_path: Path to pretrained YOLOv12 model (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        patience: Early stopping patience
        name: Experiment name
        device: Device to use
        **kwargs: Additional training arguments
        
    Returns:
        Training results
    """
    
    print("YOLOv12 Triple Input with DINOv3 Training")
    print("=" * 60)
    print(f"Data Config: {data_config}")
    print(f"DINOv3 Size: {dinov3_size}")
    print(f"Freeze DINOv3: {freeze_dinov3}")
    print(f"Triple Branches: {use_triple_branches}")
    print(f"Pretrained: {pretrained_path or 'None'}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Step 1: Setup DINOv3 requirements
    print("\n🔧 Step 1: Setting up DINOv3 requirements...")
    try:
        import transformers
        import timm
        print("✓ Required packages available")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Install with: pip install transformers timm huggingface_hub")
        return None
    
    # Step 2: Download DINOv3 model if needed
    print(f"\n📥 Step 2: Preparing DINOv3 {dinov3_size} model...")
    try:
        from download_dinov3 import DINOv3Downloader
        downloader = DINOv3Downloader()
        success, _ = downloader.download_model(dinov3_size, method="auto")
        if success:
            print(f"✓ DINOv3 {dinov3_size} model ready")
        else:
            print(f"⚠️ Failed to download DINOv3 {dinov3_size}, proceeding anyway...")
    except Exception as e:
        print(f"⚠️ DINOv3 download warning: {e}")
        print("Proceeding with training, model will be downloaded automatically if needed")
    
    # Step 3: Create model configuration
    print(f"\n🏗️ Step 3: Creating model configuration...")
    
    # Use DINOv3 model configuration
    model_config = "ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml"
    
    if not Path(model_config).exists():
        print(f"❌ Model config not found: {model_config}")
        return None
    
    # Step 4: Initialize model
    print(f"\n🚀 Step 4: Initializing model...")
    try:
        if pretrained_path:
            print(f"Loading with pretrained weights: {pretrained_path}")
            # For DINOv3 integration, we'll need custom weight loading
            from load_pretrained_triple import load_pretrained_weights_to_triple_model
            model = load_pretrained_weights_to_triple_model(
                pretrained_path=pretrained_path,
                triple_model_config=model_config
            )
        else:
            print("Training from scratch with DINOv3 features")
            model = YOLO(model_config)
        
        print("✓ Model initialized successfully")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("This might be due to DINOv3 integration complexity")
        return None
    
    # Step 5: Configure training parameters
    print(f"\n⚙️ Step 5: Configuring training parameters...")
    
    # Training configuration optimized for DINOv3
    train_args = {
        'data': data_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,
        'save': True,
        'save_period': 10,
        'name': name,
        'verbose': True,
        'val': True,
        'plots': True,
        'cache': False,  # Disable caching for triple input
        'device': device,
        
        # Learning rate configuration for DINOv3
        'lr0': 0.001 if freeze_dinov3 else 0.0001,  # Lower LR if fine-tuning DINOv3
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,  # Longer warmup for DINOv3
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Optimizer (AdamW often works better with transformers)
        'optimizer': 'AdamW',
        
        # Mixed precision (helpful for memory with DINOv3)
        'amp': True,
        
        # Additional arguments
        **kwargs
    }
    
    # Display key training parameters
    print(f"  Learning rate: {train_args['lr0']}")
    print(f"  Optimizer: {train_args['optimizer']}")
    print(f"  Mixed precision: {train_args['amp']}")
    print(f"  Warmup epochs: {train_args['warmup_epochs']}")
    
    # Step 6: Memory and performance warnings
    print(f"\n⚠️ Step 6: Performance considerations...")
    print("DINOv3 backbone requires significant memory and compute:")
    print(f"  - Recommended batch size: 4-8 (current: {batch_size})")
    print(f"  - Recommended image size: 224-384 (current: {imgsz})")
    print(f"  - DINOv3 frozen: {freeze_dinov3} (recommended for initial training)")
    
    if batch_size > 8:
        print("⚠️ Large batch size may cause OOM with DINOv3")
    
    if imgsz > 384:
        print("⚠️ Large image size may cause OOM with DINOv3")
    
    # Step 7: Start training
    print(f"\n🎯 Step 7: Starting training...")
    print("Note: First epoch may be slow due to DINOv3 model download/loading")
    
    try:
        results = model.train(**train_args)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
        print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
        
        # Step 8: Post-training validation
        print(f"\n📊 Step 8: Post-training validation...")
        val_results = model.val(data=data_config, split='val')
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Validation mAP50-95: {val_results.box.map:.4f}")
        
        # Step 9: DINOv3 feature analysis (optional)
        print(f"\n🔍 Step 9: DINOv3 integration analysis...")
        try:
            # Check if DINOv3 features are being used
            dinov3_layers = []
            for name, module in model.model.named_modules():
                if 'dinov3' in name.lower() or 'DINOv3' in str(type(module)):
                    dinov3_layers.append(name)
            
            if dinov3_layers:
                print(f"✓ DINOv3 layers found: {len(dinov3_layers)}")
                print(f"  Frozen parameters: {freeze_dinov3}")
            else:
                print("⚠️ No DINOv3 layers detected in model")
                
        except Exception as e:
            print(f"Could not analyze DINOv3 integration: {e}")
        
        # Instructions for further use
        print(f"\n🎯 Next steps:")
        print(f"1. Evaluate on test set:")
        print(f"   python test_model_evaluation.py --model runs/detect/{name}/weights/best.pt --data {data_config} --split test")
        
        if freeze_dinov3:
            print(f"2. Consider unfreezing DINOv3 for fine-tuning:")
            print(f"   # Load model and unfreeze DINOv3, then continue training with lower LR")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Common causes:")
        print("  - Out of memory (reduce batch size or image size)")
        print("  - DINOv3 model download issues (check internet connection)")
        print("  - Incompatible model configuration")
        import traceback
        traceback.print_exc()
        return None

def compare_with_without_dinov3(data_config: str, epochs: int = 50):
    """
    Compare training with and without DINOv3 backbone.
    
    Args:
        data_config: Path to dataset config
        epochs: Number of epochs for comparison
    """
    print("Comparing YOLOv12 Triple Input with and without DINOv3")
    print("=" * 70)
    
    # Train without DINOv3
    print("\n🔄 Training without DINOv3 (baseline)...")
    try:
        from train_with_validation import train_model
        baseline_results = train_model(
            model_config="ultralytics/cfg/models/v12/yolov12_triple.yaml",
            data_config=data_config,
            epochs=epochs,
            batch_size=8,
            name="triple_baseline"
        )
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        baseline_results = None
    
    # Train with DINOv3
    print(f"\n🔄 Training with DINOv3...")
    dinov3_results = train_triple_dinov3(
        data_config=data_config,
        dinov3_size="small",
        freeze_dinov3=True,
        epochs=epochs,
        batch_size=8,
        name="triple_dinov3"
    )
    
    # Compare results
    print("\n📈 Comparison Results:")
    print("-" * 40)
    
    if baseline_results and dinov3_results:
        try:
            baseline_map = baseline_results.metrics.get('metrics/mAP50(B)', 0)
            dinov3_map = dinov3_results.metrics.get('metrics/mAP50(B)', 0)
            
            print(f"Baseline mAP50:      {baseline_map:.4f}")
            print(f"DINOv3 mAP50:        {dinov3_map:.4f}")
            print(f"Improvement:         {dinov3_map - baseline_map:.4f}")
            
            if dinov3_map > baseline_map:
                print("✅ DINOv3 backbone improved performance!")
            else:
                print("⚠️ DINOv3 backbone didn't improve performance")
                print("   Consider: longer training, unfreezing DINOv3, or different model size")
        except Exception as e:
            print(f"⚠️ Could not compare metrics: {e}")
    else:
        print("⚠️ Could not complete comparison (one training failed)")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12 Triple Input with DINOv3 backbone')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration (.yaml file)')
    parser.add_argument('--dinov3-size', type=str, choices=['small', 'base', 'large', 'giant'], 
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
    parser.add_argument('--name', type=str, default='yolov12_triple_dinov3',
                       help='Experiment name (default: yolov12_triple_dinov3)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (default: 0)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with and without DINOv3 backbone')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download DINOv3 models without training')
    
    args = parser.parse_args()
    
    # Handle freeze/unfreeze logic
    freeze_dinov3 = args.freeze_dinov3 and not args.unfreeze_dinov3
    
    # Validate input files
    if not args.download_only and not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    if args.pretrained and not Path(args.pretrained).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained}")
    
    # Download only mode
    if args.download_only:
        print("Downloading DINOv3 models...")
        from download_dinov3 import DINOv3Downloader
        downloader = DINOv3Downloader()
        success, _ = downloader.download_model(args.dinov3_size, method="auto")
        if success:
            print(f"✅ Downloaded DINOv3 {args.dinov3_size}")
            # Test integration
            downloader.test_integration(args.dinov3_size)
        return
    
    # Run comparison or regular training
    if args.compare:
        compare_with_without_dinov3(args.data, epochs=min(args.epochs, 50))
    else:
        train_triple_dinov3(
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
            device=args.device
        )

if __name__ == "__main__":
    main()