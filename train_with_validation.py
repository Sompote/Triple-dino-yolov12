#!/usr/bin/env python3
"""
Training Script for YOLOv12 Triple Input with Proper Train/Val/Test Split

This script demonstrates proper training with validation during training.
The validation set is used during training for monitoring and early stopping.
The test set should only be used after training for final evaluation.

Usage:
    python train_with_validation.py --model ultralytics/cfg/models/v12/yolov12_triple.yaml --data test_triple_dataset.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(model_config, data_config, epochs=100, batch_size=16, imgsz=640, 
                patience=50, save_period=10, name="yolov12_triple_training"):
    """
    Train YOLOv12 model with triple input support.
    
    Args:
        model_config (str): Path to model configuration file
        data_config (str): Path to dataset configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        imgsz (int): Image size for training
        patience (int): Early stopping patience (epochs)
        save_period (int): Save checkpoint every N epochs
        name (str): Experiment name
    """
    
    # Load model configuration
    model = YOLO(model_config)
    
    print("Starting YOLOv12 Triple Input Training")
    print("-" * 50)
    print(f"Model Config: {model_config}")
    print(f"Data Config: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Patience: {patience}")
    print("-" * 50)
    
    # Train the model
    # The validation set specified in data_config will be used automatically during training
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,  # Early stopping patience
        save=True,
        save_period=save_period,  # Save checkpoint every N epochs
        name=name,
        verbose=True,
        val=True,  # Enable validation during training
        plots=True,  # Generate training plots
        cache=False,  # Don't cache images (for triple input compatibility)
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
    print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
    
    # Validate on validation set (already done during training, but for demonstration)
    print("\nValidating on validation set...")
    val_results = model.val(data=data_config, split='val')
    print(f"Validation mAP50: {val_results.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")
    
    print("\nTo evaluate on test set after training, run:")
    print(f"python test_model_evaluation.py --model runs/detect/{name}/weights/best.pt --data {data_config} --split test")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12 Triple Input model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model configuration file (e.g., ultralytics/cfg/models/v12/yolov12_triple.yaml)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration file (e.g., test_triple_dataset.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience in epochs (default: 50)')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--name', type=str, default='yolov12_triple_training',
                       help='Experiment name (default: yolov12_triple_training)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model config file not found: {args.model}")
    
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config file not found: {args.data}")
    
    # Train the model
    results = train_model(
        model_config=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        save_period=args.save_period,
        name=args.name
    )

if __name__ == "__main__":
    main()