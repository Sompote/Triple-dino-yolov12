#!/usr/bin/env python3
"""
Test script for YOLOv12 Triple Input functionality.
This script validates that the triple input architecture works correctly.
"""

import torch
import numpy as np
from ultralytics.nn.modules.conv import TripleInputConv
from ultralytics.nn.tasks import parse_model
from ultralytics import YOLO

def test_triple_input_conv():
    """Test the TripleInputConv module."""
    print("Testing TripleInputConv module...")
    
    # Create a 9-channel input (3 RGB images concatenated)
    batch_size = 2
    height, width = 640, 640
    input_tensor = torch.randn(batch_size, 9, height, width)
    
    # Initialize TripleInputConv
    triple_conv = TripleInputConv(c1=9, c2=64, k=3, s=2)
    
    # Forward pass
    output = triple_conv(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    expected_height = height // 2  # stride=2
    expected_width = width // 2
    expected_shape = (batch_size, 64, expected_height, expected_width)
    
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ TripleInputConv test passed!")
    
    return True

def test_model_loading():
    """Test loading the triple input model configuration."""
    print("\nTesting model configuration loading...")
    
    try:
        # Try to parse the triple input model configuration
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import yaml_load
        
        # Load the triple input config
        cfg_path = "ultralytics/cfg/models/v12/yolov12_triple.yaml"
        cfg = yaml_load(cfg_path)
        
        print(f"✓ Successfully loaded config: {cfg_path}")
        print(f"  Scales: {cfg.get('scales', {})}")
        print(f"  Backbone layers: {len(cfg.get('backbone', []))}")
        print(f"  Head layers: {len(cfg.get('head', []))}")
        
        # Test with 'n' scale
        model = DetectionModel(cfg, ch=9, nc=80)  # 9 input channels for triple input
        print(f"✓ Successfully created DetectionModel with 9 input channels")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with 9-channel input
        test_input = torch.randn(1, 9, 640, 640)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: {[x.shape for x in output]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_dataset_config():
    """Test the triple input dataset configuration."""
    print("\nTesting dataset configuration...")
    
    try:
        from ultralytics.utils import yaml_load
        
        cfg_path = "ultralytics/cfg/datasets/coco_triple.yaml"
        cfg = yaml_load(cfg_path)
        
        print(f"✓ Successfully loaded dataset config: {cfg_path}")
        print(f"  Triple input enabled: {cfg.get('triple_input', False)}")
        print(f"  Dataset path: {cfg.get('path', 'N/A')}")
        print(f"  Number of classes: {len(cfg.get('names', {}))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset config test failed: {e}")
        return False

def create_dummy_triple_dataset():
    """Create a dummy triple input dataset structure for testing."""
    print("\nCreating dummy triple dataset structure...")
    
    import os
    from pathlib import Path
    
    # Create directory structure
    base_dir = Path("test_triple_dataset")
    dirs_to_create = [
        "images/train",
        "images/train/detail1", 
        "images/train/detail2",
        "images/val",
        "images/val/detail1",
        "images/val/detail2",
        "labels/train",
        "labels/val"
    ]
    
    for dir_path in dirs_to_create:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create dummy images (small random images)
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Save some dummy images
    import cv2
    for i in range(3):
        # Primary images
        cv2.imwrite(str(base_dir / f"images/train/test_{i}.jpg"), dummy_image)
        cv2.imwrite(str(base_dir / f"images/val/test_{i}.jpg"), dummy_image)
        
        # Detail images
        cv2.imwrite(str(base_dir / f"images/train/detail1/test_{i}.jpg"), dummy_image)
        cv2.imwrite(str(base_dir / f"images/train/detail2/test_{i}.jpg"), dummy_image)
        cv2.imwrite(str(base_dir / f"images/val/detail1/test_{i}.jpg"), dummy_image)
        cv2.imwrite(str(base_dir / f"images/val/detail2/test_{i}.jpg"), dummy_image)
        
        # Create dummy label files
        with open(base_dir / f"labels/train/test_{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")  # class x y w h (normalized)
        with open(base_dir / f"labels/val/test_{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
    
    print(f"✓ Created dummy dataset at: {base_dir.absolute()}")
    
    # Create dataset config for testing
    config_content = f"""
path: {base_dir.absolute()}
train: images/train
val: images/val
triple_input: true
nc: 1
names:
  0: test_object
"""
    
    with open("test_triple_dataset.yaml", "w") as f:
        f.write(config_content)
    
    print("✓ Created dataset configuration: test_triple_dataset.yaml")
    return True

def main():
    """Run all tests."""
    print("YOLOv12 Triple Input Validation Tests")
    print("=" * 50)
    
    tests = [
        ("TripleInputConv Module", test_triple_input_conv),
        ("Model Configuration", test_model_loading), 
        ("Dataset Configuration", test_dataset_config),
        ("Dummy Dataset Creation", create_dummy_triple_dataset),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! YOLOv12 Triple Input is ready to use.")
        print("\nUsage example:")
        print("python -c \"from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml')\"")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the implementation.")

if __name__ == "__main__":
    main()