#!/usr/bin/env python3
"""
Test script for the --integrate argument with dummy data.
Tests all three integration options: initial, nodino, p3
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

def create_dummy_dataset(dataset_path="test_dataset"):
    """Create a minimal dummy dataset for testing"""
    dataset_path = Path(dataset_path)
    
    print(f"Creating dummy dataset at: {dataset_path}")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (dataset_path / subdir / split).mkdir(parents=True, exist_ok=True)
            
        # Create detail directories for triple input
        (dataset_path / 'images' / split / 'detail1').mkdir(parents=True, exist_ok=True)
        (dataset_path / 'images' / split / 'detail2').mkdir(parents=True, exist_ok=True)
    
    # Create 3 dummy images and labels for each split
    for split in ['train', 'val', 'test']:
        for i in range(3):
            img_name = f"image_{i}.jpg"
            label_name = f"image_{i}.txt"
            
            # Create primary image (640x640 RGB)
            img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            img.save(dataset_path / 'images' / split / img_name)
            
            # Create detail images (same size)
            img.save(dataset_path / 'images' / split / 'detail1' / img_name)
            img.save(dataset_path / 'images' / split / 'detail2' / img_name)
            
            # Create dummy label (one object in center)
            with open(dataset_path / 'labels' / split / label_name, 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")  # class_id x_center y_center width height
    
    # Create dataset YAML config
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'triple_input': True,
        'nc': 1,  # number of classes
        'names': ['dummy_object']
    }
    
    config_path = dataset_path / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created dummy dataset with {3*3} images")
    print(f"✓ Dataset config: {config_path}")
    return config_path

def test_integration_option(integrate_option, config_path, epochs=2):
    """Test a specific integration option"""
    print(f"\n{'='*60}")
    print(f"Testing --integrate {integrate_option}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "train_triple_dinov3.py",
        "--data", str(config_path),
        "--integrate", integrate_option,
        "--epochs", str(epochs),
        "--batch", "2",  # Small batch for testing
        "--patience", "10",
        "--name", f"test_{integrate_option}",
        "--dinov3-size", "small",  # Smallest model for testing
        "--device", "cpu"  # Use CPU for testing
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Success: --integrate {integrate_option}")
            return True
        else:
            print(f"❌ Failed: --integrate {integrate_option} (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: --integrate {integrate_option}")
        return False
    except Exception as e:
        print(f"❌ Error testing --integrate {integrate_option}: {e}")
        return False

def test_import_basic():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        from train_triple_dinov3 import train_triple_dinov3
        print("✓ train_triple_dinov3 import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("Testing YOLOv12 Triple Input with --integrate argument")
    print("="*70)
    
    # Test 1: Basic imports
    if not test_import_basic():
        print("❌ Basic import test failed. Exiting.")
        return
    
    # Test 2: Create dummy dataset
    try:
        config_path = create_dummy_dataset()
    except Exception as e:
        print(f"❌ Failed to create dummy dataset: {e}")
        return
    
    # Test 3: Test each integration option
    test_results = {}
    
    # Test nodino first (should be fastest)
    test_results['nodino'] = test_integration_option('nodino', config_path, epochs=1)
    
    # Test initial (current default)
    test_results['initial'] = test_integration_option('initial', config_path, epochs=1)
    
    # Test p3 (may fail if config doesn't exist)
    test_results['p3'] = test_integration_option('p3', config_path, epochs=1)
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for option, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"--integrate {option:8}: {status}")
    
    passed = sum(test_results.values())
    total = len(test_results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration options work correctly!")
    elif passed > 0:
        print("⚠️ Some integration options work, check failures above")
    else:
        print("❌ All tests failed, check your setup")
    
    # Cleanup
    print(f"\nCleaning up test data...")
    try:
        shutil.rmtree("test_dataset", ignore_errors=True)
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    main()