#!/usr/bin/env python3
"""Debug script for triple input implementation."""

import torch
from ultralytics.nn.modules.conv import TripleInputConv

def debug_triple_conv():
    """Debug the TripleInputConv module in detail."""
    print("Debugging TripleInputConv...")
    
    try:
        # Test with explicit parameters
        triple_conv = TripleInputConv(c1=9, c2=64, k=3, s=2, p=None, g=1, d=1, act=True)
        print("✓ TripleInputConv created successfully")
        
        # Test forward pass
        x = torch.randn(1, 9, 640, 640)
        output = triple_conv(x)
        print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in TripleInputConv: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_individual_convs():
    """Debug individual convolutions to understand the error."""
    print("\nDebugging individual Conv layers...")
    
    try:
        from ultralytics.nn.modules.conv import Conv
        
        # Test normal 3-channel conv
        conv1 = Conv(3, 64, 3, 2)
        x1 = torch.randn(1, 3, 640, 640)
        out1 = conv1(x1)
        print(f"✓ Normal Conv: {x1.shape} -> {out1.shape}")
        
        # Test fusion conv
        conv_fusion = Conv(64*3, 64, 1, 1, 0, 1, 1, True)
        x_fusion = torch.randn(1, 192, 320, 320)
        out_fusion = conv_fusion(x_fusion)
        print(f"✓ Fusion Conv: {x_fusion.shape} -> {out_fusion.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in individual convs: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_model_creation():
    """Debug the model creation process."""
    print("\nDebugging model creation...")
    
    try:
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import yaml_load
        
        # Manually create a simple config
        simple_config = {
            'nc': 80,
            'backbone': [
                [-1, 1, 'TripleInputConv', [9, 64, 3, 2]],
                [-1, 1, 'Conv', [128, 3, 2]],
            ],
            'head': [
                [-1, 1, 'Conv', [256, 3, 1]],
                [-1, 1, 'Detect', [80]],
            ]
        }
        
        print("Simple config created")
        
        # Try to create model step by step
        model = DetectionModel(simple_config, ch=9, nc=80)
        print("✓ Model created successfully")
        
        # Test forward pass
        x = torch.randn(1, 9, 640, 640)
        output = model(x)
        print(f"✓ Model forward pass: {x.shape} -> output shapes: {[o.shape for o in output]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Triple Input Debug ===")
    
    tests = [
        debug_triple_conv,
        debug_individual_convs,
        debug_model_creation,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        print()