#!/usr/bin/env python3
"""
Test script for triple input augmentation to verify the copyMakeBorder fix.
"""

import numpy as np
import cv2
from ultralytics.data.augment import LetterBox

def test_triple_input_augmentation():
    """Test LetterBox augmentation with 9-channel triple input."""
    print("Testing triple input augmentation...")
    
    # Create a test 9-channel image (triple input)
    h, w = 480, 640
    channels = 9
    test_img = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)
    
    print(f"Test image shape: {test_img.shape}")
    
    # Create LetterBox transform
    letterbox = LetterBox(new_shape=(256, 256), auto=False, scaleFill=False, scaleup=True, stride=32)
    
    # Test with proper labels structure
    from ultralytics.utils.instance import Instances
    
    # Create empty instances
    instances = Instances(bboxes=np.empty((0, 4)), segments=None, keypoints=None, bbox_format='xyxy', normalized=False)
    
    labels = {
        'img': test_img,
        'instances': instances,
        'cls': np.array([]),
        'bboxes': np.empty((0, 4)),
        'segments': [],
        'keypoints': None,
        'im_file': 'test.jpg',
        'ori_shape': test_img.shape[:2],
        'resized_shape': test_img.shape[:2],
        'ratio_pad': (1.0, (0, 0))
    }
    
    try:
        # Apply the transform
        result = letterbox(labels)
        result_img = result['img']
        
        print(f"✅ Success! Output image shape: {result_img.shape}")
        print(f"Expected shape: (256, 256, 9)")
        
        # Verify the output shape
        assert result_img.shape[2] == 9, f"Expected 9 channels, got {result_img.shape[2]}"
        assert result_img.shape[0] == 256, f"Expected height 256, got {result_img.shape[0]}"
        assert result_img.shape[1] == 256, f"Expected width 256, got {result_img.shape[1]}"
        
        print("✅ All assertions passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standard_input_augmentation():
    """Test LetterBox augmentation with standard 3-channel input."""
    print("\nTesting standard input augmentation...")
    
    # Create a test 3-channel image (standard input)
    h, w = 480, 640
    channels = 3
    test_img = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)
    
    print(f"Test image shape: {test_img.shape}")
    
    # Create LetterBox transform
    letterbox = LetterBox(new_shape=(256, 256), auto=False, scaleFill=False, scaleup=True, stride=32)
    
    # Test with proper labels structure
    from ultralytics.utils.instance import Instances
    
    # Create empty instances
    instances = Instances(bboxes=np.empty((0, 4)), segments=None, keypoints=None, bbox_format='xyxy', normalized=False)
    
    labels = {
        'img': test_img,
        'instances': instances,
        'cls': np.array([]),
        'bboxes': np.empty((0, 4)),
        'segments': [],
        'keypoints': None,
        'im_file': 'test.jpg',
        'ori_shape': test_img.shape[:2],
        'resized_shape': test_img.shape[:2],
        'ratio_pad': (1.0, (0, 0))
    }
    
    try:
        # Apply the transform
        result = letterbox(labels)
        result_img = result['img']
        
        print(f"✅ Success! Output image shape: {result_img.shape}")
        print(f"Expected shape: (256, 256, 3)")
        
        # Verify the output shape
        assert result_img.shape[2] == 3, f"Expected 3 channels, got {result_img.shape[2]}"
        assert result_img.shape[0] == 256, f"Expected height 256, got {result_img.shape[0]}"
        assert result_img.shape[1] == 256, f"Expected width 256, got {result_img.shape[1]}"
        
        print("✅ All assertions passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing LetterBox augmentation with triple input fix...")
    print("=" * 60)
    
    # Test standard input first
    success_standard = test_standard_input_augmentation()
    
    # Test triple input
    success_triple = test_triple_input_augmentation()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Standard input (3-channel): {'✅ PASS' if success_standard else '❌ FAIL'}")
    print(f"Triple input (9-channel): {'✅ PASS' if success_triple else '❌ FAIL'}")
    
    if success_standard and success_triple:
        print("\n🎉 All tests passed! The augmentation fix is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")