#!/usr/bin/env python3
"""
Test HuggingFace authentication for DINOv3 models.
"""

import os
import sys

def test_hf_auth():
    """Test HuggingFace authentication setup."""
    print("Testing HuggingFace Authentication for DINOv3")
    print("=" * 50)
    
    # Test 1: Check environment variables
    print("\n1. Checking environment variables...")
    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
    hf_token_alt = os.environ.get('HF_TOKEN')
    
    if hf_token:
        print(f"   ✓ HUGGINGFACE_HUB_TOKEN found (length: {len(hf_token)})")
    elif hf_token_alt:
        print(f"   ✓ HF_TOKEN found (length: {len(hf_token_alt)})")
    else:
        print("   ❌ No HuggingFace token found in environment variables")
        print("   Set with: export HUGGINGFACE_HUB_TOKEN='your_token'")
    
    # Test 2: Check saved token
    print("\n2. Checking saved token...")
    try:
        from huggingface_hub import HfFolder
        saved_token = HfFolder.get_token()
        if saved_token:
            print(f"   ✓ Saved token found (length: {len(saved_token)})")
        else:
            print("   ❌ No saved token found")
            print("   Run: huggingface-cli login")
    except ImportError:
        print("   ❌ huggingface_hub not available")
        print("   Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"   ❌ Error checking saved token: {e}")
    
    # Test 3: Test authentication function
    print("\n3. Testing authentication function...")
    try:
        from ultralytics.nn.modules.dinov3 import setup_huggingface_auth, get_huggingface_token
        
        # Test auth setup
        auth_success, auth_source = setup_huggingface_auth()
        if auth_success:
            print(f"   ✓ Authentication successful (source: {auth_source})")
        else:
            print(f"   ❌ Authentication failed (reason: {auth_source})")
        
        # Test token retrieval
        token = get_huggingface_token()
        if token:
            print(f"   ✓ Token retrieved (length: {len(token)})")
        else:
            print("   ❌ No token retrieved")
            
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Error testing auth function: {e}")
    
    # Test 4: Test model access (optional)
    print("\n4. Testing DINOv3 model access...")
    try:
        from transformers import AutoModel
        
        # Try to access model info (without downloading)
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        print(f"   Testing access to: {model_name}")
        
        # This will test if we can access the model repository
        token = get_huggingface_token()
        config = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=token
        ).config
        
        print(f"   ✓ Model accessible! Hidden size: {config.hidden_size}")
        
    except Exception as e:
        print(f"   ❌ Model access failed: {e}")
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            print("   This appears to be an authentication issue.")
    
    print("\n" + "=" * 50)
    print("Authentication Test Complete")
    print("\nIf tests failed, please:")
    print("1. Get token from: https://huggingface.co/settings/tokens")
    print("2. Set token: export HUGGINGFACE_HUB_TOKEN='your_token'")
    print("3. Or run: huggingface-cli login")

if __name__ == "__main__":
    test_hf_auth()