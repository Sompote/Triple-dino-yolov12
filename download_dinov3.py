#!/usr/bin/env python3
"""
DINOv3 Model Download and Setup Utility

This script downloads DINOv3 pretrained models from HuggingFace and sets up
the environment for YOLOv12 Triple Input integration.

Usage:
    python download_dinov3.py --model small
    python download_dinov3.py --model base --cache-dir ./models
    python download_dinov3.py --all --test
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import warnings

try:
    from huggingface_hub import hf_hub_download, snapshot_download, login, HfFolder
    from transformers import AutoModel, AutoImageProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace libraries not available. Install with:")
    print("pip install transformers huggingface_hub")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm not available. Install with: pip install timm")


def setup_huggingface_auth():
    """Setup HuggingFace authentication for model downloads."""
    if not HF_AVAILABLE:
        return False, "huggingface_hub not available"
    
    import os
    
    # Check for HuggingFace token
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    if not token:
        try:
            token = HfFolder.get_token()
        except:
            pass
    
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("✓ HuggingFace authentication successful")
            return True, "authenticated"
        except Exception as e:
            print(f"⚠️ HuggingFace authentication failed: {e}")
            return False, str(e)
    else:
        print("⚠️ No HuggingFace token found. Some models may not be accessible.")
        print("Set up authentication:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("  3. Or run: huggingface-cli login")
        return False, "no token"


def get_huggingface_token():
    """Get HuggingFace token from environment or saved location."""
    import os
    
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    if not token and HF_AVAILABLE:
        try:
            token = HfFolder.get_token()
        except:
            pass
    
    return token


class DINOv3Downloader:
    """Utility class for downloading and managing DINOv3 models."""
    
    MODEL_CONFIGS = {
        "small": {
            "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "timm_name": "vit_small_patch14_dinov2.lvd142m",  # Use DINOv2 as fallback
            "embed_dim": 384,
            "description": "DINOv3 Small (21M parameters)"
        },
        "base": {
            "hf_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
            "timm_name": "vit_base_patch14_dinov2.lvd142m",
            "embed_dim": 768,
            "description": "DINOv3 Base (86M parameters)"
        },
        "large": {
            "hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "timm_name": "vit_large_patch14_dinov2.lvd142m", 
            "embed_dim": 1024,
            "description": "DINOv3 Large (304M parameters)"
        },
        "giant": {
            "hf_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "timm_name": "vit_huge_patch14_dinov2.lvd142m",
            "embed_dim": 1536,
            "description": "DINOv3 Giant (1.1B parameters)"
        },
        "sat_large": {
            "hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "timm_name": "vit_large_patch14_dinov2.lvd142m",
            "embed_dim": 1024,
            "description": "DINOv3 ViT-L/16 Satellite (300M parameters, SAT-493M dataset)"
        },
        "sat_giant": {
            "hf_name": "facebook/dinov3-vit7b16-pretrain-sat493m",
            "timm_name": "vit_huge_patch14_dinov2.lvd142m",
            "embed_dim": 6144,
            "description": "DINOv3 ViT-7B/16 Satellite (6,716M parameters, SAT-493M dataset)"
        }
    }
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "yolov12_dinov3"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"DINOv3 models will be cached in: {self.cache_dir}")
    
    def list_available_models(self):
        """List all available DINOv3 models."""
        print("\nAvailable DINOv3 Models:")
        print("=" * 50)
        for model_id, config in self.MODEL_CONFIGS.items():
            print(f"{model_id:8} | {config['description']}")
            print(f"         | HF: {config['hf_name']}")
            print(f"         | Embed dim: {config['embed_dim']}")
            print()
    
    def download_from_huggingface(self, model_size: str, force_download: bool = False):
        """
        Download model from HuggingFace.
        
        Args:
            model_size: Model size (small, base, large, giant)
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded model
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace libraries required for download")
        
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = self.MODEL_CONFIGS[model_size]
        model_name = config["hf_name"]
        
        # Setup authentication
        auth_success, auth_info = setup_huggingface_auth()
        token = get_huggingface_token()
        
        print(f"\n📥 Downloading {config['description']} from HuggingFace...")
        print(f"Model: {model_name}")
        
        try:
            # Download the model with authentication
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir / "huggingface",
                force_download=force_download,
                resume_download=True,
                token=token  # Add authentication token
            )
            
            print(f"✅ Successfully downloaded to: {model_path}")
            
            # Test loading the model
            print("🧪 Testing model loading...")
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                token=token  # Add authentication token
            )
            processor = AutoImageProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True,
                token=token  # Add authentication token
            )
            
            print(f"✅ Model loaded successfully!")
            print(f"   Config: {model.config}")
            print(f"   Hidden size: {model.config.hidden_size}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to download from HuggingFace: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("❌ Authentication error detected. Please check your HuggingFace token:")
                print("  1. Get token from: https://huggingface.co/settings/tokens")
                print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
                print("  3. Or run: huggingface-cli login")
            return None
    
    def download_from_timm(self, model_size: str):
        """
        Download model using timm (fallback method).
        
        Args:
            model_size: Model size (small, base, large, giant)
            
        Returns:
            Success status
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for timm download")
        
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = self.MODEL_CONFIGS[model_size]
        timm_name = config["timm_name"]
        
        print(f"\n📥 Downloading {config['description']} from timm...")
        print(f"Model: {timm_name}")
        
        try:
            # Create model (this will download weights if not cached)
            model = timm.create_model(
                timm_name,
                pretrained=True,
                num_classes=0,
                global_pool=""
            )
            
            print(f"✅ Successfully loaded from timm!")
            print(f"   Embed dim: {model.embed_dim}")
            print(f"   Num features: {model.num_features}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download from timm: {e}")
            return False
    
    def download_model(self, model_size: str, method: str = "auto", force_download: bool = False):
        """
        Download DINOv3 model using specified method.
        
        Args:
            model_size: Model size to download
            method: Download method (auto, huggingface, timm)
            force_download: Force re-download
            
        Returns:
            Success status and model path
        """
        if method == "auto":
            # Try HuggingFace first, fallback to timm
            if HF_AVAILABLE:
                path = self.download_from_huggingface(model_size, force_download)
                if path:
                    return True, path
            
            if TIMM_AVAILABLE:
                success = self.download_from_timm(model_size)
                return success, None
            
            raise RuntimeError("Neither HuggingFace nor timm available for download")
        
        elif method == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("HuggingFace libraries not available")
            path = self.download_from_huggingface(model_size, force_download)
            return path is not None, path
        
        elif method == "timm":
            if not TIMM_AVAILABLE:
                raise ImportError("timm not available")
            success = self.download_from_timm(model_size)
            return success, None
        
        else:
            raise ValueError(f"Unknown download method: {method}")
    
    def test_integration(self, model_size: str = "small"):
        """
        Test DINOv3 integration with YOLOv12 Triple Input.
        
        Args:
            model_size: Model size to test
        """
        print(f"\n🧪 Testing DINOv3 integration ({model_size})...")
        
        try:
            # Import our DINOv3 module
            from ultralytics.nn.modules.dinov3 import create_dinov3_backbone
            
            # Test standard RGB input
            print("Testing RGB input (3 channels)...")
            backbone_rgb = create_dinov3_backbone(
                model_size=model_size,
                input_channels=3,
                output_channels=64,
                freeze=True,
                image_size=224
            )
            
            # Test forward pass
            test_input_rgb = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output_rgb = backbone_rgb(test_input_rgb)
            
            print(f"✅ RGB test passed - output shape: {output_rgb.shape}")
            
            # Test triple input
            print("Testing triple input (9 channels)...")
            backbone_triple = create_dinov3_backbone(
                model_size=model_size,
                input_channels=9,
                output_channels=64,
                freeze=True,
                image_size=224
            )
            
            test_input_triple = torch.randn(1, 9, 224, 224)
            with torch.no_grad():
                output_triple = backbone_triple(test_input_triple)
            
            print(f"✅ Triple input test passed - output shape: {output_triple.shape}")
            
            # Test with separate branches
            print("Testing triple input with separate branches...")
            backbone_separate = create_dinov3_backbone(
                model_size=model_size,
                input_channels=9,
                output_channels=64,
                freeze=True,
                use_triple_branches=True,
                image_size=224
            )
            
            with torch.no_grad():
                output_separate = backbone_separate(test_input_triple)
            
            print(f"✅ Separate branches test passed - output shape: {output_separate.shape}")
            
            print("🎉 All integration tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_requirements(self):
        """Setup required packages for DINOv3 integration."""
        print("🔧 Setting up requirements for DINOv3...")
        
        requirements = [
            "transformers>=4.21.0",
            "huggingface_hub>=0.10.0", 
            "timm>=0.6.0",
            "torch>=1.12.0",
            "torchvision>=0.13.0"
        ]
        
        print("Required packages:")
        for req in requirements:
            print(f"  - {req}")
        
        print("\nTo install all requirements:")
        print("pip install " + " ".join([req.split(">=")[0] for req in requirements]))
        
        # Check which packages are available
        print("\nCurrent availability:")
        print(f"  - transformers: {'✅' if HF_AVAILABLE else '❌'}")
        print(f"  - timm: {'✅' if TIMM_AVAILABLE else '❌'}")
        print(f"  - torch: {'✅' if 'torch' in sys.modules else '❌'}")


def main():
    parser = argparse.ArgumentParser(description='Download and setup DINOv3 models for YOLOv12 Triple Input')
    parser.add_argument('--model', type=str, choices=['small', 'base', 'large', 'giant', 'sat_large', 'sat_giant'],
                       help='DINOv3 model size to download')
    parser.add_argument('--all', action='store_true',
                       help='Download all available models')
    parser.add_argument('--method', type=str, choices=['auto', 'huggingface', 'timm'], default='auto',
                       help='Download method (default: auto)')
    parser.add_argument('--cache-dir', type=str,
                       help='Directory to cache downloaded models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if cached')
    parser.add_argument('--test', action='store_true',
                       help='Test integration after download')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--setup', action='store_true',
                       help='Show setup requirements')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DINOv3Downloader(cache_dir=args.cache_dir)
    
    if args.list:
        downloader.list_available_models()
        return
    
    if args.setup:
        downloader.setup_requirements()
        return
    
    if args.all:
        # Download all models
        models_to_download = ['small', 'base', 'large']  # Skip giant by default (very large)
        print(f"Downloading all models: {models_to_download}")
        
        for model_size in models_to_download:
            try:
                success, path = downloader.download_model(model_size, args.method, args.force)
                if success:
                    print(f"✅ {model_size} model downloaded successfully")
                else:
                    print(f"❌ {model_size} model download failed")
            except Exception as e:
                print(f"❌ Error downloading {model_size}: {e}")
    
    elif args.model:
        # Download specific model
        try:
            success, path = downloader.download_model(args.model, args.method, args.force)
            if success:
                print(f"✅ {args.model} model downloaded successfully")
                if path:
                    print(f"   Path: {path}")
            else:
                print(f"❌ {args.model} model download failed")
        except Exception as e:
            print(f"❌ Error downloading {args.model}: {e}")
            return 1
    
    else:
        # Show help and available models
        downloader.list_available_models()
        print("Use --model <size> to download a specific model")
        print("Use --all to download all models")
        print("Use --setup to see setup requirements")
        return
    
    # Run integration test if requested
    if args.test:
        test_model = args.model if args.model else 'small'
        success = downloader.test_integration(test_model)
        if not success:
            return 1
    
    print("\n🎉 DINOv3 setup complete!")
    print("You can now use DINOv3 backbones in your YOLOv12 Triple Input models.")
    
    return 0


if __name__ == "__main__":
    exit(main())