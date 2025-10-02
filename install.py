#!/usr/bin/env python3
"""
Installation script for YOLOv12 Triple Input with DINOv3 Integration

This script helps install the correct dependencies based on your system
and requirements.
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def detect_cuda():
    """Detect if CUDA is available."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
    except:
        pass
    print("ℹ️ No NVIDIA GPU detected, will install CPU version")
    return False

def get_torch_install_command(cuda_available):
    """Get the appropriate PyTorch installation command."""
    if cuda_available:
        # Install CUDA version
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Install CPU version
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

def main():
    """Main installation function."""
    print("YOLOv12 Triple Input with DINOv3 Integration - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect system
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Detect CUDA
    cuda_available = detect_cuda()
    
    # Ask user what to install
    print("\n📦 Installation Options:")
    print("1. Minimal installation (essential dependencies only)")
    print("2. Full installation (all features)")
    print("3. Development installation (includes dev tools)")
    print("4. Custom installation")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        requirements_file = "requirements-minimal.txt"
    elif choice == "2":
        requirements_file = "requirements.txt"
    elif choice == "3":
        requirements_file = "requirements-dev.txt"
    elif choice == "4":
        requirements_file = input("Enter requirements file path: ").strip()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    # Check if requirements file exists
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        sys.exit(1)
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("⚠️ Failed to upgrade pip, continuing anyway...")
    
    # Install PyTorch with appropriate backend
    torch_cmd = get_torch_install_command(cuda_available)
    if not run_command(torch_cmd, "Installing PyTorch"):
        print("❌ PyTorch installation failed")
        sys.exit(1)
    
    # Install other requirements
    if not run_command(f"pip install -r {requirements_file}", f"Installing requirements from {requirements_file}"):
        print("❌ Requirements installation failed")
        sys.exit(1)
    
    # Optional: Install flash attention if CUDA is available
    if cuda_available:
        install_flash = input("\n⚡ Install Flash Attention for better performance? (y/N): ").strip().lower()
        if install_flash == 'y':
            run_command("pip install flash-attn", "Installing Flash Attention")
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    verify_commands = [
        ("python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", "PyTorch"),
        ("python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'", "Transformers"),
        ("python -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\")'", "OpenCV"),
        ("python -c 'import timm; print(f\"TIMM: {timm.__version__}\")'", "TIMM"),
    ]
    
    for cmd, name in verify_commands:
        run_command(cmd, f"Verifying {name}")
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Prepare your triple input dataset (primary/, detail1/, detail2/ folders)")
    print("2. Update your dataset.yaml configuration")
    print("3. Run training with: python train_triple_dinov3_fixed.py --data your_dataset.yaml")
    print("\nFor help, see the documentation or run with --help")

if __name__ == "__main__":
    main()