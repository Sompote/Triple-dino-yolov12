# YOLOv12 Triple Input with DINOv3 Integration - Setup Guide

## Quick Installation

### Option 1: Automatic Installation (Recommended)
```bash
python install.py
```

### Option 2: Manual Installation

#### Minimal Installation (Essential only)
```bash
pip install -r requirements-minimal.txt
```

#### Full Installation (All features)
```bash
pip install -r requirements.txt
```

#### Development Installation (With dev tools)
```bash
pip install -r requirements-dev.txt
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.10 or 3.11
- **RAM**: 32GB or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better)
- **CUDA**: 11.8 or 12.1
- **Storage**: 50GB free space (SSD recommended)

## GPU Setup

### CUDA Installation
For NVIDIA GPUs, install CUDA toolkit:

**Linux/Windows:**
```bash
# Install CUDA 12.1 (recommended)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

### PyTorch with CUDA
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Dataset Preparation

### Triple Input Dataset Structure
Your dataset should have the following structure:
```
dataset/
├── images/
│   ├── primary/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── detail1/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── detail2/
│       ├── train/
│       ├── val/
│       └── test/
├── labels/
│   └── primary/
│       ├── train/
│       ├── val/
│       └── test/
└── dataset.yaml
```

### Dataset Configuration (dataset.yaml)
```yaml
path: /path/to/your/dataset
train: /path/to/your/dataset/images/primary/train
val: /path/to/your/dataset/images/primary/val
test: /path/to/your/dataset/images/primary/test

nc: 1  # Number of classes
names: ['your_class_name']
```

## Quick Start

### 1. Basic Training
```bash
python train_triple_dinov3_fixed.py \
    --data your_dataset.yaml \
    --integrate p3 \
    --variant s \
    --epochs 100 \
    --batch 8
```

### 2. Advanced Training
```bash
python train_triple_dinov3_fixed.py \
    --data your_dataset.yaml \
    --integrate p0p3 \
    --dinov3-size large \
    --variant x \
    --epochs 300 \
    --batch 16 \
    --patience 50 \
    --device 0
```

### 3. CPU Training (for testing)
```bash
python train_triple_dinov3_fixed.py \
    --data your_dataset.yaml \
    --integrate p3 \
    --variant n \
    --epochs 10 \
    --batch 2 \
    --device cpu
```

## Integration Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `initial` | DINOv3 before backbone | General purpose |
| `nodino` | No DINOv3 (triple input only) | Baseline comparison |
| `p3` | DINOv3 after P3 stage | Feature enhancement |
| `p0p3` | Dual DINOv3 (P0 + P3) | Maximum performance |

## Model Variants

| Variant | Width Scale | Typical Use |
|---------|-------------|-------------|
| `n` | 0.25x | Fast inference, mobile |
| `s` | 0.5x | Balanced speed/accuracy |
| `m` | 1.0x | Good accuracy |
| `l` | 1.0x | High accuracy |
| `x` | 1.5x | Maximum accuracy |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch 4

# Use smaller variant
--variant n

# Use smaller image size
--imgsz 192
```

**2. Import Errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version
```

**3. Dataset Not Found**
- Verify dataset structure matches the expected format
- Check paths in dataset.yaml
- Ensure all three image folders (primary, detail1, detail2) exist

**4. DINOv3 Download Issues**
```bash
# Set HuggingFace token
export HUGGINGFACE_HUB_TOKEN="your_token"

# Or login via CLI
huggingface-cli login
```

### Performance Optimization

**1. Memory Optimization**
- Use gradient checkpointing: Add `--amp` flag
- Reduce batch size
- Use mixed precision training

**2. Speed Optimization**
- Use GPU with high memory
- Enable Flash Attention (if available)
- Use multiple GPUs: `--device 0,1,2,3`

**3. Storage Optimization**
- Use `--cache ram` for small datasets
- Use `--cache disk` for large datasets
- Set `--save-period -1` to save only best/last

## Environment Variables

```bash
# HuggingFace authentication
export HUGGINGFACE_HUB_TOKEN="your_token"

# CUDA settings
export CUDA_VISIBLE_DEVICES="0,1"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## Support

For issues and questions:
1. Check this setup guide
2. Review error messages carefully
3. Verify dataset structure
4. Check system requirements
5. Try minimal installation first

## License

This project follows the Ultralytics AGPL-3.0 License.