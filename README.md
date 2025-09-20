<div align="center">
<h1>YOLOv12 Triple Input</h1>
<h3>YOLOv12: Enhanced Multi-Image Object Detection with Triple Input Architecture</h3>

**Research Group, Department of Civil Engineering**  
**King Mongkut's University of Technology Thonburi (KMUTT)**

Original YOLOv12 by [Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>, [Qixiang Ye](https://people.ucas.ac.cn/~qxye?language=en)<sup>2</sup>, [David Doermann](https://cse.buffalo.edu/~doermann/)<sup>1</sup>

<sup>1</sup>  University at Buffalo, SUNY, <sup>2</sup> University of Chinese Academy of Sciences.

**Enhanced with Triple Input Architecture for Civil Engineering Applications**

<p align="center">
  <img src="assets/tradeoff_turbo.svg" width=90%> <br>
  Comparison with popular methods in terms of latency-accuracy (left) and FLOPs-accuracy (right) trade-offs
</p>

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2502.12524-b31b1b.svg)](https://arxiv.org/abs/2502.12524) [![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunsmarterjieleaf/yolov12) <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov12-object-detection-model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/jxxn03x/yolov12-on-custom-data) [![LightlyTrain Notebook](https://img.shields.io/badge/LightlyTrain-Notebook-blue?)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/yolov12.ipynb) [![deploy](https://media.roboflow.com/deploy.svg)](https://blog.roboflow.com/use-yolov12-with-roboflow/#deploy-yolov12-models-with-roboflow) [![Openbayes](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/A4ac4xNrUCQ) 

## 🚀 What's New: Triple Input Architecture with DINOv3 for Civil Engineering

This repository extends YOLOv12 with **Triple Input Architecture** and **DINOv3 backbone integration**, developed by the Research Group at the Department of Civil Engineering, KMUTT. The enhancement enables processing of three related images simultaneously with advanced feature extraction for superior object detection in civil engineering applications. This implementation is inspired by [triple_YOLO13](https://github.com/Sompote/triple_YOLO13) and [DINOv3](https://github.com/facebookresearch/dinov3) and provides:

- **🎯 Enhanced Detection**: Process primary image + 2 detail images as 9-channel input
- **🤖 DINOv3 Integration**: State-of-the-art vision transformer features from Meta's DINOv3
- **🏗️ Civil Engineering Focus**: Optimized for infrastructure monitoring and analysis
- **🧊 Freezing Support**: Freeze DINOv3 backbone for efficient transfer learning
- **🧠 Smart Fallback**: Automatically uses primary image if detail images are missing  
- **⚡ Efficient Processing**: Specialized modules for optimal performance
- **🔄 Backward Compatible**: Seamlessly works with existing YOLOv12 infrastructure

### Triple Input + DINOv3 Features for Civil Engineering

- **Multi-Image Processing**: Combines three related images (primary + 2 detail views) for comprehensive structural analysis
- **DINOv3 Backbone**: Leverages Meta's state-of-the-art vision transformer for superior feature extraction
- **Automatic Model Download**: DINOv3 models are downloaded automatically from HuggingFace on first use
- **9-Channel Input**: Concatenated RGB channels from three images enhanced by DINOv3 features
- **Independent Branch Processing**: Each image processed separately before fusion for robust detection
- **Frozen Transfer Learning**: Use pre-trained DINOv3 features with frozen weights for efficient training
- **Feature Fusion**: Advanced feature combination for improved accuracy in infrastructure monitoring
- **Dataset Flexibility**: Supports both single and triple input formats for various civil engineering applications
- **Infrastructure Applications**: Optimized for crack detection, structural monitoring, and construction analysis
- **Model Variants**: Multiple DINOv3 sizes (small, base, large) for different computational requirements

## Updates

- **2025/09/20**: 🎉 **Triple Input Architecture with DINOv3** released by KMUTT Civil Engineering Research Group! Process multiple images simultaneously with state-of-the-art vision transformer features for enhanced detection accuracy in civil engineering applications.
- **2025/09/20**: 🤖 **DINOv3 Integration** - Complete implementation with HuggingFace model support, automatic downloading, and frozen backbone training.
- **2025/09/20**: 🏗️ Optimized for infrastructure monitoring, crack detection, and structural analysis applications.
- 2025/06/17: **Use this repo for YOLOv12 instead of [ultralytics](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/12). Their implementation is inefficient, requires more memory, and has unstable training, which are fixed here!**
- 2025/07/01: YOLOv12's **classification** models are released, see [code](https://github.com/sunsmarterjie/yolov12/tree/Cls).
- 2025/06/04: YOLOv12's **instance segmentation** models are released, see [code](https://github.com/sunsmarterjie/yolov12/tree/Seg).

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Enhancing the network architecture of the YOLO framework has been crucial for a long time but has focused on CNN-based improvements despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention-based models cannot match the speed of CNN-based models. This paper proposes an attention-centric YOLO framework, namely YOLOv12, that matches the speed of previous CNN-based ones while harnessing the performance benefits of attention mechanisms.

YOLOv12 surpasses all popular real-time object detectors in accuracy with competitive speed. For example, YOLOv12-N achieves 40.6% mAP with an inference latency of 1.64 ms on a T4 GPU, outperforming advanced YOLOv10-N / YOLOv11-N by 2.1%/1.2% mAP with a comparable speed. This advantage extends to other model scales. YOLOv12 also surpasses end-to-end real-time detectors that improve DETR, such as RT-DETR / RT-DETRv2: YOLOv12-S beats RT-DETR-R18 / RT-DETRv2-R18 while running 42% faster, using only 36% of the computation and 45% of the parameters.

**Triple Input Enhancement for Civil Engineering**: This repository, developed by the Research Group at the Department of Civil Engineering, KMUTT, extends YOLOv12 with multi-image processing capabilities specifically optimized for civil engineering applications. The enhancement allows the model to leverage contextual information from multiple related images for improved detection accuracy in infrastructure monitoring, structural analysis, and construction site management.
</details>

## Main Results

### Standard YOLOv12 (Single Input)
**Turbo (default)**:
| Model (det)                                                                              | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :------------------------------:| :-----------------:| :---------------:|
| [YOLO12n](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt) | 640                   | 40.4                 | 1.60                            | 2.5                | 6.0               |
| [YOLO12s](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt) | 640                   | 47.6                 | 2.42                            | 9.1                | 19.4              |
| [YOLO12m](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt) | 640                   | 52.5                 | 4.27                            | 19.6               | 59.8              |
| [YOLO12l](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt) | 640                   | 53.8                 | 5.83                            | 26.5               | 82.4              |
| [YOLO12x](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt) | 640                   | 55.4                 | 10.38                           | 59.3               | 184.6             |

### YOLOv12 Triple Input (Multi-Image) - KMUTT Civil Engineering
| Model (det)                    | size<br><sup>(pixels) | Input<br><sup>Channels | mAP<sup>val<br>50-95 | Speed (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) | Applications |
| :----------------------------- | :-------------------: | :--------------------: | :-------------------:| :------------------------------:| :-----------------:| :---------------:| :----------- |
| YOLO12n-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 2.6                | 6.2               | Infrastructure Monitoring |
| YOLO12s-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 9.2                | 19.6              | Crack Detection |
| YOLO12m-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 19.7               | 60.1              | Structural Analysis |
| YOLO12l-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 26.6               | 82.7              | Construction Monitoring |
| YOLO12x-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 59.4               | 185.0             | Large Infrastructure |

### YOLOv12 Triple Input + DINOv3 (Enhanced) - KMUTT Civil Engineering
| Model (det)                    | size<br><sup>(pixels) | DINOv3<br><sup>Size | params<br><sup>(M) | FLOPs<br><sup>(G) | Applications |
| :----------------------------- | :-------------------: | :-----------------: | :-----------------:| :---------------:| :----------- |
| YOLO12n-triple-dinov3-small    | 224                   | Small (21M)         | 23.6               | TBD               | Enhanced Infrastructure Monitoring |
| YOLO12s-triple-dinov3-small    | 224                   | Small (21M)         | 30.2               | TBD               | Advanced Crack Detection |
| YOLO12m-triple-dinov3-base     | 224                   | Base (86M)          | 105.7              | TBD               | Precision Structural Analysis |
| YOLO12l-triple-dinov3-large    | 224                   | Large (304M)        | 330.6              | TBD               | Research-Grade Monitoring |

*TBD: To Be Determined after training on civil engineering datasets

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Sompote/Triple-dino-yolov12.git
cd Triple-dino-yolov12

# Install flash attention (optional but recommended)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Create environment
conda create -n yolov12 python=3.11
conda activate yolov12

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Standard YOLOv12 Usage

#### Validation
```python
from ultralytics import YOLO

model = YOLO('yolov12n.pt')  # n/s/m/l/x
model.val(data='coco.yaml', save_json=True)
```

#### Training
```python
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')
results = model.train(
    data='coco.yaml',
    epochs=600, 
    batch=256, 
    imgsz=640,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    device="0,1,2,3",
)
```

#### Prediction
```python
from ultralytics import YOLO

model = YOLO('yolov12n.pt')
results = model.predict('path/to/image.jpg')
results[0].show()
```

## 🎯 Triple Input Usage

### Training with DINOv3 Backbone (Recommended)
```bash
# Train with DINOv3 + Triple Input (downloads models automatically)
python train_triple_dinov3.py \
    --data your_dataset.yaml \
    --dinov3-size small \
    --freeze-dinov3 \
    --epochs 100 \
    --batch 8

# Compare with/without DINOv3 (automatic download)
python train_triple_dinov3.py \
    --data your_dataset.yaml \
    --compare \
    --dinov3-size small

# Available DINOv3 sizes: small, base, large, giant
# Models are downloaded automatically from HuggingFace on first use
```

### Training with Pretrained YOLOv12 Weights
```python
# Using pretrained training script
python train_triple_pretrained.py \
    --pretrained yolov12n.pt \
    --data your_dataset.yaml \
    --epochs 100 \
    --batch 16

# Manual pretrained weight loading
from ultralytics import YOLO
from load_pretrained_triple import load_pretrained_weights_to_triple_model

# Load pretrained weights into triple input model
model = load_pretrained_weights_to_triple_model(
    pretrained_path='yolov12n.pt',
    triple_model_config='ultralytics/cfg/models/v12/yolov12_triple.yaml'
)

# Fine-tune with triple input data
results = model.train(
    data='your_dataset.yaml',
    epochs=100,
    batch=16,
    lr0=0.001,      # Lower learning rate for fine-tuning
    patience=50
)
```

### Training from Scratch
```python
from ultralytics import YOLO

# Load triple input model configuration
model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml')

# Train with automatic validation during training
results = model.train(
    data='your_dataset.yaml',  # Triple input dataset config
    epochs=100,
    batch=16,  # Smaller batch size due to 3x memory usage
    imgsz=640,
    val=True,        # Enable validation during training
    patience=50,     # Early stopping patience
    device="0,1"
)

# After training, evaluate on test set for final performance
test_results = model.val(
    data='your_dataset.yaml',
    split='test'     # Use test set for final evaluation
)
```

### Training and Evaluation Scripts
```bash
# Train with pretrained weights
python train_triple_pretrained.py \
    --pretrained yolov12n.pt \
    --data your_dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --patience 50

# Train from scratch
python train_with_validation.py \
    --model ultralytics/cfg/models/v12/yolov12_triple.yaml \
    --data your_dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --patience 50

# Load pretrained weights manually
python load_pretrained_triple.py \
    --pretrained yolov12s.pt \
    --model ultralytics/cfg/models/v12/yolov12_triple.yaml \
    --save yolov12s_triple_pretrained.pt

# Evaluate trained model
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
model.val(data='your_dataset.yaml', split='test')
"
```

### Dataset Structure for Triple Input
```
dataset_root/
├── images/
│   ├── train/                  # Training images
│   │   ├── image1.jpg          # Primary images
│   │   ├── image2.jpg
│   │   ├── detail1/            # Detail images 1
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── detail2/            # Detail images 2
│   │       ├── image1.jpg
│   │       └── image2.jpg
│   ├── val/                    # Validation images (used during training)
│   │   └── (same structure as train)
│   └── test/                   # Test images (used for final evaluation)
│       └── (same structure as train)
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    ├── val/
    │   └── (same structure as train)
    └── test/
        └── (same structure as train)
```

### Triple Input Dataset Configuration
```yaml
# your_dataset.yaml
path: ../datasets/your_dataset
train: images/train    # Training set
val: images/val        # Validation set (used during training)
test: images/test      # Test set (used for final evaluation)
triple_input: true     # Enable triple input mode
nc: 80                 # Number of classes
names: ['class1', 'class2', ...]  # Class names
```

### Manual Triple Input Processing
```python
import torch
from ultralytics.nn.modules.conv import TripleInputConv

# Method 1: Create with pretrained weights
conv = TripleInputConv.from_pretrained(
    c1=9, c2=64, 
    pretrained_model_path='yolov12n.pt',
    k=3, s=2
)

# Method 2: Standard creation
conv = TripleInputConv(c1=9, c2=64, k=3, s=2)

# Create 9-channel input (3 RGB images concatenated)
primary = torch.randn(1, 3, 640, 640)
detail1 = torch.randn(1, 3, 640, 640)
detail2 = torch.randn(1, 3, 640, 640)
triple_input = torch.cat([primary, detail1, detail2], dim=1)  # [1, 9, 640, 640]

# Process with TripleInputConv
output = conv(triple_input)  # [1, 64, 320, 320]
```

## 🔧 Triple Input Architecture Details

### TripleInputConv Module
```python
class TripleInputConv(nn.Module):
    """Processes 9-channel input as three separate 3-channel images.
    
    Supports loading pretrained weights from standard YOLOv12 models.
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, pretrained_weights=None):
        # c1: Input channels (must be 9)
        # c2: Output channels
        # Three separate Conv layers + fusion layer
        # pretrained_weights: Optional pretrained weights from YOLOv12
    
    @classmethod
    def from_pretrained(cls, c1, c2, pretrained_model_path, **kwargs):
        """Create TripleInputConv with pretrained YOLOv12 weights."""
        # Automatically loads and replicates first layer weights
```

### Key Features
- **Input**: 9-channel tensor (3 RGB images concatenated)
- **Processing**: Three independent 3-channel branches
- **Fusion**: Feature combination via 1×1 convolution
- **Pretrained Weight Support**: Compatible with YOLOv12 pretrained models
- **Transfer Learning**: Enables fine-tuning from standard YOLOv12 weights
- **Smart Fallback**: Uses primary image if detail images missing
- **Memory Efficient**: Optimized for practical deployment

### Architecture Comparison
| Feature | Standard YOLOv12 | Triple Input YOLOv12 |
|---------|------------------|---------------------|
| Input Channels | 3 (RGB) | 9 (3×RGB) |
| First Layer | `Conv(3, 64, 3, 2)` | `TripleInputConv(9, 64, 3, 2)` |
| Pretrained Weights | Direct loading | Compatible via transfer learning |
| Memory Usage | Baseline | ~3x for inputs |
| Dataset Format | Single images | Primary + 2 detail images |
| Fallback Support | N/A | Uses primary if details missing |
| Training Strategy | From scratch or pretrained | From scratch or YOLOv12 pretrained |

## 🧪 Quick Verification

```bash
# Test DINOv3 import
python -c "from ultralytics.nn.modules.dinov3 import create_dinov3_backbone; print('DINOv3 ready')"

# Test Triple Input import  
python -c "from ultralytics.nn.modules.conv import TripleInputConv; print('Triple Input ready')"

# Test model creation
python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml'); print('Model creation successful')"
```

## 🎨 Web Demo
```bash
python app.py
# Visit http://127.0.0.1:7860
```

## 📊 Export Models
```python
from ultralytics import YOLO

# Standard model
model = YOLO('yolov12n.pt')
model.export(format="engine", half=True)  # TensorRT

# Triple input model  
model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml')
model.export(format="onnx")  # ONNX format
```

## 🔧 Advanced Configuration

### Memory Optimization for Triple Input
```python
# Reduce batch size for triple input training
model.train(
    data='triple_dataset.yaml',
    batch=8,  # Reduced from 16 due to 3x memory usage
    workers=4,
    cache=False,  # Disable caching to save memory
    amp=True,     # Use automatic mixed precision
)
```

### Custom Triple Input Dataset Creation
```python
from pathlib import Path
import cv2

def create_triple_dataset(base_dir):
    """Create triple input dataset structure with train/val/test splits."""
    base_dir = Path(base_dir)
    
    # Create directory structure for train/val/test splits
    for split in ['train', 'val', 'test']:
        (base_dir / f'images/{split}').mkdir(parents=True, exist_ok=True)
        (base_dir / f'images/{split}/detail1').mkdir(parents=True, exist_ok=True) 
        (base_dir / f'images/{split}/detail2').mkdir(parents=True, exist_ok=True)
        (base_dir / f'labels/{split}').mkdir(parents=True, exist_ok=True)
    
    print(f"Triple input dataset structure created at {base_dir}")
    print("Remember:")
    print("- Use 'train' for training")
    print("- Use 'val' for validation during training")
    print("- Use 'test' for final evaluation after training")

# Usage
create_triple_dataset("my_triple_dataset")
```

## 📚 Documentation

For detailed documentation on the triple input and DINOv3 implementation, see:

### DINOv3 Integration
- [download_dinov3.py](download_dinov3.py) - DINOv3 model downloader (optional, automatic in training)
- [train_triple_dinov3.py](train_triple_dinov3.py) - Training script with DINOv3 backbone
- [ultralytics/nn/modules/dinov3.py](ultralytics/nn/modules/dinov3.py) - DINOv3 backbone implementation
- [ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml](ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml) - DINOv3 model configuration

### Triple Input Implementation
- [TRIPLE_INPUT_README.md](TRIPLE_INPUT_README.md) - Comprehensive implementation guide
- [train_triple_pretrained.py](train_triple_pretrained.py) - Training with pretrained weights
- [load_pretrained_triple.py](load_pretrained_triple.py) - Pretrained weight loading utilities
- [train_with_validation.py](train_with_validation.py) - Training script for training from scratch
- [debug_triple.py](debug_triple.py) - Debug utilities

## 🔄 Migration Guide

### From Standard YOLOv12 to Triple Input

1. **Update model configuration**: Use `yolov12_triple.yaml` instead of `yolov12.yaml`
2. **Prepare dataset**: Organize in triple input structure with train/val/test splits
3. **Adjust training parameters**: Reduce batch size due to increased memory usage
4. **Update data config**: Add `triple_input: true` and specify train/val/test splits
5. **Update workflow**: Use validation during training and test set for final evaluation

### Example Migration
```python
# Before (Standard YOLOv12)
model = YOLO('yolov12n.yaml')
model.train(data='coco.yaml', batch=32)

# After Option 1: Triple Input with DINOv3 (RECOMMENDED - automatic download)
# python train_triple_dinov3.py --data your_dataset.yaml --dinov3-size small --freeze-dinov3 --epochs 100 --batch 8

# After Option 2: Triple Input with pretrained weights
from load_pretrained_triple import load_pretrained_weights_to_triple_model

# Load pretrained weights into triple input model
model = load_pretrained_weights_to_triple_model(
    pretrained_path='yolov12n.pt',
    triple_model_config='ultralytics/cfg/models/v12/yolov12_triple.yaml'
)

# Fine-tune with triple input data
model.train(
    data='your_dataset.yaml', 
    batch=16,
    lr0=0.001,       # Lower learning rate for fine-tuning
    val=True,        # Enable validation during training
    patience=50      # Early stopping
)

# Final evaluation on test set
model.val(data='your_dataset.yaml', split='test')

# After Option 3: Triple Input with DINOv3 programmatically
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml')
model.train(
    data='your_dataset.yaml',
    epochs=100,
    batch=8,        # Smaller batch for DINOv3
    imgsz=224,      # DINOv3 optimized size
    lr0=0.001,      # Lower LR for frozen DINOv3
    optimizer='AdamW'
)
```

## 🤝 Contributing

We welcome contributions to the triple input implementation from the civil engineering and computer vision communities! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🏫 About KMUTT Civil Engineering Research Group

This implementation is developed by the Research Group at the Department of Civil Engineering, King Mongkut's University of Technology Thonburi (KMUTT). Our research focuses on applying advanced computer vision and AI techniques to civil engineering challenges including:

- **Infrastructure Monitoring**: Automated inspection and condition assessment
- **Structural Health Monitoring**: Real-time analysis of structural integrity
- **Construction Management**: Site monitoring and safety analysis
- **Smart Cities**: Urban infrastructure analysis and planning

For collaboration opportunities or research inquiries, please contact the research group.

## 📝 License

This project is licensed under the AGPL-3.0 License - see the original YOLOv12 license.

## 🙏 Acknowledgements

- **KMUTT Civil Engineering Research Group** for the triple input architecture development and DINOv3 integration
- **Meta AI Research** for [DINOv3](https://github.com/facebookresearch/dinov3) - self-supervised vision transformer foundation models
- **HuggingFace** for model hosting and transformers library integration
- Original YOLOv12 by [Yunjie Tian](https://sunsmarterjie.github.io/) et al.
- Triple input architecture inspired by [triple_YOLO13](https://github.com/Sompote/triple_YOLO13)
- Built on [ultralytics](https://github.com/ultralytics/ultralytics) framework
- King Mongkut's University of Technology Thonburi (KMUTT) for research support

## 📖 Citation

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@misc{yolov12_triple_input,
  title={YOLOv12 Triple Input with DINOv3 Integration for Civil Engineering Applications},
  author={Research Group, Department of Civil Engineering, KMUTT},
  institution={King Mongkut's University of Technology Thonburi},
  year={2025},
  note={Multi-image object detection enhancement with DINOv3 vision transformer backbone for infrastructure monitoring and analysis}
}

@misc{dinov3,
  title={DINOv3: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  year={2023},
  eprint={2304.07193},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

<div align="center">

**🚀 Ready to enhance your object detection with triple input architecture and DINOv3? Get started now!**

[🤖 DINOv3 Training](train_triple_dinov3.py) | [📥 Download DINOv3](download_dinov3.py) | [📚 Documentation](TRIPLE_INPUT_README.md) | [🔧 Pretrained Training](train_triple_pretrained.py) | [⚙️ Configuration](ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml)

</div>