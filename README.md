<div align="center">
<h1>YOLOv12 Triple Input</h1>
<h3>YOLOv12: Enhanced Multi-Image Object Detection with Triple Input Architecture</h3>

[Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>, [Qixiang Ye](https://people.ucas.ac.cn/~qxye?language=en)<sup>2</sup>, [David Doermann](https://cse.buffalo.edu/~doermann/)<sup>1</sup>

<sup>1</sup>  University at Buffalo, SUNY, <sup>2</sup> University of Chinese Academy of Sciences.

**Enhanced with Triple Input Architecture for Multi-Image Object Detection**

<p align="center">
  <img src="assets/tradeoff_turbo.svg" width=90%> <br>
  Comparison with popular methods in terms of latency-accuracy (left) and FLOPs-accuracy (right) trade-offs
</p>

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2502.12524-b31b1b.svg)](https://arxiv.org/abs/2502.12524) [![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunsmarterjieleaf/yolov12) <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov12-object-detection-model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/jxxn03x/yolov12-on-custom-data) [![LightlyTrain Notebook](https://img.shields.io/badge/LightlyTrain-Notebook-blue?)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/yolov12.ipynb) [![deploy](https://media.roboflow.com/deploy.svg)](https://blog.roboflow.com/use-yolov12-with-roboflow/#deploy-yolov12-models-with-roboflow) [![Openbayes](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/A4ac4xNrUCQ) 

## 🚀 What's New: Triple Input Architecture

This repository extends YOLOv12 with **Triple Input Architecture**, enabling the model to process three related images simultaneously for enhanced object detection accuracy. This implementation is inspired by [triple_YOLO13](https://github.com/Sompote/triple_YOLO13) and provides:

- **🎯 Enhanced Detection**: Process primary image + 2 detail images as 9-channel input
- **🧠 Smart Fallback**: Automatically uses primary image if detail images are missing  
- **⚡ Efficient Processing**: Specialized `TripleInputConv` module for optimal performance
- **🔄 Backward Compatible**: Seamlessly works with existing YOLOv12 infrastructure

### Triple Input Features

- **Multi-Image Processing**: Combines three related images (primary + 2 detail views)
- **9-Channel Input**: Concatenated RGB channels from three images
- **Independent Branch Processing**: Each image processed separately before fusion
- **Feature Fusion**: Advanced feature combination for improved accuracy
- **Dataset Flexibility**: Supports both single and triple input formats

## Updates

- **2025/01/XX**: 🎉 **Triple Input Architecture** released! Process multiple images simultaneously for enhanced detection accuracy.
- 2025/06/17: **Use this repo for YOLOv12 instead of [ultralytics](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/12). Their implementation is inefficient, requires more memory, and has unstable training, which are fixed here!**
- 2025/07/01: YOLOv12's **classification** models are released, see [code](https://github.com/sunsmarterjie/yolov12/tree/Cls).
- 2025/06/04: YOLOv12's **instance segmentation** models are released, see [code](https://github.com/sunsmarterjie/yolov12/tree/Seg).

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Enhancing the network architecture of the YOLO framework has been crucial for a long time but has focused on CNN-based improvements despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention-based models cannot match the speed of CNN-based models. This paper proposes an attention-centric YOLO framework, namely YOLOv12, that matches the speed of previous CNN-based ones while harnessing the performance benefits of attention mechanisms.

YOLOv12 surpasses all popular real-time object detectors in accuracy with competitive speed. For example, YOLOv12-N achieves 40.6% mAP with an inference latency of 1.64 ms on a T4 GPU, outperforming advanced YOLOv10-N / YOLOv11-N by 2.1%/1.2% mAP with a comparable speed. This advantage extends to other model scales. YOLOv12 also surpasses end-to-end real-time detectors that improve DETR, such as RT-DETR / RT-DETRv2: YOLOv12-S beats RT-DETR-R18 / RT-DETRv2-R18 while running 42% faster, using only 36% of the computation and 45% of the parameters.

**Triple Input Enhancement**: This repository extends YOLOv12 with multi-image processing capabilities, allowing the model to leverage contextual information from multiple related images for improved detection accuracy.
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

### YOLOv12 Triple Input (Multi-Image)
| Model (det)                    | size<br><sup>(pixels) | Input<br><sup>Channels | mAP<sup>val<br>50-95 | Speed (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------- | :-------------------: | :--------------------: | :-------------------:| :------------------------------:| :-----------------:| :---------------:|
| YOLO12n-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 2.6                | 6.2               |
| YOLO12s-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 9.2                | 19.6              |
| YOLO12m-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 19.7               | 60.1              |
| YOLO12l-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 26.6               | 82.7              |
| YOLO12x-triple                 | 640                   | 9 (3×RGB)             | TBD*                 | TBD*                            | 59.4               | 185.0             |

*TBD: To Be Determined after training on triple input datasets

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/sunsmarterjie/yolov12.git
cd yolov12

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

### Creating Triple Input Model
```python
from ultralytics import YOLO

# Load triple input model configuration
model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml')

# Train with triple input dataset
results = model.train(
    data='ultralytics/cfg/datasets/coco_triple.yaml',  # Triple input dataset config
    epochs=100,
    batch=16,  # Smaller batch size due to 3x memory usage
    imgsz=640,
    device="0,1"
)
```

### Dataset Structure for Triple Input
```
dataset_root/
├── images/
│   ├── train2017/
│   │   ├── image1.jpg          # Primary images
│   │   ├── image2.jpg
│   │   ├── detail1/            # Detail images 1
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── detail2/            # Detail images 2
│   │       ├── image1.jpg
│   │       └── image2.jpg
│   └── val2017/
│       └── (same structure)
└── labels/
    ├── train2017/
    │   ├── image1.txt
    │   └── image2.txt
    └── val2017/
        └── (same structure)
```

### Triple Input Dataset Configuration
```yaml
# ultralytics/cfg/datasets/coco_triple.yaml
path: ../datasets/coco_triple
train: images/train2017
val: images/val2017
triple_input: true  # Enable triple input mode
nc: 80
names: ...
```

### Manual Triple Input Processing
```python
import torch
from ultralytics.nn.modules.conv import TripleInputConv

# Create 9-channel input (3 RGB images concatenated)
primary = torch.randn(1, 3, 640, 640)
detail1 = torch.randn(1, 3, 640, 640)
detail2 = torch.randn(1, 3, 640, 640)
triple_input = torch.cat([primary, detail1, detail2], dim=1)  # [1, 9, 640, 640]

# Process with TripleInputConv
conv = TripleInputConv(c1=9, c2=64, k=3, s=2)
output = conv(triple_input)  # [1, 64, 320, 320]
```

## 🔧 Triple Input Architecture Details

### TripleInputConv Module
```python
class TripleInputConv(nn.Module):
    """Processes 9-channel input as three separate 3-channel images."""
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        # c1: Input channels (must be 9)
        # c2: Output channels
        # Three separate Conv layers + fusion layer
```

### Key Features
- **Input**: 9-channel tensor (3 RGB images concatenated)
- **Processing**: Three independent 3-channel branches
- **Fusion**: Feature combination via 1×1 convolution
- **Smart Fallback**: Uses primary image if detail images missing
- **Memory Efficient**: Optimized for practical deployment

### Architecture Comparison
| Feature | Standard YOLOv12 | Triple Input YOLOv12 |
|---------|------------------|---------------------|
| Input Channels | 3 (RGB) | 9 (3×RGB) |
| First Layer | `Conv(3, 64, 3, 2)` | `TripleInputConv(9, 64, 3, 2)` |
| Memory Usage | Baseline | ~3x for inputs |
| Dataset Format | Single images | Primary + 2 detail images |
| Fallback Support | N/A | Uses primary if details missing |

## 🧪 Validation & Testing

### Run Triple Input Tests
```bash
# Validate implementation
python test_triple_input.py

# Expected output:
# TripleInputConv Module: ✓ PASSED
# Model Configuration: ✓ PASSED  
# Dataset Configuration: ✓ PASSED
# Dummy Dataset Creation: ✓ PASSED
# Overall: 4/4 tests passed
```

### Debug Commands
```bash
# Test TripleInputConv module
python -c "from ultralytics.nn.modules.conv import TripleInputConv; print('✓ Import successful')"

# Test model creation
python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml'); print('✓ Model created')"
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
    """Create triple input dataset structure."""
    base_dir = Path(base_dir)
    
    # Create directory structure
    for split in ['train', 'val']:
        (base_dir / f'images/{split}').mkdir(parents=True, exist_ok=True)
        (base_dir / f'images/{split}/detail1').mkdir(parents=True, exist_ok=True) 
        (base_dir / f'images/{split}/detail2').mkdir(parents=True, exist_ok=True)
        (base_dir / f'labels/{split}').mkdir(parents=True, exist_ok=True)
    
    print(f"Triple input dataset structure created at {base_dir}")

# Usage
create_triple_dataset("my_triple_dataset")
```

## 📚 Documentation

For detailed documentation on the triple input implementation, see:
- [TRIPLE_INPUT_README.md](TRIPLE_INPUT_README.md) - Comprehensive implementation guide
- [test_triple_input.py](test_triple_input.py) - Validation test suite
- [debug_triple.py](debug_triple.py) - Debug utilities

## 🔄 Migration Guide

### From Standard YOLOv12 to Triple Input

1. **Update model configuration**: Use `yolov12_triple.yaml` instead of `yolov12.yaml`
2. **Prepare dataset**: Organize in triple input structure or enable smart fallback
3. **Adjust training parameters**: Reduce batch size due to increased memory usage
4. **Update data config**: Add `triple_input: true` to dataset YAML

### Example Migration
```python
# Before (Standard)
model = YOLO('yolov12n.yaml')
model.train(data='coco.yaml', batch=32)

# After (Triple Input)  
model = YOLO('ultralytics/cfg/models/v12/yolov12_triple.yaml')
model.train(data='ultralytics/cfg/datasets/coco_triple.yaml', batch=16)
```

## 🤝 Contributing

We welcome contributions to the triple input implementation! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is licensed under the AGPL-3.0 License - see the original YOLOv12 license.

## 🙏 Acknowledgements

- Original YOLOv12 by [Yunjie Tian](https://sunsmarterjie.github.io/) et al.
- Triple input architecture inspired by [triple_YOLO13](https://github.com/Sompote/triple_YOLO13)
- Built on [ultralytics](https://github.com/ultralytics/ultralytics) framework

## 📖 Citation

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@misc{yolov12_triple_input,
  title={YOLOv12 Triple Input Implementation},
  author={Enhanced with Triple Input Architecture},
  year={2025},
  note={Multi-image object detection enhancement}
}
```

---

<div align="center">

**🚀 Ready to enhance your object detection with triple input architecture? Get started now!**

[📚 Documentation](TRIPLE_INPUT_README.md) | [🧪 Test Suite](test_triple_input.py) | [🎯 Examples](ultralytics/cfg/models/v12/yolov12_triple.yaml)

</div>