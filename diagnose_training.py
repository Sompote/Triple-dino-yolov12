#!/usr/bin/env python3
"""
Diagnostic script to identify training issues with YOLOv12 Triple Input DINOv3 model.
"""

import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset


def diagnose_training_issues(data_config_path: str, model_config_path: str):
    """Diagnose potential training issues."""
    print("🔍 YOLOv12 Triple DINOv3 Training Diagnostics")
    print("=" * 60)
    
    # Load dataset config
    print(f"\n1. 📂 Dataset Configuration Analysis:")
    try:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"✓ Dataset config loaded: {data_config_path}")
        print(f"  Number of classes: {data_config.get('nc', 'NOT FOUND')}")
        print(f"  Class names: {data_config.get('names', 'NOT FOUND')}")
        print(f"  Train path: {data_config.get('train', 'NOT FOUND')}")
        print(f"  Val path: {data_config.get('val', 'NOT FOUND')}")
    except Exception as e:
        print(f"❌ Error loading dataset config: {e}")
        return
    
    # Check dataset structure
    print(f"\n2. 📁 Dataset Structure Analysis:")
    train_path = data_config.get('train', '')
    if train_path and Path(train_path).exists():
        base_path = Path(train_path).parent.parent
        expected_folders = ["primary", "detail1", "detail2"]
        is_triple_input = all((base_path / "images" / folder).exists() for folder in expected_folders)
        
        print(f"  Dataset base path: {base_path}")
        print(f"  Triple input structure: {'✓ Detected' if is_triple_input else '❌ Not found'}")
        
        # Count images and labels
        if is_triple_input:
            for phase in ["train", "val", "test"]:
                img_count = len(list((base_path / "images" / "primary" / phase).glob("*"))) if (base_path / "images" / "primary" / phase).exists() else 0
                label_count = len(list((base_path / "labels" / "primary" / phase).glob("*.txt"))) if (base_path / "labels" / "primary" / phase).exists() else 0
                print(f"  {phase.upper()}: {img_count} images, {label_count} labels")
                
                # Check for empty labels
                if label_count > 0:
                    label_dir = base_path / "labels" / "primary" / phase
                    empty_labels = 0
                    total_objects = 0
                    for label_file in label_dir.glob("*.txt"):
                        with open(label_file, 'r') as f:
                            lines = f.read().strip().split('\n')
                            if not lines or lines == ['']:
                                empty_labels += 1
                            else:
                                total_objects += len(lines)
                    
                    print(f"    Empty labels: {empty_labels}/{label_count}")
                    print(f"    Total objects: {total_objects}")
                    if label_count > 0:
                        print(f"    Avg objects per image: {total_objects/label_count:.2f}")
    else:
        print(f"  ❌ Train path does not exist: {train_path}")
    
    # Model loading test
    print(f"\n3. 🤖 Model Configuration Analysis:")
    try:
        if Path(model_config_path).exists():
            model = YOLO(model_config_path)
            print(f"✓ Model config loaded: {model_config_path}")
            print(f"  Model type: {type(model.model).__name__}")
            
            # Check model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Frozen ratio: {((total_params - trainable_params) / total_params * 100):.1f}%")
            
            # Check for DINOv3 components
            dinov3_params = 0
            dinov3_frozen = 0
            for name, param in model.model.named_parameters():
                if any(x in name for x in ['.dino_model.', '.dino_branch1.', '.dino_branch2.', '.dino_branch3.']):
                    dinov3_params += param.numel()
                    if not param.requires_grad:
                        dinov3_frozen += param.numel()
            
            if dinov3_params > 0:
                print(f"  DINOv3 parameters: {dinov3_params:,}")
                print(f"  DINOv3 frozen: {dinov3_frozen:,} ({dinov3_frozen/dinov3_params*100:.1f}%)")
            
        else:
            print(f"❌ Model config not found: {model_config_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    # Dataset loading test
    print(f"\n4. 📊 Dataset Loading Test:")
    try:
        # Try to build dataset
        dataset = build_yolo_dataset(
            args=type('Args', (), {
                'cache': False,
                'data': data_config_path,
                'imgsz': 224,
                'augment': False,
                'rect': False,
                'batch_size': 1,
                'workers': 0,
                'close_mosaic': 0,
                'overlap_mask': False,
                'mask_ratio': 4,
                'single_cls': False,
                'degrees': 0.0,
                'translate': 0.0,
                'scale': 0.0,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.0,
                'hsv_h': 0.0,
                'hsv_s': 0.0,
                'hsv_v': 0.0,
                'mosaic': 0.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'auto_augment': None,
                'erasing': 0.0,
                'crop_fraction': 1.0,
            })(),
            img_path=data_config.get('train', ''),
            batch=1,
            data=data_config,
            mode='train',
            rect=False,
            stride=32
        )
        
        print(f"✓ Dataset built successfully")
        print(f"  Dataset type: {type(dataset).__name__}")
        print(f"  Dataset length: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample image shape: {sample['img'].shape}")
            print(f"  Sample has bboxes: {'bboxes' in sample}")
            print(f"  Sample has cls: {'cls' in sample}")
            
            if 'cls' in sample and len(sample['cls']) > 0:
                print(f"  Sample objects: {len(sample['cls'])}")
                print(f"  Sample classes: {sample['cls'].tolist() if hasattr(sample['cls'], 'tolist') else sample['cls']}")
            else:
                print(f"  ⚠️ Sample has no objects!")
                
    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Recommendations
    print(f"\n5. 💡 Recommendations:")
    print("Based on the analysis:")
    print("  1. If too many empty labels → Check label format and annotation quality")
    print("  2. If high frozen ratio → Verify DINOv3 freezing is intentional")
    print("  3. If dataset loading fails → Check image-label correspondence")
    print("  4. If zero mAP persists → Check class definitions and target assignment")


if __name__ == "__main__":
    import sys
    
    # Default paths (update these for your setup)
    data_config = "/workspace/dataset3/dataset.yaml"
    model_config = "ultralytics/cfg/models/v12/yolov12_triple_dinov3.yaml"
    
    if len(sys.argv) > 1:
        data_config = sys.argv[1]
    if len(sys.argv) > 2:
        model_config = sys.argv[2]
    
    diagnose_training_issues(data_config, model_config)