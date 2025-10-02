"""
Triple input dataset class for YOLOv12 Triple Input training.

This module handles loading and combining three separate images (primary, detail1, detail2)
into a single 9-channel input tensor for triple input training.
"""

import os
import cv2
import math
import numpy as np
from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class TripleInputDataset(YOLODataset):
    """
    Dataset class for triple input training.
    
    Extends YOLODataset to load three images (primary, detail1, detail2) and combine them
    into a single 9-channel tensor.
    """
    
    def __init__(self, img_path, imgsz=640, cache=False, augment=True, hyp=None, prefix="", rect=False, batch_size=16, stride=32, pad=0.0, single_cls=False, classes=None, fraction=1.0, data=None, task="detect", **kwargs):
        """Initialize triple input dataset."""
        
        # Store triple input paths
        self.img_path = img_path
        self.is_triple_input = self._detect_triple_input_structure()
        
        if self.is_triple_input:
            LOGGER.info(f"Triple input structure detected at {img_path}")
            # For triple input, we use the provided img_path as-is since it already points to primary
            super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, fraction, data=data, task=task, **kwargs)
        else:
            LOGGER.warning(f"Standard dataset structure detected at {img_path} - using single image input")
            super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, fraction, data=data, task=task, **kwargs)
    
    def _detect_triple_input_structure(self):
        """Detect if the dataset has triple input structure (primary, detail1, detail2)."""
        base_path = Path(self.img_path).parent.parent
        
        expected_folders = ["primary", "detail1", "detail2"]
        for folder in expected_folders:
            folder_path = base_path / folder / Path(self.img_path).name
            if not folder_path.exists():
                return False
        
        return True
    
    def load_image(self, i, rect_mode=True):
        """Load and combine triple input images."""
        if not self.is_triple_input:
            # Fall back to standard single image loading
            return super().load_image(i, rect_mode)
        
        # Get the image path from the dataset
        f = self.im_files[i]
        base_name = Path(f).name
        base_dir = Path(f).parent.parent.parent  # Go up to dataset root
        
        # Define the three image paths
        primary_path = base_dir / "primary" / Path(f).parent.name / base_name
        detail1_path = base_dir / "detail1" / Path(f).parent.name / base_name  
        detail2_path = base_dir / "detail2" / Path(f).parent.name / base_name
        
        # Load the three images
        images = []
        original_sizes = []
        
        for img_path in [primary_path, detail1_path, detail2_path]:
            if img_path.exists():
                # Load image
                im = cv2.imread(str(img_path))  # Keep BGR format like original
                if im is None:
                    LOGGER.warning(f"Failed to load image: {img_path}")
                    # Create a dummy image with same size as expected
                    im = np.zeros((640, 640, 3), dtype=np.uint8)
                images.append(im)
                original_sizes.append(im.shape[:2])
            else:
                LOGGER.warning(f"Image not found: {img_path}")
                # Create a dummy image
                im = np.zeros((640, 640, 3), dtype=np.uint8) 
                images.append(im)
                original_sizes.append((640, 640))
        
        # Use the original size from the first (primary) image
        h0, w0 = original_sizes[0] if original_sizes else (640, 640)
        
        # Ensure all images have the same size
        if len(images) == 3:
            # Get the target size (use the first image's size)
            target_h, target_w = images[0].shape[:2]
            
            # Apply the same resizing logic as the original load_image method
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    target_h, target_w = h, w
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                target_h, target_w = self.imgsz, self.imgsz
            
            # Resize all images to the target size
            resized_images = []
            for img in images:
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_images.append(img)
            
            # Combine into 9-channel image
            combined = np.concatenate(resized_images, axis=2)  # Shape: (H, W, 9)
            
            # Handle caching like the original method
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = combined, (h0, w0), combined.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            
            return combined, (h0, w0), combined.shape[:2]
        else:
            LOGGER.error(f"Failed to load all three images for index {i}")
            # Return a dummy 9-channel image
            dummy = np.zeros((self.imgsz, self.imgsz, 9), dtype=np.uint8)
            return dummy, (self.imgsz, self.imgsz), dummy.shape[:2]
    
    def cache_images_to_disk(self, i):
        """Cache triple input images to disk (override to handle 9-channel images)."""
        if not self.is_triple_input:
            return super().cache_images_to_disk(i)
        
        # For triple input, we don't cache to avoid complexity
        # The images will be loaded fresh each time
        return None


def build_triple_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, single_cls=False, augment=False, cache=False, prefix="", classes=None, fraction=1.0):
    """Build triple input dataset."""
    
    # Check if this is a triple input dataset structure
    base_path = Path(img_path).parent.parent if Path(img_path).parent.parent.exists() else Path(img_path).parent
    
    expected_folders = ["primary", "detail1", "detail2"]
    is_triple = all((base_path / folder).exists() for folder in expected_folders)
    
    if is_triple:
        LOGGER.info(f"Building TripleInputDataset for {mode}")
        return TripleInputDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=augment,  # augmentation
            hyp=cfg,  # hyperparameters  
            rect=rect,  # rectangular batches
            cache=cache,
            single_cls=single_cls or cfg.single_cls or False,  # single class training
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=prefix,
            classes=classes,
            fraction=fraction if mode == "train" else 1.0,
        )
    else:
        LOGGER.info(f"Building standard YOLODataset for {mode}")
        return YOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=augment,
            hyp=cfg,
            rect=rect,
            cache=cache,
            single_cls=single_cls or cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=prefix,
            classes=classes,
            fraction=fraction if mode == "train" else 1.0,
        )