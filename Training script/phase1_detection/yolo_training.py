# -*- coding: utf-8 -*-
"""
PHASE 1A: YOLOv8 Detection Training
Trains YOLOv8m on the combined Dataset 1 phase-1 preparation
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

from phase1_dataset_builder import build_phase1_combined_dataset

print("=" * 80)
print(" " * 25 + "PHASE 1A: YOLO DETECTION TRAINING")
print(" " * 20 + "(YOLOv8m on combined Dataset 1)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths (auto-detect thesis root)
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    SOURCE_DATASET_ROOT = THESIS_ROOT / 'NeSy' / 'Dataset 1'
    KVASIR_ROOT = SOURCE_DATASET_ROOT / 'Kvasir-SEG'
    POLYPGEN_ROOT = SOURCE_DATASET_ROOT / 'PolypGen2021_MultiCenterData_v3' / 'PolypGen2021_MultiCenterData_v3'
    CROPPED_SEQUENCE_ROOT = SOURCE_DATASET_ROOT / 'sequence_data_positive_cropped'
    DATASET_1_ROOT = THESIS_ROOT / 'thesis_outputs' / 'phase1_combined_dataset'
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    DETECTION_OUTPUT = OUTPUT_ROOT / 'detection_models'
    SEGMENTATION_OUTPUT = OUTPUT_ROOT / 'segmentation_models'

    # Training Config (RTX 4080 SUPER with 16 GB VRAM, 64 GB System RAM)
    # PIPELINE_EPOCHS env var overrides default (e.g. set to 2 for a quick test run)
    EPOCHS = 100  # Resume from checkpoint → train to 100 total epochs
    YOLO_BATCH = 24   # RTX 4080 SUPER 16GB VRAM – YOLOv8m @ 960px with AMP (optimized)
    IMG_SIZE = 960
    DEVICE = 'cuda'   # Force GPU usage
    NUM_WORKERS = 2  # Reduced to 2 for disk cache (prevents memory explosion)

    # ROI Cropping Config
    ROI_SIZE = 256
    ROI_BUFFER = 20
    CROPPED_OUTPUT = OUTPUT_ROOT / 'cropped_rois'

    # UNet++ Config (referenced only for dir creation)
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    LR = 0.0001

# Create directories
Config.DETECTION_OUTPUT.mkdir(parents=True, exist_ok=True)
Config.SEGMENTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
Config.CROPPED_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n[Configuration]:")
print(f"   Device: {Config.DEVICE}")
print(f"   Epochs: {Config.EPOCHS}")
print(f"   Source data: {Config.SOURCE_DATASET_ROOT}")
print(f"   Prepared dataset: {Config.DATASET_1_ROOT}")
print(f"   Output: {Config.OUTPUT_ROOT}")

# ==========================================
# PART 1: MASK TO YOLO CONVERSION
# ==========================================
def convert_masks_to_yolo_labels(masks_dir, labels_dir):
    """Convert segmentation masks to YOLO bounding box format"""
    print("\n" + "=" * 80)
    print(" " * 25 + "CONVERTING MASKS TO LABELS")
    print("=" * 80)

    labels_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted([f for f in masks_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png']])

    converted = 0
    total_polyps = 0

    for mask_file in tqdm(mask_files, desc="Converting masks"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        h, w = mask.shape
        _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_content = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Lower cutoff, audit manually
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x_center = np.clip((x + bw / 2) / w, 0, 1)
            y_center = np.clip((y + bh / 2) / h, 0, 1)
            norm_width = np.clip(bw / w, 0, 1)
            norm_height = np.clip(bh / h, 0, 1)

            # Validate label noise: ensure reasonable box size
            if norm_width < 0.01 or norm_height < 0.01 or norm_width > 1 or norm_height > 1:
                print(f"   ⚠️  Skipping invalid box: w={norm_width:.3f}, h={norm_height:.3f}")
                continue

            label_content.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            total_polyps += 1

        if label_content:
            label_file = labels_dir / f"{mask_file.stem}.txt"
            label_file.write_text('\n'.join(label_content))
            converted += 1
        else:
            (labels_dir / f"{mask_file.stem}.txt").write_text('')

    print(f"\n✅ Conversion complete:")
    print(f"   Images with polyps: {converted}/{len(mask_files)}")
    print(f"   Total polyp instances: {total_polyps}")

    return converted > 0

# ==========================================
# PART 2: YOLO TRAINING
# ==========================================
def train_yolo():
    """Train YOLOv8 model"""
    print("\n" + "=" * 80)
    print(" " * 30 + "TRAINING YOLO")
    print("=" * 80)

    try:
        from ultralytics import YOLO
        # Pin versions for reproducibility
        import ultralytics
        if ultralytics.__version__ != '8.4.21':  # Current version
            print(f"   ⚠️  Ultralytics version {ultralytics.__version__} may differ from tested 8.4.21")

        phase1_dataset = build_phase1_combined_dataset(Config.THESIS_ROOT)
        data_yaml = phase1_dataset.data_yaml

        # Train
        print(f"\n🚀 Training YOLOv8 for {Config.EPOCHS} epochs...")
        
        # Check for existing checkpoint to resume from last training
        checkpoint_path = Config.DETECTION_OUTPUT / 'yolov8m' / 'weights' / 'last.pt'
        if checkpoint_path.exists():
            print(f"   ✅ Found checkpoint at {checkpoint_path}")
            print(f"   🔄 Resuming training from checkpoint...")
            model = YOLO(str(checkpoint_path))  # Load from checkpoint directly (not base model)
        else:
            model = YOLO('yolov8m.pt')  # Fresh model for first training

        results = model.train(
            data=str(data_yaml),
            epochs=Config.EPOCHS,
            batch=Config.YOLO_BATCH,
            imgsz=Config.IMG_SIZE,
            device=Config.DEVICE,
            workers=Config.NUM_WORKERS,
            project=str(Config.DETECTION_OUTPUT),
            name='yolov8m',  # Match the checkpoint directory
            exist_ok=True,
            verbose=False,
            plots=False,  # Disable plots for speed
            patience=50,
            amp=True,      # Mixed precision for faster training on 16GB VRAM
            cache=None,    # No disk cache (saves SSD storage), load images on-the-fly
            close_mosaic=10,  # Disable mosaic last 10 epochs for better convergence
            resume=True,   # Resume from checkpoint
            # Endoscopy-realistic augmentations: blur, glare, motion, compression
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Lighting changes
            degrees=5.0, translate=0.05, scale=0.3, shear=2.0,  # Motion artifacts
            perspective=0.001, flipud=0.0, fliplr=0.5,  # Scope orientation
            mosaic=0.5, mixup=0.1,  # Domain mixing
            multi_scale=False  # Fixed scale for consistency
        )

        return True

    except Exception as e:
        print(f"\n⚠️  YOLOv8 Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        print("\n[*] Verifying Dataset 1...")
        if not Config.SOURCE_DATASET_ROOT.exists():
            print(f"[X] Dataset 1 source root not found at {Config.SOURCE_DATASET_ROOT}")
            sys.exit(1)

        print("[OK] Dataset 1 source root found\n")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print(" " * 20 + "STARTING YOLO TRAINING")
        print("=" * 80)

        success = train_yolo()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print(" " * 25 + "YOLO TRAINING COMPLETE!")
        print("=" * 80)
        status = "✅ Success" if success else "⚠️  Failed"
        print(f"\n   YOLOv8: {status}")
        print(f"\n📁 Model saved to: {Config.DETECTION_OUTPUT}")
        print("\n➡️  Next: python phase1_detection/rtdetr_training.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
