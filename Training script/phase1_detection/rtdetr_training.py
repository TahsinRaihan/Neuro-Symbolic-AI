# -*- coding: utf-8 -*-
"""
PHASE 1B: RT-DETR Detection Training
Trains RT-DETR-L on the combined Dataset 1 phase-1 preparation
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
from pathlib import Path
from tqdm import tqdm

from phase1_dataset_builder import build_phase1_combined_dataset

print("=" * 80)
print(" " * 25 + "PHASE 1B: RT-DETR DETECTION TRAINING")
print(" " * 20 + "(RT-DETR-L on combined Dataset 1)")
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
    EPOCHS = 100
    RTDETR_BATCH = 12  # Optimized for 16 GB VRAM (increased from 8)
    IMG_SIZE = 640
    DEVICE = 'cuda'   # Force GPU usage
    NUM_WORKERS = 2   # Reduced for disk cache (prevents memory explosion)

    # ROI Cropping Config
    ROI_SIZE = 256
    ROI_BUFFER = 20
    CROPPED_OUTPUT = OUTPUT_ROOT / 'cropped_rois'

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
# MASK TO YOLO CONVERSION (for data.yaml check)
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
            if cv2.contourArea(cnt) < 100:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x_center = np.clip((x + bw / 2) / w, 0, 1)
            y_center = np.clip((y + bh / 2) / h, 0, 1)
            norm_width = np.clip(bw / w, 0, 1)
            norm_height = np.clip(bh / h, 0, 1)

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
# RT-DETR TRAINING
# ==========================================
def train_rtdetr():
    """Train RT-DETR model"""
    print("\n" + "=" * 80)
    print(" " * 28 + "TRAINING RT-DETR")
    print("=" * 80)

    try:
        from ultralytics import RTDETR

        phase1_dataset = build_phase1_combined_dataset(Config.THESIS_ROOT)
        data_yaml = phase1_dataset.data_yaml

        # Check for corrupted checkpoints and clean them up
        corrupted_dir = Config.DETECTION_OUTPUT / 'rtdetr_polyp'
        for bad_ckpt in corrupted_dir.rglob('*.pt') if corrupted_dir.exists() else []:
            try:
                import torch as _torch
                ckpt = _torch.load(str(bad_ckpt), map_location='cpu')
                model_weights = ckpt.get('model', None)
                if model_weights is not None:
                    flat = _torch.cat([p.flatten() for p in
                                       (model_weights.parameters() if hasattr(model_weights, 'parameters')
                                        else [_torch.tensor([])])])
                    if not _torch.isfinite(flat).all():
                        print(f"   🗑️  Removing corrupted checkpoint: {bad_ckpt.name}")
                        bad_ckpt.unlink()
            except Exception:
                pass

        print(f"\n🚀 Training RT-DETR for {Config.EPOCHS} epochs (AMP disabled for stability)...")
        
        # Check for existing checkpoint to resume from last training
        checkpoint_path = Config.DETECTION_OUTPUT / 'rtdetr_polyp' / 'weights' / 'last.pt'
        if checkpoint_path.exists():
            print(f"   ✅ Found checkpoint at {checkpoint_path}")
            print(f"   🔄 Resuming training from checkpoint...")
            model = RTDETR(str(checkpoint_path))  # Load from checkpoint directly (not base model)
        else:
            model = RTDETR('rtdetr-l.pt')  # Fresh model for first training

        results = model.train(
            data=str(data_yaml),
            epochs=Config.EPOCHS,
            batch=Config.RTDETR_BATCH,
            imgsz=Config.IMG_SIZE,
            device=Config.DEVICE,
            workers=Config.NUM_WORKERS,
            project=str(Config.DETECTION_OUTPUT),
            name='rtdetr_polyp',
            exist_ok=True,
            verbose=False,
            plots=True,
            patience=50,
            amp=False,      # MUST be False: deformable attention grid_sample diverges in FP16
            cache=None,     # No disk cache (saves SSD storage), load images on-the-fly
            lr0=1e-4,       # Lower LR to prevent NaN explosion
            lrf=0.01,       # Final LR = lr0 * lrf
            warmup_epochs=min(3, Config.EPOCHS),  # Never exceed total epochs
            weight_decay=1e-4,
            optimizer='AdamW'
        )

        metrics = model.val()
        print(f"\n✅ RT-DETR Training Complete!")
        print(f"   mAP50: {float(metrics.box.map50):.4f}")
        print(f"   mAP50-95: {float(metrics.box.map):.4f}")

        best_path = Config.DETECTION_OUTPUT / 'rtdetr_best.pt'
        model.save(str(best_path))

        return True

    except Exception as e:
        print(f"\n⚠️  RT-DETR Training Failed: {e}")
        print("   This is optional - continuing with YOLO only")
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
        print(" " * 20 + "STARTING RT-DETR TRAINING")
        print("=" * 80)

        success = train_rtdetr()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print(" " * 25 + "RT-DETR TRAINING COMPLETE!")
        print("=" * 80)
        status = "✅ Success" if success else "⚠️  Skipped/Failed"
        print(f"\n   RT-DETR: {status}")
        print(f"\n📁 Model saved to: {Config.DETECTION_OUTPUT}")
        print("\n➡️  Next: python phase1_detection/unetpp_training.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
