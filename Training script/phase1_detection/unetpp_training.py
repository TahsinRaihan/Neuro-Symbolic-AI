# -*- coding: utf-8 -*-
"""
PHASE 1C: UNet++ Segmentation Training + ROI Extraction
Trains UNet++ (ResNet34 encoder) on the combined Dataset 1 phase-1 preparation
then extracts 256x256 ROI crops from ground-truth masks.
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

from phase1_dataset_builder import build_phase1_combined_dataset, find_first_matching_file, IMAGE_EXTENSIONS

print("=" * 80)
print(" " * 20 + "PHASE 1C: UNET++ SEGMENTATION TRAINING")
print(" " * 15 + "(UNet++ ResNet34 on combined Dataset 1 + ROI Extraction)")
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
    EPOCHS = 100
    UNET_BATCH = 24   # Optimized for 16 GB VRAM – UNet++ resnet34 @ 512px with AMP
    UNET_IMG_SIZE = 512  # Higher resolution for better segmentation quality
    DEVICE = 'cuda'   # Force GPU usage
    NUM_WORKERS = 2   # Optimal for disk cache (prevents memory explosion)

    # ROI Cropping Config
    ROI_SIZE = 256
    ROI_BUFFER = 20
    CROPPED_OUTPUT = OUTPUT_ROOT / 'cropped_rois'

    # UNet++ Config
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
# PART 1: DATASET CLASS (Module Level for pickle compatibility)
# ==========================================
class PolypSegmentationDataset(torch.utils.data.Dataset):
    """Dataset for polyp segmentation - Module level for pickle compatibility"""

    def __init__(self, images_dir, masks_dir, img_size=320):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size

        # Get all image and mask files
        all_images = {f.stem: f for f in self.images_dir.glob('*')
                     if f.suffix.lower() in IMAGE_EXTENSIONS}
        all_masks = {f.stem: f for f in self.masks_dir.glob('*')
                    if f.suffix.lower() in IMAGE_EXTENSIONS}

        # Only keep images that have corresponding masks
        common_stems = sorted(set(all_images.keys()) & set(all_masks.keys()))

        self.image_files = [all_images[stem] for stem in common_stems]
        self.mask_files = [all_masks[stem] for stem in common_stems]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.mask_files[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Normalize mask
        mask = (mask > 10).astype(np.float32)

        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# ==========================================
# UNET++ METRICS
# ==========================================
def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Jaccard) score"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

# ==========================================
# UNET++ TRAINING FUNCTIONS
# ==========================================
def train_unet_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train UNet++ for one epoch (AMP-aware)"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward + Backward with optional AMP
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # Metrics
        with torch.no_grad():
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        total_acc += acc.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n, total_acc/n

def validate_unet_epoch(model, loader, criterion, device):
    """Validate UNet++ for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            total_acc += acc.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n, total_acc/n

# ==========================================
# UNET++ MAIN TRAINING
# ==========================================
def train_unet():
    """Train UNet++ segmentation model"""
    print("\n" + "=" * 80)
    print(" " * 28 + "TRAINING UNET++")
    print("=" * 80)

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("\n⚠️  Installing segmentation-models-pytorch...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                             "segmentation-models-pytorch", "-q"])
        import segmentation_models_pytorch as smp
        print("✅ Installed!")

    try:
        phase1_dataset = build_phase1_combined_dataset(Config.THESIS_ROOT)

        # Prepare dataset splits
        train_images_dir = phase1_dataset.train_images
        train_masks_dir = phase1_dataset.train_masks
        val_images_dir = phase1_dataset.val_images
        val_masks_dir = phase1_dataset.val_masks

        if not train_images_dir.exists() or not train_masks_dir.exists():
            print(f"❌ Training split not found!")
            print(f"   Images: {train_images_dir}")
            print(f"   Masks: {train_masks_dir}")
            return False

        if not val_images_dir.exists() or not val_masks_dir.exists():
            print(f"❌ Validation split not found!")
            print(f"   Images: {val_images_dir}")
            print(f"   Masks: {val_masks_dir}")
            return False

        print("\n📦 Loading dataset splits...")
        train_dataset = PolypSegmentationDataset(train_images_dir, train_masks_dir, Config.UNET_IMG_SIZE)
        val_dataset = PolypSegmentationDataset(val_images_dir, val_masks_dir, Config.UNET_IMG_SIZE)

        # DataLoaders (num_workers=4 for CPU utilization)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.UNET_BATCH,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.UNET_BATCH,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        print(f"✅ Dataset loaded:")
        print(f"   Training: {len(train_dataset)}")
        print(f"   Validation: {len(val_dataset)}")

        # Create model
        print(f"\n🔧 Building UNet++ model...")
        model = smp.UnetPlusPlus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            classes=1,
            activation='sigmoid'
        )
        model = model.to(Config.DEVICE)

        print(f"✅ Model created:")
        print(f"   Encoder: {Config.ENCODER}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        # AMP scaler for mixed-precision training (faster + larger batch on 16GB VRAM)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        print(f"\n🚀 Starting training for {Config.EPOCHS} epochs (AMP={'enabled' if scaler else 'disabled'})...")

        best_dice = 0
        start_epoch = 0
        checkpoint_dir = Config.SEGMENTATION_OUTPUT / 'unetpp_checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_checkpoint = checkpoint_dir / 'last.pth'

        # Resume from last checkpoint if it exists
        if last_checkpoint.exists():
            print(f"   ✅ Found checkpoint at {last_checkpoint}")
            print(f"   🔄 Resuming training from checkpoint...")
            try:
                checkpoint = torch.load(str(last_checkpoint), map_location=Config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_dice = checkpoint.get('best_dice', 0)
                print(f"   📍 Resuming from epoch {start_epoch+1}, best Dice: {best_dice:.4f}")
            except Exception as e:
                print(f"   ⚠️  Failed to load checkpoint: {e}")
                print(f"   🔄 Starting fresh...")
                start_epoch = 0
                best_dice = 0

        for epoch in range(start_epoch, Config.EPOCHS):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{Config.EPOCHS}")
            print(f"{'='*80}")

            # Train
            train_loss, train_dice, train_iou, train_acc = train_unet_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE, scaler=scaler
            )

            # Validate
            val_loss, val_dice, val_iou, val_acc = validate_unet_epoch(
                model, val_loader, criterion, Config.DEVICE
            )

            # Update scheduler
            scheduler.step(val_dice)

            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train - Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f} | Acc: {train_acc:.4f}")
            print(f"   Val   - Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Acc: {val_acc:.4f}")

            # Save checkpoint at every epoch (for resuming if interrupted)
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
            }
            torch.save(checkpoint_data, str(last_checkpoint))

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_path = Config.SEGMENTATION_OUTPUT / 'unetpp_best.pth'
                torch.save(checkpoint_data, str(best_model_path))
                print(f"   💾 Best model saved (Dice: {best_dice:.4f})")

            # Memory cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save final model
        final_model_path = Config.SEGMENTATION_OUTPUT / 'unetpp_final.pth'
        torch.save(model.state_dict(), str(final_model_path))

        print(f"\n✅ UNet++ Training Complete!")
        print(f"   Best Validation Dice: {best_dice:.4f}")

        return True

    except Exception as e:
        print(f"\n⚠️  UNet++ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# PART 2: ROI EXTRACTION - CROP TO CONCEPT
# ==========================================
def extract_rois_from_masks():
    """Extract 256x256 ROI crops from images using ground truth masks"""
    print("\n" + "=" * 80)
    print(" " * 25 + "EXTRACTING ROI CROPS (256x256)")
    print("=" * 80)

    phase1_dataset = build_phase1_combined_dataset(Config.THESIS_ROOT)

    # Create output directories
    cropped_images_dir = Config.CROPPED_OUTPUT / 'images'
    cropped_masks_dir = Config.CROPPED_OUTPUT / 'masks'
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    cropped_masks_dir.mkdir(parents=True, exist_ok=True)

    split_layout = [
        ('train', phase1_dataset.train_images, phase1_dataset.train_masks),
        ('val', phase1_dataset.val_images, phase1_dataset.val_masks),
    ]

    print(f"\n🔍 Processing dataset splits...")

    successful_crops = 0
    skipped_no_polyp = 0

    # Metadata dictionary to store original relative areas
    metadata = {}

    for split_name, images_dir, masks_dir in split_layout:
        mask_files = sorted([f for f in masks_dir.glob('*') if f.suffix.lower() in IMAGE_EXTENSIONS])
        print(f"   • {split_name}: {len(mask_files)} masks")

        for mask_file in tqdm(mask_files, desc=f"Extracting ROIs ({split_name})"):
            try:
                # Load mask - force grayscale
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                # Ensure mask is 2D (grayscale) - if still 3 channels, take first channel
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]  # Take first channel instead of color conversion

                # Get original frame dimensions
                original_h, original_w = mask.shape[:2]
                original_area = original_h * original_w

                # Find polyp bounding box
                _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    skipped_no_polyp += 1
                    continue

                # Get largest contour (main polyp)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # ⭐ METADATA PRESERVATION: Calculate original relative area BEFORE cropping
                box_area = w * h
                original_relative_area = box_area / original_area if original_area > 0 else 0.0

                # Add 20px buffer
                x1 = max(0, x - Config.ROI_BUFFER)
                y1 = max(0, y - Config.ROI_BUFFER)
                x2 = min(mask.shape[1], x + w + Config.ROI_BUFFER)
                y2 = min(mask.shape[0], y + h + Config.ROI_BUFFER)

                # Load corresponding image
                image_file = find_first_matching_file(images_dir, mask_file.stem, IMAGE_EXTENSIONS)
                if image_file is None:
                    continue

                image = cv2.imread(str(image_file))
                if image is None:
                    continue

                # Crop image and mask
                cropped_image = image[y1:y2, x1:x2]
                cropped_mask = mask[y1:y2, x1:x2]

                # Handle edge case: if crop is at boundary, pad with black
                crop_h, crop_w = cropped_image.shape[:2]
                if crop_h == 0 or crop_w == 0:
                    continue

                # Resize to 256x256 using INTER_CUBIC for texture integrity
                resized_image = cv2.resize(cropped_image, (Config.ROI_SIZE, Config.ROI_SIZE),
                                           interpolation=cv2.INTER_CUBIC)
                resized_mask = cv2.resize(cropped_mask, (Config.ROI_SIZE, Config.ROI_SIZE),
                                         interpolation=cv2.INTER_NEAREST)

                # Save cropped ROIs
                output_image_path = cropped_images_dir / mask_file.name
                output_mask_path = cropped_masks_dir / mask_file.name

                cv2.imwrite(str(output_image_path), resized_image)
                cv2.imwrite(str(output_mask_path), resized_mask)

                # ⭐ METADATA PRESERVATION: Save original relative area
                metadata[mask_file.stem] = {
                    'split': split_name,
                    'original_relative_area': float(original_relative_area),
                    'original_width': original_w,
                    'original_height': original_h,
                    'box_width': w,
                    'box_height': h
                }

                successful_crops += 1

            except Exception as e:
                print(f"\nError processing {mask_file.name}: {e}")
                continue

    # ⭐ METADATA PRESERVATION: Save metadata JSON
    import json
    metadata_path = Config.CROPPED_OUTPUT / 'roi_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ ROI Extraction Complete:")
    print(f"   Successful crops: {successful_crops}")
    print(f"   Skipped (no polyp): {skipped_no_polyp}")
    print(f"   Output: {Config.CROPPED_OUTPUT}")
    print(f"   Metadata: {metadata_path}")

    return successful_crops > 0

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

        results = {}

        print("\n" + "=" * 80)
        print(" " * 20 + "STARTING UNET++ TRAINING")
        print("=" * 80)

        results['UNet++'] = train_unet()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract ROI crops
        print("\n" + "=" * 80)
        print(" " * 20 + "EXTRACTING 256x256 ROI CROPS")
        print("=" * 80)
        results['ROI_Extraction'] = extract_rois_from_masks()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Summary
        print("\n" + "=" * 80)
        print(" " * 25 + "PHASE 1C COMPLETE!")
        print("=" * 80)
        print("\n📊 Results:")
        for model_name, success in results.items():
            status = "✅ Success" if success else "⚠️  Skipped/Failed"
            print(f"   {model_name:15s}: {status}")

        print(f"\n📁 Segmentation model saved to: {Config.SEGMENTATION_OUTPUT}")
        print(f"\n📁 Cropped ROIs saved to: {Config.CROPPED_OUTPUT}")
        print("\n➡️  Next: python phase2_ssl/ssl_training.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
