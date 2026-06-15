# -*- coding: utf-8 -*-
"""
PHASE 2.5: Neopolyp Dataset Preparation
Validates and prepares the Neopolyp dataset with color-coded masks for Phase 4
RED pixels = Neoplastic (High Risk)
GREEN pixels = Non-Neoplastic (Low Risk)
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

print("=" * 80)
print(" " * 20 + "PHASE 2.5: NEOPOLYP DATASET PREPARATION")
print(" " * 15 + "(Validate Color-Coded Masks)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    NEOPOLYP_ROOT = THESIS_ROOT / 'NeSy' / 'Neo polyp Dataset'
    TRAIN_IMAGES = NEOPOLYP_ROOT / 'train' / 'train'
    TRAIN_MASKS = NEOPOLYP_ROOT / 'train_gt' / 'train_gt'

    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    NEOPOLYP_OUTPUT = OUTPUT_ROOT / 'neopolyp_processed'
    VISUAL_OUTPUT = OUTPUT_ROOT / 'visualizations'

    # Color thresholds for mask parsing (in RGB)
    # RED channel dominant = Neoplastic
    # GREEN channel dominant = Non-Neoplastic
    RED_THRESHOLD = 100  # Minimum red value
    GREEN_THRESHOLD = 100  # Minimum green value

Config.NEOPOLYP_OUTPUT.mkdir(parents=True, exist_ok=True)
Config.VISUAL_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Train Images: {Config.TRAIN_IMAGES}")
print(f"   Train Masks: {Config.TRAIN_MASKS}")
print(f"   Output: {Config.NEOPOLYP_OUTPUT}")

# ==========================================
# DATASET VALIDATION
# ==========================================
def validate_dataset():
    """Validate that images and masks are properly paired"""
    print("\n" + "=" * 80)
    print(" " * 25 + "DATASET VALIDATION")
    print("=" * 80)

    # Collect images and masks
    image_paths = sorted(list(Config.TRAIN_IMAGES.glob('*.jpeg')) +
                        list(Config.TRAIN_IMAGES.glob('*.jpg')) +
                        list(Config.TRAIN_IMAGES.glob('*.png')))

    mask_paths = sorted(list(Config.TRAIN_MASKS.glob('*.jpeg')) +
                       list(Config.TRAIN_MASKS.glob('*.jpg')) +
                       list(Config.TRAIN_MASKS.glob('*.png')))

    print(f"   Found {len(image_paths):,} images")
    print(f"   Found {len(mask_paths):,} masks")

    # Check for matching pairs
    image_names = {p.stem for p in image_paths}
    mask_names = {p.stem for p in mask_paths}

    missing_masks = image_names - mask_names
    missing_images = mask_names - image_names

    if missing_masks:
        print(f"   ⚠️  {len(missing_masks)} images without masks")
    if missing_images:
        print(f"   ⚠️  {len(missing_images)} masks without images")

    # Create matched pairs
    paired_data = []
    for img_path in image_paths:
        mask_path = Config.TRAIN_MASKS / img_path.name
        if mask_path.exists():
            paired_data.append((img_path, mask_path))

    print(f"✅ {len(paired_data):,} valid image-mask pairs")

    return paired_data

# ==========================================
# MASK PARSING
# ==========================================
def parse_mask_label(mask_path):
    """
    Parse color-coded mask to determine risk label
    Returns:
        - 1 if mask contains RED pixels (Neoplastic/High Risk)
        - 0 if mask contains only GREEN pixels (Non-Neoplastic/Low Risk)
        - 1 if both colors present (Safety Priority - default to High Risk)
    """
    mask = cv2.imread(str(mask_path))
    if mask is None:
        print(f"   ⚠️  Failed to load mask: {mask_path.name}")
        return None

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Extract channels
    r_channel = mask[:, :, 0]
    g_channel = mask[:, :, 1]
    b_channel = mask[:, :, 2]

    # Detect RED pixels (high R, low G, low B)
    red_mask = (r_channel > Config.RED_THRESHOLD) & \
               (g_channel < Config.RED_THRESHOLD) & \
               (b_channel < Config.RED_THRESHOLD)
    has_red = np.any(red_mask)

    # Detect GREEN pixels (low R, high G, low B)
    green_mask = (r_channel < Config.GREEN_THRESHOLD) & \
                 (g_channel > Config.GREEN_THRESHOLD) & \
                 (b_channel < Config.GREEN_THRESHOLD)
    has_green = np.any(green_mask)

    # Determine label
    if has_red:
        label = 1  # Neoplastic (High Risk)
        color_type = 'RED'
    elif has_green:
        label = 0  # Non-Neoplastic (Low Risk)
        color_type = 'GREEN'
    else:
        # No clear color detected - check which is more prominent
        avg_red = np.mean(r_channel)
        avg_green = np.mean(g_channel)
        if avg_red > avg_green:
            label = 1
            color_type = 'RED (inferred)'
        else:
            label = 0
            color_type = 'GREEN (inferred)'

    return {
        'label': label,
        'color_type': color_type,
        'has_red': has_red,
        'has_green': has_green,
        'red_pixels': int(np.sum(red_mask)),
        'green_pixels': int(np.sum(green_mask))
    }

# ==========================================
# DATASET PROCESSING
# ==========================================
def process_dataset(paired_data):
    """Process all image-mask pairs and extract labels"""
    print("\n" + "=" * 80)
    print(" " * 25 + "PROCESSING DATASET")
    print("=" * 80)

    processed_data = []
    label_counts = {0: 0, 1: 0}
    color_stats = {'RED': 0, 'GREEN': 0, 'BOTH': 0, 'UNCLEAR': 0}

    for img_path, mask_path in tqdm(paired_data, desc="Processing images"):
        mask_info = parse_mask_label(mask_path)

        if mask_info is None:
            continue

        # Store processed info
        processed_data.append({
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'image_name': img_path.name,
            'label': mask_info['label'],
            'color_type': mask_info['color_type'],
            'red_pixels': mask_info['red_pixels'],
            'green_pixels': mask_info['green_pixels']
        })

        # Update statistics
        label_counts[mask_info['label']] += 1

        if mask_info['has_red'] and mask_info['has_green']:
            color_stats['BOTH'] += 1
        elif mask_info['has_red']:
            color_stats['RED'] += 1
        elif mask_info['has_green']:
            color_stats['GREEN'] += 1
        else:
            color_stats['UNCLEAR'] += 1

    # Print statistics
    print(f"\n✅ Processed {len(processed_data):,} images")
    print(f"\n📊 Label Distribution:")
    print(f"   High Risk (Neoplastic):     {label_counts[1]:,} ({label_counts[1]/len(processed_data)*100:.1f}%)")
    print(f"   Low Risk (Non-Neoplastic):  {label_counts[0]:,} ({label_counts[0]/len(processed_data)*100:.1f}%)")

    print(f"\n🎨 Color Distribution:")
    print(f"   RED only:    {color_stats['RED']:,}")
    print(f"   GREEN only:  {color_stats['GREEN']:,}")
    print(f"   BOTH colors: {color_stats['BOTH']:,}")
    print(f"   UNCLEAR:     {color_stats['UNCLEAR']:,}")

    return processed_data

# ==========================================
# SAVE METADATA
# ==========================================
def save_metadata(processed_data):
    """Save processed metadata for Phase 4"""
    print("\n" + "=" * 80)
    print(" " * 25 + "SAVING METADATA")
    print("=" * 80)

    metadata_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_metadata.json'

    with open(metadata_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"💾 Metadata saved: {metadata_path}")

    # Also save a simple image-label mapping
    label_mapping = {item['image_name']: item['label'] for item in processed_data}
    label_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_labels.json'

    with open(label_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print(f"💾 Label mapping saved: {label_path}")

# ==========================================
# VISUALIZATION
# ==========================================
def visualize_samples(processed_data):
    """Visualize sample images with their masks and labels"""
    print("\n" + "=" * 80)
    print(" " * 25 + "GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Sample images from each category
    high_risk_samples = [item for item in processed_data if item['label'] == 1][:5]
    low_risk_samples = [item for item in processed_data if item['label'] == 0][:5]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Plot high risk samples
    for idx, item in enumerate(high_risk_samples):
        if idx >= 5:
            break
        img = cv2.imread(item['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(item['mask_path'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

        axes[0, idx].imshow(overlay)
        axes[0, idx].set_title(f"HIGH RISK\n{item['color_type']}", fontsize=10, color='red')
        axes[0, idx].axis('off')

    # Plot low risk samples
    for idx, item in enumerate(low_risk_samples):
        if idx >= 5:
            break
        img = cv2.imread(item['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(item['mask_path'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f"LOW RISK\n{item['color_type']}", fontsize=10, color='green')
        axes[1, idx].axis('off')

    plt.tight_layout()
    viz_path = Config.VISUAL_OUTPUT / 'neopolyp_samples.png'
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Visualization saved: {viz_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STARTING PHASE 2.5")
    print("=" * 80)

    # Step 1: Validate dataset
    paired_data = validate_dataset()

    if len(paired_data) == 0:
        print("\n❌ No valid image-mask pairs found!")
        return

    # Step 2: Process dataset
    processed_data = process_dataset(paired_data)

    # Step 3: Save metadata
    save_metadata(processed_data)

    # Step 4: Visualize samples
    visualize_samples(processed_data)

    print("\n" + "=" * 80)
    print(" " * 25 + "PHASE 2.5 COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Processed {len(processed_data):,} Neopolyp images")
    print(f"   Next: Run Phase 3 for feature extraction and clustering")

if __name__ == '__main__':
    main()
