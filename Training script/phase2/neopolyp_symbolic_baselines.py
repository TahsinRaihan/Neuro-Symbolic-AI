# -*- coding: utf-8 -*-
"""
PHASE 2.6: Neopolyp Symbolic Baselines
Analyzes NeoPolyp dataset to extract statistical baselines for high-risk (red) and low-risk (green) polyps.
Computes mean (μ) and standard deviation (σ) for 60-dimensional symbolic features.
Outputs neopolyp_ground_truth_baselines.json for symbolic reasoning calibration.
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
import json

print("=" * 80)
print(" " * 20 + "PHASE 2.6: NEOPOLYP SYMBOLIC BASELINES")
print(" " * 15 + "(Extract Statistical Baselines for Symbolic Reasoning)")
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

    # Color thresholds for mask parsing (in RGB)
    RED_THRESHOLD = 100  # Minimum red value
    GREEN_THRESHOLD = 100  # Minimum green value

# Create directories
Config.NEOPOLYP_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Train Images: {Config.TRAIN_IMAGES}")
print(f"   Train Masks: {Config.TRAIN_MASKS}")
print(f"   Output: {Config.NEOPOLYP_OUTPUT}")

# ==========================================
# IMPORT FEATURE EXTRACTION
# ==========================================
sys.path.insert(0, str(Path(__file__).parent / '..' / 'phase3_clustering'))
from feature_extraction import extract_biomarkers, pad_to_square

# Biomarker field names (in extraction order)
BIOMARKER_NAMES = (
    [f'LAB_L_bin{i}' for i in range(3)] +
    [f'LAB_a_bin{i}' for i in range(3)] +
    [f'LAB_b_bin{i}' for i in range(3)] +
    [f'Sat_bin{i}'   for i in range(16)] +
    [f'Haralick_{i}' for i in range(13)] +
    [f'LBP_bin{i}'   for i in range(19)] +
    ['Texture_Complexity', 'Relative_Area', 'Compactness']
)

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
    """
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return None

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Extract channels
    r_channel = mask[:, :, 0]
    g_channel = mask[:, :, 1]
    b_channel = mask[:, :, 2]

    # Detect RED pixels
    red_mask = (r_channel > Config.RED_THRESHOLD) & \
               (g_channel < Config.RED_THRESHOLD) & \
               (b_channel < Config.RED_THRESHOLD)
    has_red = np.any(red_mask)

    # Detect GREEN pixels
    green_mask = (r_channel < Config.GREEN_THRESHOLD) & \
                 (g_channel > Config.GREEN_THRESHOLD) & \
                 (b_channel < Config.GREEN_THRESHOLD)
    has_green = np.any(green_mask)

    # Determine label
    if has_red:
        label = 1  # High Risk
    elif has_green:
        label = 0  # Low Risk
    else:
        # Fallback: compare averages
        avg_red = np.mean(r_channel)
        avg_green = np.mean(g_channel)
        label = 1 if avg_red > avg_green else 0

    return label

# ==========================================
# FEATURE EXTRACTION AND STATISTICS
# ==========================================
def extract_symbolic_baselines(paired_data):
    """Extract biomarkers and compute statistical baselines"""
    print("\n" + "=" * 80)
    print(" " * 20 + "EXTRACTING SYMBOLIC BASELINES")
    print("=" * 80)

    high_risk_features = []
    low_risk_features = []

    for img_path, mask_path in tqdm(paired_data, desc="Processing images"):
        # Get label
        label = parse_mask_label(mask_path)
        if label is None:
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pad to square (required for biomarker extraction)
        image_pil = Image.fromarray(image)
        image_padded = pad_to_square(image_pil, 256)  # Use same size as in feature_extraction
        image_np = np.array(image_padded)

        # Extract biomarkers
        try:
            biomarkers = extract_biomarkers(image_np)
        except Exception as e:
            print(f"   ⚠️  Failed to extract features for {img_path.name}: {e}")
            continue

        # Collect features by risk group
        if label == 1:  # High Risk
            high_risk_features.append(biomarkers)
        else:  # Low Risk
            low_risk_features.append(biomarkers)

    print(f"\n📊 Feature Extraction Complete:")
    print(f"   High-Risk samples: {len(high_risk_features)}")
    print(f"   Low-Risk samples: {len(low_risk_features)}")

    if not high_risk_features or not low_risk_features:
        print("   ❌ Insufficient data for both risk groups")
        return None

    # Convert to numpy arrays
    high_risk_features = np.array(high_risk_features)
    low_risk_features = np.array(low_risk_features)

    # Compute statistics
    baselines = {
        'metadata': {
            'description': 'Statistical baselines for symbolic reasoning from NeoPolyp dataset',
            'high_risk_samples': len(high_risk_features),
            'low_risk_samples': len(low_risk_features),
            'feature_dimensions': len(BIOMARKER_NAMES),
            'feature_names': BIOMARKER_NAMES
        },
        'high_risk': {},
        'low_risk': {},
        'differences': {}
    }

    for i, name in enumerate(BIOMARKER_NAMES):
        # High Risk stats
        hr_mean = float(np.mean(high_risk_features[:, i]))
        hr_std = float(np.std(high_risk_features[:, i]))

        # Low Risk stats
        lr_mean = float(np.mean(low_risk_features[:, i]))
        lr_std = float(np.std(low_risk_features[:, i]))

        # Difference
        diff = hr_mean - lr_mean
        abs_diff = abs(diff)

        baselines['high_risk'][name] = {
            'mean': hr_mean,
            'std': hr_std
        }
        baselines['low_risk'][name] = {
            'mean': lr_mean,
            'std': lr_std
        }
        baselines['differences'][name] = {
            'mean_difference': diff,
            'absolute_difference': abs_diff,
            'effect_size': abs_diff / np.sqrt((hr_std**2 + lr_std**2) / 2) if (hr_std + lr_std) > 0 else 0
        }

    return baselines

# ==========================================
# SAVE BASELINES
# ==========================================
def save_baselines(baselines):
    """Save baselines to JSON file"""
    output_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_ground_truth_baselines.json'

    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2)

    print(f"\n💾 Baselines saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size:,} bytes")

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"\n🚀 Starting NeoPolyp Symbolic Baselines Extraction...")

    # Validate dataset
    paired_data = validate_dataset()
    if not paired_data:
        print("❌ No valid data pairs found")
        return

    # Extract baselines
    baselines = extract_symbolic_baselines(paired_data)
    if baselines is None:
        print("❌ Failed to extract baselines")
        return

    # Save results
    save_baselines(baselines)

    # Summary
    print("\n" + "=" * 80)
    print(" " * 25 + "EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"High-Risk samples: {baselines['metadata']['high_risk_samples']}")
    print(f"Low-Risk samples: {baselines['metadata']['low_risk_samples']}")
    print(f"Features analyzed: {baselines['metadata']['feature_dimensions']}")

    # Show top differences
    differences = sorted(baselines['differences'].items(),
                        key=lambda x: x[1]['absolute_difference'], reverse=True)

if __name__ == "__main__":
    main()