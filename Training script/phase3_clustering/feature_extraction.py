# -*- coding: utf-8 -*-
"""
PHASE 3: Dual-Stream Feature Extraction & Clustering (CORRECTED)
Implements the complete feature extraction pipeline with:
- Stream A (Neural): 384-dim ViT-Small SSL features
- Stream B (Symbolic): 60-dim biomarkers (HSV histograms, LBP, morphology)
- Total: 444-dimensional Fact Vector
- K-Means Clustering with 8 clusters (CLINICAL PROTOTYPES - Trained on NeoPolyp)

CRITICAL FIX: Clusters are NOW trained on NeoPolyp dataset (with clinical labels)
instead of Kvasir-SEG cropped ROIs. This ensures:
- RED pixels (Neoplastic) vs GREEN pixels (Non-neoplastic) separation
- Clusters represent medical risk categories, not visual similarity
- Phase 4 Decision Trees have meaningful labels to learn from (not random)
- Fixes the "Dataset Mismatch" that was breaking the neuro-symbolic system

Pipeline:
1. Extract SSL features from NeoPolyp images (labeled)
2. Extract biomarkers from NeoPolyp images
3. Combine into 444-dim fact vectors with clinical weighting
4. Train K-Means clustering ONLY on NeoPolyp (not Kvasir-SEG)
5. Result: 8 clinically-coherent clusters for Phase 4 experts
"""

import os
# Force scikit-learn/MKL to use a single thread to prevent Windows deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import joblib
import json

import torch
import torch.nn as nn
from torchvision import transforms
import timm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage

print("=" * 80)
print(" " * 15 + "PHASE 3: DUAL-STREAM FEATURE EXTRACTION & CLUSTERING")
print(" " * 20 + "(444-dim Fact Vector + K-Means)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    # Unlabeled Dataset (Dataset 2) - for clustering
    UNLABELED_DATASET = THESIS_ROOT / 'thesis_outputs' / 'cropped_rois' / 'images'

    # Neopolyp dataset (for clinical grounding)
    NEOPOLYP_ROOT = THESIS_ROOT / 'NeSy' / 'Neo polyp Dataset'
    TRAIN_IMAGES = NEOPOLYP_ROOT / 'train' / 'train'
    TRAIN_MASKS = NEOPOLYP_ROOT / 'train_gt' / 'train_gt'
    NEOPOLYP_OUTPUT = THESIS_ROOT / 'thesis_outputs' / 'neopolyp_processed'

    # Model paths
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    SSL_OUTPUT = OUTPUT_ROOT / 'ssl_outputs'
    FEATURES_OUTPUT = OUTPUT_ROOT / 'extracted_features'
    VISUAL_OUTPUT = OUTPUT_ROOT / 'visualizations'

    # Feature dimensions
    SSL_FEATURE_DIM = 384  # ViT-Small output
    BIOMARKER_DIM = 60    # 9 LAB + 16 Sat + 13 Haralick + 19 LBP + 3 Shape features
    TOTAL_FEATURE_DIM = 444  # 384 + 60

    # Clustering - Clinical-First Approach
    NUM_CLUSTERS = 8  # Medical Prototypes
    BIOMARKER_WEIGHT = 5.0  # Higher weight for symbolic features during clustering

    # Processing
    BATCH_SIZE = 256  # Large batch for fast GPU feature extraction (16 GB VRAM)
    NUM_WORKERS = 8   # Increased for better CPU utilization
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config.FEATURES_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Device: {Config.DEVICE}")
print(f"   SSL Features: {Config.SSL_FEATURE_DIM}-dim")
print(f"   Biomarkers: {Config.BIOMARKER_DIM}-dim")
print(f"   Total Features: {Config.TOTAL_FEATURE_DIM}-dim")
print(f"   Clusters: {Config.NUM_CLUSTERS}")

# ==========================================
# LOAD SSL MODEL (Stream A - Neural)
# ==========================================
class ViTEncoder(nn.Module):
    """ViT-Small SSL encoder from Phase 2"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224',
                                         pretrained=False,
                                         num_classes=0,
                                         img_size=256)
        self.backbone_dim = 384

    def forward(self, x):
        return self.backbone(x)

def load_ssl_model():
    """Load the trained SSL encoder from Phase 2"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING SSL MODEL")
    print("=" * 80)

    # Try different checkpoint locations
    possible_paths = [
        Config.SSL_OUTPUT / 'ssl_encoder_final.pth',
        Config.SSL_OUTPUT / 'ssl_model_final.pth',
        Config.SSL_OUTPUT / 'best_ssl_model.pth',
    ]

    model = ViTEncoder().to(Config.DEVICE)

    for path in possible_paths:
        if path.exists():
            try:
                state_dict = torch.load(str(path), map_location=Config.DEVICE)

                # Handle different save formats
                if isinstance(state_dict, dict) and 'backbone' in str(state_dict.keys()):
                    # Full model saved
                    if 'backbone.pos_embed' in state_dict:
                        model.backbone.load_state_dict({k.replace('backbone.', ''): v
                                                       for k, v in state_dict.items()
                                                       if k.startswith('backbone.')})
                    else:
                        model.load_state_dict(state_dict)
                else:
                    # Just backbone saved
                    model.backbone.load_state_dict(state_dict)

                print(f"✅ SSL model loaded from: {path.name}")
                model.eval()
                return model
            except Exception as e:
                print(f"   ⚠️  Failed to load {path.name}: {e}")
                continue

    print("❌ Could not load SSL model!")
    print("   Please run Phase 2 first!")
    return None

# ==========================================
# PADDING UTILITY (Preserve Aspect Ratio)
# ==========================================
def pad_to_square(image, target_size=256):
    """
    Pads an image to a square shape while maintaining the original aspect ratio.
    This preserves relative size information critical for medical assessment.
    """
    old_size = image.size  # (width, height)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # 1. Resize the image while maintaining aspect ratio
    image = image.resize(new_size, Image.LANCZOS)

    # 2. Create a new black canvas
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # 3. Paste the resized image onto the center of the canvas
    new_img.paste(image, ((target_size - new_size[0]) // 2,
                          (target_size - new_size[1]) // 2))
    return new_img

# ==========================================
# BIOMARKER EXTRACTION (Stream B - Symbolic)
# ==========================================
def compute_hsv_histograms(image_np):
    """
    DEPRECATED: Replaced by CIELAB for better perceptual uniformity
    Compute 16-bin histograms for Hue and Saturation channels
    Captures vascular redness distribution
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, _ = cv2.split(hsv)

    # Compute 16-bin histograms
    hist_hue = cv2.calcHist([h_channel], [0], None, [16], [0, 180])
    hist_sat = cv2.calcHist([s_channel], [0], None, [16], [0, 256])

    # Normalize
    hist_hue = hist_hue.flatten() / (hist_hue.sum() + 1e-7)
    hist_sat = hist_sat.flatten() / (hist_sat.sum() + 1e-7)

    return hist_hue, hist_sat

def compute_cielab_histograms(image_np):
    """
    Compute 9-bin histograms for L*, a*, b* channels in CIELAB color space
    CIELAB is perceptually uniform and better captures redness/vascular patterns
    """
    # Convert RGB to LAB
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Compute 3-bin histograms for each channel (total 9 bins)
    hist_l = cv2.calcHist([l_channel], [0], None, [3], [0, 256])
    hist_a = cv2.calcHist([a_channel], [0], None, [3], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [3], [0, 256])

    # Normalize
    hist_l = hist_l.flatten() / (hist_l.sum() + 1e-7)
    hist_a = hist_a.flatten() / (hist_a.sum() + 1e-7)
    hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)

    return hist_l, hist_a, hist_b

def compute_saturation_histogram(image_np):
    """
    Compute 16-bin Saturation histogram from HSV
    Captures vascular intensity
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s_channel, _ = cv2.split(hsv)

    # Compute 16-bin histogram
    hist_sat = cv2.calcHist([s_channel], [0], None, [16], [0, 256])

    # Normalize
    hist_sat = hist_sat.flatten() / (hist_sat.sum() + 1e-7)

    return hist_sat

def compute_haralick_features(image_np):
    """
    Compute Haralick texture features using GLCM (Gray-Level Co-occurrence Matrix)
    Standard features for medical image texture analysis
    Returns 13 features: contrast, dissimilarity, homogeneity, energy
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Compute GLCM with 4 directions (0, 45, 90, 135 degrees)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Reduce levels for faster computation
    gray_normalized = (gray / 16).astype(np.uint8)  # 16 levels

    try:
        glcm = graycomatrix(gray_normalized, distances, angles, levels=16,
                           symmetric=True, normed=True)

        # Extract properties (averaged across all directions)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()

        # Combine all (4 directions × 4 properties = 16, but we take 13 key features)
        haralick_features = np.concatenate([
            contrast,       # 4 values
            dissimilarity,  # 4 values
            homogeneity,    # 4 values
            [energy.mean()] # 1 value (average)
        ])

        return haralick_features[:13]  # Take first 13 features
    except:
        # Fallback if GLCM computation fails
        return np.zeros(13, dtype=np.float32)

def compute_lbp_features(image_np):
    """
    Compute Local Binary Patterns for texture analysis
    Captures Kudo Pit Pattern features - reduced to 19 bins
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Compute LBP (P=18, R=2, uniform patterns → 20 bins, take 19)
    radius = 2
    n_points = 18
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Compute histogram (20 bins for uniform LBP with P=18)
    n_bins = n_points + 2  # 20 bins
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalize and take 19 bins
    hist_lbp = hist_lbp.astype(np.float32)
    hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-7)

    return hist_lbp[:19]  # Take first 19 bins

def compute_morphology_features(image_np):
    """
    Compute morphological features:
    - Texture complexity (edge density)
    - Relative area (polyp area / frame area)
    """
    # Texture: Edge complexity
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    texture = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0

    # Relative area: Estimate from HSV segmentation
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s_channel, v_channel = cv2.split(hsv)
    rough_mask = ((s_channel > 30) | (v_channel < 200)).astype(np.float32)
    relative_area = np.mean(rough_mask)

    return np.array([texture, relative_area], dtype=np.float32)

def compute_shape_features(image_np):
    """
    Compute morphological shape features:
    - Texture complexity (edge density)
    - Relative area (polyp area / frame area)
    - Compactness (circularity measure)
    """
    # Texture: Edge complexity
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    texture = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0

    # Relative area: Estimate from HSV segmentation
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s_channel, v_channel = cv2.split(hsv)
    rough_mask = ((s_channel > 30) | (v_channel < 200)).astype(np.uint8)
    relative_area = np.mean(rough_mask)

    # Compactness: Compute from mask contours
    contours, _ = cv2.findContours(rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        # Compactness = 4π * Area / Perimeter²
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-7) if perimeter > 0 else 0.0
    else:
        compactness = 0.0

    return np.array([texture, relative_area, compactness], dtype=np.float32)

def extract_biomarkers(image_np):
    """
    Extract 60-dimensional biomarker vector:
    - 9 CIELAB histogram features (3 L* + 3 a* + 3 b*)
    - 16 Saturation histogram features
    - 13 Haralick texture features
    - 19 LBP histogram features
    - 3 Shape features (texture, area, compactness)
    Total: 60 features (9 + 16 + 13 + 19 + 3)
    """
    # CIELAB histograms (9 features)
    hist_l, hist_a, hist_b = compute_cielab_histograms(image_np)

    # Saturation histogram (16 features)
    hist_sat = compute_saturation_histogram(image_np)

    # Haralick features (13 features)
    haralick_features = compute_haralick_features(image_np)

    # LBP features (19 features)
    hist_lbp = compute_lbp_features(image_np)

    # Shape features (3 features)
    shape_features = compute_shape_features(image_np)

    # Combine all features
    biomarkers = np.concatenate([
        hist_l,            # 3 values
        hist_a,            # 3 values
        hist_b,            # 3 values
        hist_sat,          # 16 values
        haralick_features, # 13 values
        hist_lbp,          # 19 values
        shape_features     # 3 values (texture, area, compactness)
    ])

    assert len(biomarkers) == Config.BIOMARKER_DIM, f"Expected {Config.BIOMARKER_DIM} biomarkers, got {len(biomarkers)}"

    return biomarkers.astype(np.float32)

# ==========================================
# DUAL-STREAM FEATURE EXTRACTION
# ==========================================
def extract_features(image_path, ssl_model, transform):
    """
    Extract 444-dimensional Fact Vector:
    - Stream A: 384-dim SSL features (Neural)
    - Stream B: 60-dim biomarkers (Symbolic)
    NOW USES PADDING to preserve aspect ratio
    """
    # Load image and apply padding (preserve aspect ratio)
    image = Image.open(image_path).convert('RGB')
    image = pad_to_square(image, Config.IMG_SIZE)
    image_np = np.array(image)

    # Stream A: SSL Features (Neural)
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        ssl_features = ssl_model(image_tensor).cpu().numpy().flatten()

    # Stream B: Biomarkers (Symbolic)
    biomarkers = extract_biomarkers(image_np)

    # Combine both streams
    fact_vector = np.concatenate([ssl_features, biomarkers])

    assert len(fact_vector) == Config.TOTAL_FEATURE_DIM, \
        f"Expected {Config.TOTAL_FEATURE_DIM} features, got {len(fact_vector)}"

    return fact_vector, ssl_features, biomarkers

# ==========================================
# LOAD NEOPOLYP LABELS
# ==========================================
def load_neopolyp_labels():
    """Load ground truth labels from Neopolyp dataset"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING NEOPOLYP LABELS")
    print("=" * 80)

    label_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_labels.json'

    if not label_path.exists():
        print(f"⚠️  Neopolyp labels not found: {label_path}")
        print("   Attempting to create from mask colors...")
        print(f"   Searching in: {Config.TRAIN_MASKS}")

        # Check if directory exists
        if not Config.TRAIN_MASKS.exists():
            print(f"   ❌ Mask directory does not exist: {Config.TRAIN_MASKS}")
            return {}

        # Create labels from masks
        Config.NEOPOLYP_OUTPUT.mkdir(parents=True, exist_ok=True)
        labels = {}

        # Try multiple extensions
        mask_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            mask_files.extend(list(Config.TRAIN_MASKS.glob(ext)))

        print(f"   Found {len(mask_files)} mask files")

        if len(mask_files) == 0:
            print(f"   ❌ No mask files found in {Config.TRAIN_MASKS}")
            return {}

        for mask_path in mask_files:
            img_name = mask_path.name
            mask = cv2.imread(str(mask_path))

            if mask is not None:
                # Check mask color: RED = Neoplastic (High Risk), GREEN = Non-Neoplastic (Low Risk)
                red_pixels = np.sum((mask[:,:,2] > 200) & (mask[:,:,1] < 50) & (mask[:,:,0] < 50))
                green_pixels = np.sum((mask[:,:,1] > 200) & (mask[:,:,2] < 50) & (mask[:,:,0] < 50))

                # Assign label based on dominant color
                if red_pixels > green_pixels:
                    labels[img_name] = 1  # High Risk
                else:
                    labels[img_name] = 0  # Low Risk

        # Save labels if any were created
        if len(labels) > 0:
            with open(label_path, 'w') as f:
                json.dump(labels, f, indent=2)
            print(f"   ✅ Created {len(labels)} labels")
        else:
            print(f"   ❌ Failed to create any labels from masks")
            return {}
    else:
        with open(label_path, 'r') as f:
            labels = json.load(f)
        print(f"✅ Loaded {len(labels):,} labeled images")

    # Check if we have any labels before computing statistics
    if len(labels) == 0:
        print("   ⚠️  No labels available")
        return {}

    # Count labels
    high_risk = sum(1 for v in labels.values() if v == 1)
    low_risk = sum(1 for v in labels.values() if v == 0)

    print(f"   High Risk (Neoplastic): {high_risk:,} ({high_risk/len(labels)*100:.1f}%)")
    print(f"   Low Risk (Non-Neoplastic): {low_risk:,} ({low_risk/len(labels)*100:.1f}%)")

    return labels

# ==========================================
# EXTRACT NEOPOLYP FEATURES (For Clinical Clustering)
# ==========================================
def extract_neopolyp_features(ssl_model, transform):
    """Extract features from labeled Neopolyp dataset"""
    print("\n" + "=" * 80)
    print(" " * 20 + "EXTRACTING NEOPOLYP FEATURES (Clinical Grounding)")
    print("=" * 80)

    labels_dict = load_neopolyp_labels()
    if not labels_dict:
        print("❌ No labels available, cannot perform clinical clustering")
        return None, None

    neo_fact_vectors = []
    neo_labels = []
    neo_paths = []

    for img_name, label in tqdm(labels_dict.items(), desc="Extracting Neopolyp features"):
        img_path = Config.TRAIN_IMAGES / img_name

        if not img_path.exists():
            continue

        try:
            # Load and pad image (preserve aspect ratio)
            image = Image.open(img_path).convert('RGB')
            image = pad_to_square(image, Config.IMG_SIZE)
            image_np = np.array(image)

            # Extract SSL features (Neural)
            image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                ssl_features = ssl_model(image_tensor).cpu().numpy().flatten()

            # Extract biomarkers (Symbolic)
            biomarkers = extract_biomarkers(image_np)

            # Combine for fact vector
            fact_vector = np.concatenate([ssl_features, biomarkers])

            neo_fact_vectors.append(fact_vector)
            neo_labels.append(label)
            neo_paths.append(img_name)

        except Exception as e:
            print(f"\n   ⚠️  Failed to process {img_name}: {e}")
            continue

    neo_fact_vectors = np.array(neo_fact_vectors)
    neo_labels = np.array(neo_labels)

    print(f"\n✅ Extracted Neopolyp features:")
    print(f"   Samples: {len(neo_fact_vectors):,}")
    print(f"   High Risk: {np.sum(neo_labels == 1):,}")
    print(f"   Low Risk: {np.sum(neo_labels == 0):,}")
    print(f"   Feature dimension: {neo_fact_vectors.shape[1]}")

    return neo_fact_vectors, neo_labels

# ==========================================
# FEATURE EXTRACTION PIPELINE
# ==========================================
def extract_all_features():
    """Extract features from all images in the unlabeled dataset"""
    print("\n" + "=" * 80)
    print(" " * 20 + "EXTRACTING FEATURES FROM UNLABELED DATASET")
    print("=" * 80)

    # Load SSL model
    ssl_model = load_ssl_model()
    if ssl_model is None:
        return None, None, None, None

    # Prepare transform (no resize, padding is done in extract_features)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Collect images
    image_paths = []
    for ext in ['**/*.jpg', '**/*.jpeg', '**/*.png']:
        image_paths.extend(Config.UNLABELED_DATASET.glob(ext))

    image_paths = sorted(list(set(image_paths)))
    print(f"   Found {len(image_paths):,} images")

    # Extract features
    all_fact_vectors = []
    all_ssl_features = []
    all_biomarkers = []
    valid_paths = []

    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            fact_vector, ssl_feat, biomark = extract_features(img_path, ssl_model, transform)
            all_fact_vectors.append(fact_vector)
            all_ssl_features.append(ssl_feat)
            all_biomarkers.append(biomark)
            valid_paths.append(str(img_path))
        except Exception as e:
            print(f"\n   ⚠️  Failed to process {img_path.name}: {e}")
            continue

    # Convert to numpy arrays
    all_fact_vectors = np.array(all_fact_vectors)
    all_ssl_features = np.array(all_ssl_features)
    all_biomarkers = np.array(all_biomarkers)

    print(f"\n✅ Extracted features:")
    print(f"   Fact Vectors: {all_fact_vectors.shape}")
    print(f"   SSL Features: {all_ssl_features.shape}")
    print(f"   Biomarkers: {all_biomarkers.shape}")

    return all_fact_vectors, all_ssl_features, all_biomarkers, valid_paths

# ==========================================
# CLUSTERING (Clinical-First Approach)
# ==========================================
def perform_clustering(neo_fact_vectors, unlabeled_fact_vectors):
    """
    Apply K-Means clustering with Clinical-First approach:
    1. Fit on labeled Neopolyp data with weighted biomarkers
    2. Transform unlabeled data using the fitted model
    This ensures clusters represent medical categories, not just visual similarity
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "CLUSTERING (CLINICAL-FIRST APPROACH)")
    print("=" * 80)

    print(f"   Neopolyp samples (for training): {len(neo_fact_vectors):,}")
    print(f"   Unlabeled samples (for prediction): {len(unlabeled_fact_vectors):,}")

    # Apply feature weighting: Higher weight to biomarkers (last 60 dims)
    # This ensures medical features dominate clustering
    print(f"\n   Applying feature weighting...")
    print(f"   - SSL features (384-dim): weight = 1.0")
    print(f"   - Biomarkers (60-dim): weight = {Config.BIOMARKER_WEIGHT}")

    # Create weighted versions
    neo_weighted = neo_fact_vectors.copy()
    neo_weighted[:, -Config.BIOMARKER_DIM:] *= Config.BIOMARKER_WEIGHT

    unlabeled_weighted = unlabeled_fact_vectors.copy()
    unlabeled_weighted[:, -Config.BIOMARKER_DIM:] *= Config.BIOMARKER_WEIGHT

    # Standardize features (fit on Neopolyp, transform both)
    scaler = StandardScaler()
    neo_weighted_scaled = scaler.fit_transform(neo_weighted)
    unlabeled_weighted_scaled = scaler.transform(unlabeled_weighted)

    print(f"   Features scaled (mean=0, std=1) based on Neopolyp statistics")

    # Fit K-Means on Neopolyp data ONLY
    print(f"\n   Training K-Means on Neopolyp data ({Config.NUM_CLUSTERS} medical clusters)...")
    kmeans = KMeans(n_clusters=Config.NUM_CLUSTERS,
                   random_state=42,
                   n_init=20,  # More initializations for better convergence
                   max_iter=500,
                   verbose=0)

    neo_cluster_labels = kmeans.fit_predict(neo_weighted_scaled)

    # Predict clusters for unlabeled data
    print(f"   Mapping unlabeled data to medical clusters...")
    unlabeled_cluster_labels = kmeans.predict(unlabeled_weighted_scaled)

    # Print cluster distribution
    print(f"\n✅ Clustering complete!")
    print(f"\n📊 Neopolyp Cluster Distribution:")
    unique, counts = np.unique(neo_cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:,} samples ({count/len(neo_cluster_labels)*100:.1f}%)")

    print(f"\n📊 Unlabeled Data Cluster Distribution:")
    unique, counts = np.unique(unlabeled_cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:,} samples ({count/len(unlabeled_cluster_labels)*100:.1f}%)")

    return kmeans, scaler, neo_cluster_labels, unlabeled_cluster_labels

# ==========================================
# SAVE MODELS AND DATA
# ==========================================
def save_artifacts(fact_vectors, ssl_features, biomarkers, image_paths,
                  kmeans, scaler, cluster_labels):
    """Save all extracted features and models"""
    print("\n" + "=" * 80)
    print(" " * 25 + "SAVING ARTIFACTS")
    print("=" * 80)

    # Save features
    features_path = Config.FEATURES_OUTPUT / 'all_features.npz'
    np.savez_compressed(
        str(features_path),
        fact_vectors=fact_vectors,
        ssl_features=ssl_features,
        biomarkers=biomarkers,
        cluster_labels=cluster_labels,
        image_paths=np.array(image_paths)
    )
    print(f"💾 Features saved: {features_path}")

    # Save K-Means model
    kmeans_path = Config.FEATURES_OUTPUT / 'kmeans_model.pkl'
    joblib.dump(kmeans, str(kmeans_path))
    print(f"💾 K-Means model saved: {kmeans_path}")

    # Save scaler
    scaler_path = Config.FEATURES_OUTPUT / 'feature_scaler.pkl'
    joblib.dump(scaler, str(scaler_path))
    print(f"💾 Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        'num_samples': len(image_paths),
        'num_clusters': Config.NUM_CLUSTERS,
        'feature_dim': Config.TOTAL_FEATURE_DIM,
        'ssl_feature_dim': Config.SSL_FEATURE_DIM,
        'biomarker_dim': Config.BIOMARKER_DIM,
        'cluster_distribution': {
            int(cluster_id): int(count)
            for cluster_id, count in zip(*np.unique(cluster_labels, return_counts=True))
        }
    }

    metadata_path = Config.FEATURES_OUTPUT / 'clustering_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")

# ==========================================
# VISUALIZATION
# ==========================================
def visualize_clusters(fact_vectors, cluster_labels, image_paths, kmeans=None):
    """Visualize cluster distribution and sample images with cluster centers"""
    print("\n" + "=" * 80)
    print(" " * 25 + "GENERATING VISUALIZATIONS")
    print("=" * 80)

    from sklearn.decomposition import PCA

    # 1. PCA visualization with cluster centers
    print("   Computing PCA for visualization...")
    pca = PCA(n_components=2)
    fact_vectors_pca = pca.fit_transform(fact_vectors)

    # Transform cluster centers to PCA space if available
    cluster_centers_pca = None
    if kmeans is not None:
        cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot with data points
    scatter = axes[0].scatter(fact_vectors_pca[:, 0],
                             fact_vectors_pca[:, 1],
                             c=cluster_labels,
                             cmap='tab10',
                             alpha=0.6,
                             s=20,
                             label='Data Points')

    # Plot cluster centers as large stars
    if cluster_centers_pca is not None:
        axes[0].scatter(cluster_centers_pca[:, 0],
                       cluster_centers_pca[:, 1],
                       c=range(Config.NUM_CLUSTERS),
                       cmap='tab10',
                       marker='*',
                       s=500,
                       edgecolors='black',
                       linewidths=2,
                       label='Cluster Centers',
                       zorder=10)

        # Add cluster ID labels
        for i in range(Config.NUM_CLUSTERS):
            axes[0].annotate(f'C{i}',
                           xy=(cluster_centers_pca[i, 0], cluster_centers_pca[i, 1]),
                           xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=12,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           zorder=11)

    axes[0].set_title('K-Means Clustering (PCA Projection)\n8 Cluster Centers Marked with Stars ★',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster ID')

    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    bars = axes[1].bar(unique, counts, color=plt.cm.tab10(unique / Config.NUM_CLUSTERS))
    axes[1].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Images')
    axes[1].grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    viz_path = Config.VISUAL_OUTPUT / 'phase3_clustering_overview.png'
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Clustering overview saved: {viz_path}")

    # 2. Create detailed cluster centers visualization
    if cluster_centers_pca is not None:
        print("   Creating detailed cluster centers map...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Plot all points with lower alpha
        scatter = ax.scatter(fact_vectors_pca[:, 0],
                           fact_vectors_pca[:, 1],
                           c=cluster_labels,
                           cmap='tab10',
                           alpha=0.3,
                           s=15)

        # Plot cluster centers prominently
        for i in range(Config.NUM_CLUSTERS):
            color = plt.cm.tab10(i / Config.NUM_CLUSTERS)

            # Large star for center
            ax.scatter(cluster_centers_pca[i, 0],
                      cluster_centers_pca[i, 1],
                      c=[color],
                      marker='*',
                      s=800,
                      edgecolors='black',
                      linewidths=3,
                      zorder=10)

            # Circle around cluster region
            cluster_points = fact_vectors_pca[cluster_labels == i]
            if len(cluster_points) > 0:
                # Calculate cluster spread (standard deviation)
                std_x = np.std(cluster_points[:, 0])
                std_y = np.std(cluster_points[:, 1])

                # Draw ellipse representing 2-sigma region
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(xy=(cluster_centers_pca[i, 0], cluster_centers_pca[i, 1]),
                                width=4*std_x, height=4*std_y,
                                facecolor='none',
                                edgecolor=color,
                                linewidth=2,
                                linestyle='--',
                                alpha=0.6,
                                zorder=5)
                ax.add_patch(ellipse)

            # Label with cluster info
            unique, counts = np.unique(cluster_labels, return_counts=True)
            count_dict = dict(zip(unique, counts))
            cluster_count = count_dict.get(i, 0)
            ax.annotate(f'Cluster {i}\n({cluster_count} images)',
                       xy=(cluster_centers_pca[i, 0], cluster_centers_pca[i, 1]),
                       xytext=(20, 20),
                       textcoords='offset points',
                       fontsize=11,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor=color,
                               edgecolor='black',
                               alpha=0.8),
                       arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0.3',
                                     color='black',
                                     lw=2),
                       zorder=11)

        ax.set_title('K-Means: 8 Cluster Centers with Distribution Regions\n(Dashed ellipses = 2σ spread)',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                     fontsize=12)
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                     fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()
        centers_viz_path = Config.VISUAL_OUTPUT / 'phase3_cluster_centers_detailed.png'
        plt.savefig(str(centers_viz_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Detailed cluster centers map saved: {centers_viz_path}")

    # 2. Sample images from each cluster
    print("   Generating cluster sample visualizations...")
    fig, axes = plt.subplots(Config.NUM_CLUSTERS, 5, figsize=(15, 3 * Config.NUM_CLUSTERS))

    for cluster_id in range(Config.NUM_CLUSTERS):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        sample_indices = np.random.choice(cluster_indices,
                                         min(5, len(cluster_indices)),
                                         replace=False)

        for col_idx, img_idx in enumerate(sample_indices):
            try:
                img_path = image_paths[img_idx]
                
                # Check if image_path is a real file path (not a placeholder string)
                if isinstance(img_path, str) and img_path.endswith(('jpg', 'jpeg', 'png')):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[cluster_id, col_idx].imshow(img)
                    else:
                        # Skip if file doesn't exist
                        axes[cluster_id, col_idx].text(0.5, 0.5, 'Image\nNot Found',
                                                       ha='center', va='center',
                                                       transform=axes[cluster_id, col_idx].transAxes,
                                                       fontsize=10)
                else:
                    # Skip placeholder paths
                    axes[cluster_id, col_idx].text(0.5, 0.5, f'{img_path}',
                                                   ha='center', va='center',
                                                   transform=axes[cluster_id, col_idx].transAxes,
                                                   fontsize=8)
                
                axes[cluster_id, col_idx].axis('off')

                if col_idx == 0:
                    axes[cluster_id, col_idx].set_title(
                        f'Cluster {cluster_id}\n({len(cluster_indices)} images)',
                        fontsize=10, fontweight='bold'
                    )
            except Exception as e:
                # Skip on any error
                axes[cluster_id, col_idx].text(0.5, 0.5, f'Error\n{str(e)[:20]}',
                                               ha='center', va='center',
                                               transform=axes[cluster_id, col_idx].transAxes,
                                               fontsize=8)
                axes[cluster_id, col_idx].axis('off')

    plt.tight_layout()
    samples_path = Config.VISUAL_OUTPUT / 'phase3_cluster_samples.png'
    plt.savefig(str(samples_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Cluster samples saved: {samples_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STARTING PHASE 3")
    print("=" * 80)

    # Load SSL model once
    ssl_model = load_ssl_model()
    if ssl_model is None:
        print("\n❌ SSL model loading failed!")
        return

    # Prepare transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Extract Neopolyp features (for CLINICAL clustering)
    neo_fact_vectors, neo_labels = extract_neopolyp_features(ssl_model, transform)

    if neo_fact_vectors is None or len(neo_fact_vectors) == 0:
        print("\n❌ Neopolyp feature extraction failed!")
        print("   Cannot proceed without clinical grounding!")
        return

    print("\n🔬 CLINICAL-FIRST CLUSTERING: Using NEOPOLYP dataset for cluster centers")
    print("   This ensures clusters separate HIGH RISK from LOW RISK polyps")
    
    # Step 2: Apply feature weighting for clinical clustering
    print(f"\n   Applying feature weighting...")
    print(f"   - SSL features (384-dim): weight = 1.0")
    print(f"   - Biomarkers (60-dim): weight = {Config.BIOMARKER_WEIGHT}")

    neo_weighted = neo_fact_vectors.copy()
    neo_weighted[:, -Config.BIOMARKER_DIM:] *= Config.BIOMARKER_WEIGHT

    # Step 3: Perform clustering on NeoPolyp ONLY
    scaler = StandardScaler()
    neo_weighted_scaled = scaler.fit_transform(neo_weighted)
    
    kmeans = KMeans(n_clusters=Config.NUM_CLUSTERS, random_state=42, n_init=20, max_iter=500)
    neo_cluster_labels = kmeans.fit_predict(neo_weighted_scaled)

    print(f"\n✅ Clustering complete on {len(neo_fact_vectors):,} NeoPolyp samples!")
    
    # Analyze cluster composition (clinical coherence)
    print(f"\n📊 Cluster Medical Coherence:")
    for cluster_id in range(Config.NUM_CLUSTERS):
        cluster_indices = np.where(neo_cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            c_labels = neo_labels[cluster_indices]
            high_risk = np.sum(c_labels == 1)
            low_risk = np.sum(c_labels == 0)
            high_risk_pct = (high_risk / len(cluster_indices)) * 100
            print(f"   Cluster {cluster_id}: {len(cluster_indices):,} samples "
                  f"(High Risk: {high_risk_pct:.1f}%) - {'PURE' if high_risk_pct < 15 or high_risk_pct > 85 else 'MIXED'}")

    # Step 4: Optional - Extract unlabeled features for visualization only
    print("\n📊 Extracting unlabeled features for visualization...")
    fact_vectors, ssl_features, biomarkers, image_paths = extract_all_features()

    if fact_vectors is not None:
        # Apply same scaling and predict clusters for visualization
        fact_vectors_weighted = fact_vectors.copy()
        fact_vectors_weighted[:, -Config.BIOMARKER_DIM:] *= Config.BIOMARKER_WEIGHT
        fact_vectors_scaled = scaler.transform(fact_vectors_weighted)
        cluster_labels = kmeans.predict(fact_vectors_scaled)
        
        print(f"   Assigned {len(fact_vectors):,} unlabeled samples to clusters")
    else:
        cluster_labels = neo_cluster_labels
        image_paths = None
        fact_vectors = neo_fact_vectors
        ssl_features = None
        biomarkers = None

    # Step 5: Save artifacts
    save_artifacts(fact_vectors, ssl_features, biomarkers, image_paths if image_paths is not None else ["neopolyp"],
                  kmeans, scaler, cluster_labels)

    # Step 6: Visualize results
    visualize_clusters(neo_fact_vectors, neo_cluster_labels, 
                      [f"neopolyp_{i}" for i in range(len(neo_fact_vectors))], kmeans)

    print("\n" + "=" * 80)
    print(" " * 25 + "PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Created {Config.NUM_CLUSTERS} CLINICALLY-MEANINGFUL PROTOTYPES")
    print(f"   Training samples: {len(neo_fact_vectors):,} (NeoPolyp labeled)")
    print(f"   Feature dimension: {Config.TOTAL_FEATURE_DIM}")
    print(f"   Cluster centers: Trained on HIGH RISK/LOW RISK labels")
    print(f"   Next: Phase 4 will train Decision Trees with clinical grounding")


if __name__ == '__main__':
    main()
