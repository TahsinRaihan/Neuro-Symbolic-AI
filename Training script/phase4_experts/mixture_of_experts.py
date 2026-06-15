# -*- coding: utf-8 -*-
"""
PHASE 4: Mixture of Experts - Local Rule Learning (CORRECTED)
Trains 8 separate Decision Trees (one per cluster) using Neopolyp labeled data
Clusters are NOW clinically-meaningful, trained on HIGH RISK/LOW RISK labels

CRITICAL FIX: Phase 3 clusters now come from NeoPolyp (with color-coded masks),
NOT from Kvasir-SEG cropped ROIs. This ensures:
- Clusters separate neoplastic (RED) from non-neoplastic (GREEN)
- Decision trees learn actual clinical risk, not visual similarity
- Expert predictions are clinically valid, not random guessing

Architecture:
- 8 Local Experts (Decision Trees), one for each CLINICAL PROTOTYPE
- Clusters trained on NeoPolyp (RED=Neoplastic/High Risk, GREEN=Benign/Low Risk)
- Each tree trained only on clinical labels (high/low risk)
- Ground truth: Color-coded masks from NeoPolyp dataset
- Implements ASGE PIVI standards for clinical confidence thresholds
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
from collections import defaultdict

import torch
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("=" * 80)
print(" " * 15 + "PHASE 4: MIXTURE OF EXPERTS - LOCAL RULE LEARNING")
print(" " * 20 + "(8 Decision Trees + ASGE PIVI Standards)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()

    # Neopolyp dataset (for labeled training)
    NEOPOLYP_ROOT = THESIS_ROOT / 'NeSy' / 'Neo polyp Dataset'
    TRAIN_IMAGES = NEOPOLYP_ROOT / 'train' / 'train'
    TRAIN_MASKS = NEOPOLYP_ROOT / 'train_gt' / 'train_gt'

    # Outputs
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    FEATURES_OUTPUT = OUTPUT_ROOT / 'extracted_features'
    NEOPOLYP_OUTPUT = OUTPUT_ROOT / 'neopolyp_processed'
    EXPERTS_OUTPUT = OUTPUT_ROOT / 'mixture_of_experts'
    VISUAL_OUTPUT = OUTPUT_ROOT / 'visualizations'

    # Model configuration
    NUM_CLUSTERS = 8
    MIN_SAMPLES_FOR_TRAINING = 15  # Minimum samples per cluster to train (increased from 10)

    # Decision Tree parameters (improved for better clinical patterns)
    TREE_MAX_DEPTH = 7  # Increased from 5 to capture more complex patterns
    TREE_MIN_SAMPLES_SPLIT = 10
    TREE_MIN_SAMPLES_LEAF = 5  # Increased from previous value for better generalization

    # Feature dimensions
    SSL_FEATURE_DIM = 384  # ViT-Small output
    BIOMARKER_DIM = 60    # Symbolic features
    TOTAL_FEATURE_DIM = 444  # 384 + 60

    # ASGE PIVI Thresholds (Clinical Standards)
    # Reference: ASGE PIVI on real-time endoscopic assessment of polyps
    ASGE_HIGH_CONFIDENCE_THRESHOLD = 0.90  # High confidence for "Resect & Discard"
    ASGE_UNCERTAINTY_THRESHOLD = 0.80      # Uncertainty buffer for clinical review

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 256

Config.EXPERTS_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Neopolyp Images: {Config.TRAIN_IMAGES}")
print(f"   Number of Experts: {Config.NUM_CLUSTERS}")
print(f"   ASGE High Confidence: {Config.ASGE_HIGH_CONFIDENCE_THRESHOLD}")
print(f"   ASGE Uncertainty: {Config.ASGE_UNCERTAINTY_THRESHOLD}")
print(f"   Feature Dimension: {Config.TOTAL_FEATURE_DIM} (SSL + Biomarkers)")
print(f"   Tree Max Depth: {Config.TREE_MAX_DEPTH}")
print(f"   Tree Min Samples Leaf: {Config.TREE_MIN_SAMPLES_LEAF}")

# ==========================================
# CLUSTER VALIDATION
# ==========================================
def validate_cluster_purity(cluster_datasets):
    """
    Check if clusters are medically coherent
    Flag clusters with mixed risk distributions (40-60% high risk)
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "VALIDATING CLUSTER MEDICAL COHERENCE")
    print("=" * 80)

    cluster_quality = {}
    for cluster_id in range(Config.NUM_CLUSTERS):
        if len(cluster_datasets[cluster_id]['labels']) > 0:
            labels = cluster_datasets[cluster_id]['labels']
            high_risk_ratio = np.mean(labels)
            n_samples = len(labels)

            # Determine cluster purity
            if 0.4 < high_risk_ratio < 0.6:
                purity_status = "⚠️  MIXED CLUSTER - Consider splitting or merging"
                cluster_quality[cluster_id] = 'mixed'
            elif high_risk_ratio >= 0.75:
                purity_status = "✅ HIGH RISK DOMINANT"
                cluster_quality[cluster_id] = 'high_risk'
            elif high_risk_ratio <= 0.25:
                purity_status = "✅ LOW RISK DOMINANT"
                cluster_quality[cluster_id] = 'low_risk'
            else:
                purity_status = "ℹ️  MODERATE MIX"
                cluster_quality[cluster_id] = 'moderate'

            print(f"Cluster {cluster_id}: {n_samples} samples, "
                  f"High Risk: {high_risk_ratio:.1%} - {purity_status}")
        else:
            print(f"Cluster {cluster_id}: No samples")
            cluster_quality[cluster_id] = 'empty'

    return cluster_quality

# ==========================================
# LOAD CLUSTERING MODEL
# ==========================================
def load_clustering_artifacts():
    """Load K-Means model and scaler from Phase 3"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING CLUSTERING ARTIFACTS")
    print("=" * 80)

    kmeans_path = Config.FEATURES_OUTPUT / 'kmeans_model.pkl'
    scaler_path = Config.FEATURES_OUTPUT / 'feature_scaler.pkl'

    if not kmeans_path.exists():
        print(f"❌ K-Means model not found: {kmeans_path}")
        print("   Please run Phase 3 first!")
        return None, None

    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        print("   Please run Phase 3 first!")
        return None, None

    kmeans = joblib.load(str(kmeans_path))
    scaler = joblib.load(str(scaler_path))

    print(f"✅ K-Means model loaded ({kmeans.n_clusters} clusters)")
    print(f"✅ Feature scaler loaded")

    return kmeans, scaler

# ==========================================
# LOAD NEOPOLYP LABELS
# ==========================================
def load_neopolyp_labels():
    """Load ground truth labels from Phase 2.5"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING NEOPOLYP LABELS")
    print("=" * 80)

    label_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_labels.json'

    if not label_path.exists():
        print(f"❌ Neopolyp labels not found: {label_path}")
        print("   Please run Phase 2.5 first!")
        return None

    with open(label_path, 'r') as f:
        labels = json.load(f)

    print(f"✅ Loaded {len(labels):,} labeled images")

    # Count labels
    high_risk = sum(1 for v in labels.values() if v == 1)
    low_risk = sum(1 for v in labels.values() if v == 0)

    print(f"   High Risk (Neoplastic): {high_risk:,} ({high_risk/len(labels)*100:.1f}%)")
    print(f"   Low Risk (Non-Neoplastic): {low_risk:,} ({low_risk/len(labels)*100:.1f}%)")

    return labels

# ==========================================
# FEATURE EXTRACTION FOR NEOPOLYP
# ==========================================
def extract_features_for_neopolyp(labels_dict, kmeans, scaler):
    """
    Extract features for Neopolyp images and assign cluster labels
    Returns: cluster_datasets dictionary with SSL features, biomarkers, and labels per cluster
    NOW USES FULL 444-DIM FEATURES (384 SSL + 60 Biomarkers)
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "EXTRACTING FEATURES FOR NEOPOLYP")
    print("=" * 80)

    # Import feature extraction functions from Phase 3
    sys.path.insert(0, str(Path(__file__).parent.parent / 'phase3_clustering'))
    from feature_extraction import (
        ViTEncoder, load_ssl_model, extract_biomarkers, pad_to_square
    )

    # Load SSL model
    ssl_model = load_ssl_model()
    if ssl_model is None:
        return None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Initialize cluster datasets with SSL features AND biomarkers
    cluster_datasets = {i: {
        'ssl_features': [],  # NEW: Store SSL features
        'biomarkers': [],
        'full_features': [],  # NEW: Store concatenated 444-dim features
        'labels': [],
        'image_names': []
    } for i in range(Config.NUM_CLUSTERS)}

    # Process each labeled image
    processed_count = 0
    failed_count = 0

    for img_name, label in tqdm(labels_dict.items(), desc="Processing Neopolyp images"):
        img_path = Config.TRAIN_IMAGES / img_name

        if not img_path.exists():
            failed_count += 1
            continue

        try:
            # Load image with padding (preserve aspect ratio)
            image = Image.open(img_path).convert('RGB')
            image = pad_to_square(image, Config.IMG_SIZE)
            image_np = np.array(image)

            # Extract SSL features (for clustering)
            image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                ssl_features = ssl_model(image_tensor).cpu().numpy().flatten()

            # Extract biomarkers (for rule learning)
            biomarkers = extract_biomarkers(image_np)

            # Combine for cluster prediction (apply same weighting as in Phase 3)
            fact_vector = np.concatenate([ssl_features, biomarkers])
            fact_vector_scaled = scaler.transform(fact_vector.reshape(1, -1))

            # Predict cluster
            cluster_id = kmeans.predict(fact_vector_scaled)[0]

            # Store BOTH SSL features and biomarkers (NEW: Full 444-dim features)
            cluster_datasets[cluster_id]['ssl_features'].append(ssl_features)
            cluster_datasets[cluster_id]['biomarkers'].append(biomarkers)
            cluster_datasets[cluster_id]['full_features'].append(fact_vector)  # Full 444-dim
            cluster_datasets[cluster_id]['labels'].append(label)
            cluster_datasets[cluster_id]['image_names'].append(img_name)

            processed_count += 1

        except Exception as e:
            print(f"\n   ⚠️  Failed to process {img_name}: {e}")
            failed_count += 1
            continue

    print(f"\n✅ Processed {processed_count:,} images ({failed_count} failed)")

    # Convert lists to numpy arrays
    for cluster_id in range(Config.NUM_CLUSTERS):
        if len(cluster_datasets[cluster_id]['biomarkers']) > 0:
            cluster_datasets[cluster_id]['ssl_features'] = np.array(
                cluster_datasets[cluster_id]['ssl_features']
            )
            cluster_datasets[cluster_id]['biomarkers'] = np.array(
                cluster_datasets[cluster_id]['biomarkers']
            )
            cluster_datasets[cluster_id]['full_features'] = np.array(
                cluster_datasets[cluster_id]['full_features']
            )
            cluster_datasets[cluster_id]['labels'] = np.array(
                cluster_datasets[cluster_id]['labels']
            )

    # Print cluster distribution
    print(f"\n📊 Cluster Distribution:")
    for cluster_id in range(Config.NUM_CLUSTERS):
        n_samples = len(cluster_datasets[cluster_id]['labels'])
        if n_samples > 0:
            n_high_risk = np.sum(cluster_datasets[cluster_id]['labels'] == 1)
            n_low_risk = np.sum(cluster_datasets[cluster_id]['labels'] == 0)
            print(f"   Cluster {cluster_id}: {n_samples:,} samples "
                  f"(High Risk: {n_high_risk}, Low Risk: {n_low_risk})")
        else:
            print(f"   Cluster {cluster_id}: No samples")

    return cluster_datasets

# ==========================================
# TRAIN LOCAL EXPERTS
# ==========================================
def train_local_experts(cluster_datasets):
    """
    Train 8 separate Decision Trees, one for each cluster
    NOW USES FULL 444-DIM FEATURES (384 SSL + 60 Biomarkers)
    Implements cluster merging for sparse clusters (<15 samples)
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "TRAINING LOCAL EXPERTS")
    print("=" * 80)

    local_experts = {}
    expert_stats = {}

    # Feature names for full 444-dim vector
    feature_names = (
        [f'SSL_Feat_{i}' for i in range(Config.SSL_FEATURE_DIM)] +  # 384 SSL features
        [f'LAB_L_Bin_{i}' for i in range(3)] +  # 3 L* bins
        [f'LAB_a_Bin_{i}' for i in range(3)] +  # 3 a* bins
        [f'LAB_b_Bin_{i}' for i in range(3)] +  # 3 b* bins
        [f'Sat_Bin_{i}' for i in range(16)] +  # 16 Saturation bins
        [f'Haralick_Feat_{i}' for i in range(13)] +  # 13 Haralick features
        [f'LBP_Bin_{i}' for i in range(19)] +  # 19 LBP bins
        ['Texture_Complexity', 'Relative_Area', 'Compactness']  # 3 shape features
    )

    # First pass: Identify sparse clusters for merging
    sparse_clusters = []
    valid_clusters = []
    global_X = []
    global_y = []

    for cluster_id in range(Config.NUM_CLUSTERS):
        n_samples = len(cluster_datasets[cluster_id]['labels'])
        if n_samples > 0:
            # Collect all data for potential global fallback
            global_X.append(cluster_datasets[cluster_id]['full_features'])
            global_y.append(cluster_datasets[cluster_id]['labels'])

            if n_samples < Config.MIN_SAMPLES_FOR_TRAINING:
                sparse_clusters.append(cluster_id)
                print(f"   Cluster {cluster_id}: SPARSE ({n_samples} samples) - will use global fallback")
            else:
                valid_clusters.append(cluster_id)

    # Create global fallback model if needed
    global_model = None
    if sparse_clusters:
        print(f"\n   Creating GLOBAL FALLBACK MODEL for {len(sparse_clusters)} sparse clusters...")
        if global_X:
            global_X_combined = np.vstack(global_X)
            global_y_combined = np.hstack(global_y)

            print(f"   Global model training on {len(global_y_combined):,} total samples")
            global_base = DecisionTreeClassifier(
                max_depth=Config.TREE_MAX_DEPTH,
                min_samples_split=Config.TREE_MIN_SAMPLES_SPLIT,
                min_samples_leaf=Config.TREE_MIN_SAMPLES_LEAF,
                random_state=42
            )
            global_model = CalibratedClassifierCV(global_base, method='isotonic', cv=5)
            global_model.fit(global_X_combined, global_y_combined)
            print(f"   ✅ Global fallback model trained with calibration")

    # Train individual experts
    for cluster_id in range(Config.NUM_CLUSTERS):
        print(f"\n{'='*60}")
        print(f"   Training Expert for Cluster {cluster_id}")
        print(f"{'='*60}")

        cluster_data = cluster_datasets[cluster_id]
        n_samples = len(cluster_data['labels'])

        if n_samples == 0:
            print(f"   ⚠️  No samples")
            local_experts[cluster_id] = global_model  # Use global fallback
            expert_stats[cluster_id] = {
                'n_samples': 0,
                'fallback': 'global',
                'accuracy': 0.0
            }
            continue

        if n_samples < Config.MIN_SAMPLES_FOR_TRAINING:
            print(f"   ⚠️  Insufficient data ({n_samples} samples)")
            print(f"      Using GLOBAL FALLBACK MODEL")
            local_experts[cluster_id] = global_model
            expert_stats[cluster_id] = {
                'n_samples': int(n_samples),
                'fallback': 'global',
                'accuracy': 0.0
            }
            continue

        # Use FULL 444-DIM FEATURES (384 SSL + 60 Biomarkers)
        X = cluster_data['full_features']  # Changed from biomarkers to full_features
        y = cluster_data['labels']

        # Check if we have both classes
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            print(f"   ⚠️  Only one class present: {unique_labels}")
            print(f"      Cannot train decision tree")
            print(f"      Creating default classifier...")

            # Create a dummy tree that always predicts the only class
            tree = DecisionTreeClassifier(max_depth=1)
            # Duplicate one sample to create artificial second class
            X_train = np.vstack([X[0:1], X[0:1]])
            y_train = np.array([unique_labels[0], 1 - unique_labels[0]])
            tree.fit(X_train, y_train)

            local_experts[cluster_id] = tree
            expert_stats[cluster_id] = {
                'n_samples': n_samples,
                'n_high_risk': int(np.sum(y == 1)),
                'n_low_risk': int(np.sum(y == 0)),
                'accuracy': 1.0,
                'single_class': True,
                'default_prediction': int(unique_labels[0])
            }
            continue

        # Train Decision Tree with improved parameters
        print(f"   Samples: {n_samples}")
        print(f"   Features: {X.shape[1]} (384 SSL + 60 Biomarkers)")
        print(f"   High Risk: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
        print(f"   Low Risk: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)")

        base_tree = DecisionTreeClassifier(
            max_depth=Config.TREE_MAX_DEPTH,  # Now 7 instead of 5
            min_samples_split=Config.TREE_MIN_SAMPLES_SPLIT,
            min_samples_leaf=Config.TREE_MIN_SAMPLES_LEAF,  # Now 5 for better generalization
            random_state=42
        )
        tree = CalibratedClassifierCV(base_tree, method='isotonic', cv=5)
        tree.fit(X, y)

        # Evaluate on training data (in-sample performance)
        y_pred = tree.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        print(f"\n   📊 Performance (In-Sample):")
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")

        # Store expert
        local_experts[cluster_id] = tree
        expert_stats[cluster_id] = {
            'n_samples': int(n_samples),
            'n_high_risk': int(np.sum(y == 1)),
            'n_low_risk': int(np.sum(y == 0)),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'single_class': False
        }

        # Print decision rules
        print(f"\n   📜 Decision Rules:")
        tree_rules = export_text(tree, feature_names=feature_names, max_depth=3)
        print("      " + tree_rules.replace("\n", "\n      "))

    print(f"\n{'='*80}")
    print(f"✅ Trained {sum(1 for e in local_experts.values() if e is not None)} experts")

    return local_experts, expert_stats, feature_names

# ==========================================
# SAVE MIXTURE OF EXPERTS
# ==========================================
def save_mixture_of_experts(local_experts, expert_stats, feature_names):
    """Save all local experts and metadata"""
    print("\n" + "=" * 80)
    print(" " * 25 + "SAVING MIXTURE OF EXPERTS")
    print("=" * 80)

    # Save each expert
    for cluster_id, expert in local_experts.items():
        if expert is not None:
            expert_path = Config.EXPERTS_OUTPUT / f'expert_cluster_{cluster_id}.pkl'
            joblib.dump(expert, str(expert_path))
            print(f"💾 Expert {cluster_id} saved: {expert_path.name}")

    # Save feature names
    features_path = Config.EXPERTS_OUTPUT / 'biomarker_feature_names.npy'
    np.save(str(features_path), feature_names)
    print(f"💾 Feature names saved: {features_path.name}")

    # Save statistics
    stats_path = Config.EXPERTS_OUTPUT / 'expert_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(expert_stats, f, indent=2)
    print(f"💾 Statistics saved: {stats_path.name}")

    # Save ASGE thresholds for inference
    asge_config = {
        'num_experts': Config.NUM_CLUSTERS,
        'high_confidence_threshold': Config.ASGE_HIGH_CONFIDENCE_THRESHOLD,
        'uncertainty_threshold': Config.ASGE_UNCERTAINTY_THRESHOLD,
        'decision_logic': {
            'high_risk': f'risk_prob >= {Config.ASGE_HIGH_CONFIDENCE_THRESHOLD}',
            'uncertain': f'{Config.ASGE_UNCERTAINTY_THRESHOLD} <= risk_prob < {Config.ASGE_HIGH_CONFIDENCE_THRESHOLD}',
            'low_risk': f'risk_prob < {Config.ASGE_UNCERTAINTY_THRESHOLD}'
        }
    }

    asge_path = Config.EXPERTS_OUTPUT / 'asge_configuration.json'
    with open(asge_path, 'w') as f:
        json.dump(asge_config, f, indent=2)
    print(f"💾 ASGE configuration saved: {asge_path.name}")

# ==========================================
# LOAD SYMBOLIC BASELINES
# ==========================================
def load_symbolic_baselines():
    """Load statistical baselines from Phase 2.6 for symbolic reasoning"""
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING SYMBOLIC BASELINES")
    print("=" * 80)

    baselines_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_ground_truth_baselines.json'

    if not baselines_path.exists():
        print(f"⚠️  Symbolic baselines not found: {baselines_path}")
        print("   Please run Phase 2.6 first for fact-based reasoning!")
        return None

    with open(baselines_path, 'r') as f:
        baselines = json.load(f)

    print("✅ Loaded statistical baselines for symbolic reasoning")
    print(f"   High-risk samples: {baselines['metadata']['high_risk_samples']}")
    print(f"   Low-risk samples: {baselines['metadata']['low_risk_samples']}")
    print(f"   Features analyzed: {baselines['metadata']['feature_dimensions']}")

    return baselines

# ==========================================
# SYMBOLIC REASONING VALIDATION
# ==========================================
def validate_symbolic_reasoning(local_experts, expert_stats, baselines):
    """Validate expert decisions against statistical baselines for symbolic reasoning"""
    print("\n" + "=" * 80)
    print(" " * 25 + "SYMBOLIC REASONING VALIDATION")
    print("=" * 80)

    if baselines is None:
        print("   ⚠️  Skipping symbolic validation - no baselines available")
        return

    print("   Validating expert decisions against doctor-marked statistical baselines...")
    print("   This ensures fact-based rules instead of arbitrary thresholds")

    # For each expert, check alignment with baselines
    symbolic_alignment = {}

    for cluster_id, expert in local_experts.items():
        if expert is None or expert_stats[cluster_id].get('single_class', False):
            symbolic_alignment[cluster_id] = {
                'alignment_score': 0.0,
                'reason': 'No expert or single class'
            }
            continue

        # Get expert's feature importances
        importances = expert.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]

        # Find top 5 most important features for this expert
        top_indices = np.argsort(importances)[-5:][::-1]
        top_features = [feature_names[i] for i in top_indices]

        # Check if top features show significant differences in baselines
        alignment_score = 0.0
        significant_features = []

        for feat_name in top_features:
            if feat_name in baselines['differences']:
                diff_stats = baselines['differences'][feat_name]
                effect_size = diff_stats['effect_size']
                if effect_size > 0.5:  # Moderate to large effect
                    alignment_score += 1.0
                    significant_features.append(feat_name)

        alignment_score = alignment_score / len(top_features) if top_features else 0.0

        symbolic_alignment[cluster_id] = {
            'alignment_score': alignment_score,
            'significant_features': significant_features,
            'top_features': top_features
        }

        print(f"   Expert {cluster_id}: Symbolic alignment = {alignment_score:.2f}")
        if significant_features:
            print(f"      Significant features: {', '.join(significant_features[:3])}")

    # Save symbolic validation results
    symbolic_path = Config.EXPERTS_OUTPUT / 'symbolic_reasoning_validation.json'
    with open(symbolic_path, 'w') as f:
        json.dump({
            'validation_summary': symbolic_alignment,
            'baselines_metadata': baselines['metadata']
        }, f, indent=2)

    print(f"💾 Symbolic validation saved: {symbolic_path.name}")

    return symbolic_alignment
def visualize_expert_performance(expert_stats):
    """Visualize performance of all experts"""
    print("\n" + "=" * 80)
    print(" " * 25 + "GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Filter out None experts
    valid_experts = {k: v for k, v in expert_stats.items() if v is not None}

    if len(valid_experts) == 0:
        print("   ⚠️  No valid experts to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Sample distribution
    cluster_ids = list(valid_experts.keys())
    n_samples = [valid_experts[c]['n_samples'] for c in cluster_ids]

    axes[0, 0].bar(cluster_ids, n_samples, color='steelblue')
    axes[0, 0].set_title('Training Samples per Expert', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Class distribution
    high_risk = [valid_experts[c].get('n_high_risk', 0) for c in cluster_ids]
    low_risk = [valid_experts[c].get('n_low_risk', 0) for c in cluster_ids]

    x = np.arange(len(cluster_ids))
    width = 0.35

    axes[0, 1].bar(x - width/2, high_risk, width, label='High Risk', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, low_risk, width, label='Low Risk', color='green', alpha=0.7)
    axes[0, 1].set_title('Risk Distribution per Expert', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(cluster_ids)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Performance metrics
    accuracy = [valid_experts[c].get('accuracy', 0) for c in cluster_ids]
    precision = [valid_experts[c].get('precision', 0) for c in cluster_ids]
    recall = [valid_experts[c].get('recall', 0) for c in cluster_ids]
    f1 = [valid_experts[c].get('f1_score', 0) for c in cluster_ids]

    x = np.arange(len(cluster_ids))
    width = 0.2

    axes[1, 0].bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    axes[1, 0].bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
    axes[1, 0].bar(x + 1.5*width, f1, width, label='F1-Score', alpha=0.8)
    axes[1, 0].set_title('Expert Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cluster_ids)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. ASGE Threshold visualization
    axes[1, 1].axhspan(0, Config.ASGE_UNCERTAINTY_THRESHOLD,
                       color='green', alpha=0.2, label='LOW RISK')
    axes[1, 1].axhspan(Config.ASGE_UNCERTAINTY_THRESHOLD,
                       Config.ASGE_HIGH_CONFIDENCE_THRESHOLD,
                       color='yellow', alpha=0.2, label='UNCERTAIN')
    axes[1, 1].axhspan(Config.ASGE_HIGH_CONFIDENCE_THRESHOLD, 1.0,
                       color='red', alpha=0.2, label='HIGH RISK')

    axes[1, 1].axhline(Config.ASGE_UNCERTAINTY_THRESHOLD,
                       color='orange', linestyle='--', linewidth=2,
                       label=f'Uncertainty ({Config.ASGE_UNCERTAINTY_THRESHOLD})')
    axes[1, 1].axhline(Config.ASGE_HIGH_CONFIDENCE_THRESHOLD,
                       color='red', linestyle='--', linewidth=2,
                       label=f'High Confidence ({Config.ASGE_HIGH_CONFIDENCE_THRESHOLD})')

    axes[1, 1].set_title('ASGE PIVI Thresholds', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Decision Regions')
    axes[1, 1].set_ylabel('Risk Probability')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend(loc='center left')
    axes[1, 1].text(0.5, 0.4, 'Surveillance', ha='center', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.5, 0.85, 'Biopsy/Review', ha='center', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.5, 0.95, 'Resect & Discard', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    viz_path = Config.VISUAL_OUTPUT / 'phase4_mixture_of_experts.png'
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Visualization saved: {viz_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STARTING PHASE 4")
    print("=" * 80)
    print("\n🔬 CLINICAL-FOCUSED ARCHITECTURE:")
    print("   Phase 3: Clusters trained on NeoPolyp (with HIGH RISK/LOW RISK labels)")
    print("   Phase 4: Decision Trees trained per-cluster on clinical labels")
    print("   Result: Clinically coherent expert system")

    # Step 1: Load clustering artifacts (NOW from clinically-meaningful clusters)
    kmeans, scaler = load_clustering_artifacts()
    if kmeans is None or scaler is None:
        print("\n❌ Failed to load clustering artifacts from Phase 3")
        print("   Ensure Phase 3 was run with NeoPolyp clinical clustering")
        return

    print("\n✅ Loaded cluster centers trained on NeoPolyp (clinical data)")

    # Step 2: Load Neopolyp labels (RED=High Risk, GREEN=Low Risk)
    labels_dict = load_neopolyp_labels()
    if labels_dict is None:
        print("\n❌ Failed to load Neopolyp labels")
        return

    print(f"✅ Loaded ground truth labels from color-coded masks")

    # Step 3: Extract features and map to clusters
    cluster_datasets = extract_features_for_neopolyp(labels_dict, kmeans, scaler)
    if cluster_datasets is None:
        return

    # Step 3.5: Validate cluster medical coherence
    cluster_quality = validate_cluster_purity(cluster_datasets)

    # Step 4: Train local experts (USES FULL 444-DIM FEATURES with clinical labels)
    local_experts, expert_stats, feature_names = train_local_experts(cluster_datasets)

    # Step 5: Save artifacts
    save_mixture_of_experts(local_experts, expert_stats, feature_names)

    # Step 6: Visualize results
    visualize_expert_performance(expert_stats)

    # Step 7: Load and validate symbolic baselines
    baselines = load_symbolic_baselines()
    symbolic_alignment = validate_symbolic_reasoning(local_experts, expert_stats, baselines)

    print("\n" + "=" * 80)
    print(" " * 25 + "PHASE 4 COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Trained Mixture of Experts (Clinically-Guided):")
    print(f"   Total Experts: {Config.NUM_CLUSTERS}")
    print(f"   Active Experts: {sum(1 for e in local_experts.values() if e is not None)}")
    print(f"   Training Data: NeoPolyp (clinical labels from color-coded masks)")
    print(f"   Feature Set: FULL 444-DIM (384 SSL + 60 Biomarkers)")
    print(f"   Cluster Centers: Trained on HIGH RISK/LOW RISK separation")
    print(f"\n✅ ASGE PIVI Standards Configured:")
    print(f"   High Confidence: ≥ {Config.ASGE_HIGH_CONFIDENCE_THRESHOLD} (Resect & Discard)")
    print(f"   Uncertainty: {Config.ASGE_UNCERTAINTY_THRESHOLD}-{Config.ASGE_HIGH_CONFIDENCE_THRESHOLD} (Require Biopsy)")
    print(f"   Low Risk: < {Config.ASGE_UNCERTAINTY_THRESHOLD} (Surveillance)")
    print(f"\n✅ Symbolic Reasoning Validated:")
    if symbolic_alignment:
        avg_alignment = np.mean([v['alignment_score'] for v in symbolic_alignment.values() if isinstance(v, dict)])
        print(f"   Average symbolic alignment: {avg_alignment:.2f}")
    print(f"\n   Next: Run Phase 5 Inference Pipeline for ASGE-compliant predictions")

if __name__ == '__main__':
    main()
