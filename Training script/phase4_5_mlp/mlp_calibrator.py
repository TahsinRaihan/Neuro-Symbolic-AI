# -*- coding: utf-8 -*-
"""
PHASE 4.5: MLP PROBABILITY CALIBRATOR
Trains a Multi-Layer Perceptron (MLP) on SSL-extracted 444-dim feature vectors
to output instance-level, polyp-specific cancerous probability.

PROBLEM SOLVED:
    Legacy approach: cluster-static probability
        e.g., cluster 2 has 13/20 high-risk  --> ALL polyps in cluster 2 get 65%
    This approach: instance-level MLP probability
        e.g., two polyps in cluster 2 get 0.71 and 0.58 based on their own 444-dim fingerprint

ARCHITECTURE:
    Input  : 452-dim  =  444 raw features  +  8 one-hot cluster ID
    Layer 1: Linear(452 → 256) + BatchNorm + ReLU + Dropout(0.35)
    Layer 2: Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.30)
    Layer 3: Linear(128 →  64) + BatchNorm + ReLU + Dropout(0.20)
    Output : Linear(64  →   1)  →  Sigmoid  →  P(Neoplastic / High-Risk)

TRAINING DATA:
    NeoPolyp dataset (same images used in Phase 4):
        RED  mask pixel dominant  →  label 1  (Neoplastic / High-Risk)
        GREEN mask pixel dominant →  label 0  (Non-neoplastic / Low-Risk)

OUTPUTS:
    thesis_outputs/mlp_calibrator/
        mlp_model.pth                  ← model weights
        mlp_scaler.pkl                 ← StandardScaler fitted on 444-dim features
        mlp_metadata.json              ← architecture + training statistics
        ground_truth_analysis.txt      ← human-readable GT + feature report
        polyp_characteristics.csv      ← full per-image feature table
        visualizations/
            training_curves.png
            calibration_comparison.png ← MLP vs cluster-static probability
            probability_distribution.png
"""

import os
# Prevent Windows OpenMP / MKL multi-thread deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
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
import csv
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)

print("=" * 80)
print(" " * 10 + "PHASE 4.5: MLP PROBABILITY CALIBRATOR")
print(" " * 8 + "(Instance-Level Polyp-Specific Cancerous Probability)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT   = Path(__file__).parent.parent.parent.absolute()

    # Data sources
    NEOPOLYP_ROOT = THESIS_ROOT / 'NeSy' / 'Neo polyp Dataset'
    TRAIN_IMAGES  = NEOPOLYP_ROOT / 'train' / 'train'
    TRAIN_MASKS   = NEOPOLYP_ROOT / 'train_gt' / 'train_gt'

    # Phase 3 artifacts (K-Means + scaler)
    FEATURES_OUTPUT  = THESIS_ROOT / 'thesis_outputs' / 'extracted_features'
    SSL_OUTPUT       = THESIS_ROOT / 'thesis_outputs' / 'ssl_outputs'
    NEOPOLYP_OUTPUT  = THESIS_ROOT / 'thesis_outputs' / 'neopolyp_processed'

    # MLP outputs
    MLP_OUTPUT       = THESIS_ROOT / 'thesis_outputs' / 'mlp_calibrator'
    VISUAL_OUTPUT    = MLP_OUTPUT / 'visualizations'

    # Feature dimensions (must match Phase 3)
    SSL_FEATURE_DIM   = 384
    BIOMARKER_DIM     = 60
    TOTAL_FEATURE_DIM = 444          # SSL + Biomarkers
    NUM_CLUSTERS      = 8
    MLP_INPUT_DIM     = TOTAL_FEATURE_DIM + NUM_CLUSTERS  # 444 + 8 = 452

    # MLP hyper-parameters
    HIDDEN_DIMS   = [256, 128, 64]
    DROPOUT_RATES = [0.35, 0.30, 0.20]  # per hidden layer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    BATCH_SIZE    = 64  # Increased for better GPU utilization
    MAX_EPOCHS    = 100
    PATIENCE      = min(20, max(2, MAX_EPOCHS // 5))  # scale patience with epochs

    # Train / validation split
    VAL_SPLIT     = 0.20
    RANDOM_SEED   = 42

    # ASGE PIVI thresholds (same as rest of pipeline)
    ASGE_HIGH_CONFIDENCE = 0.90
    ASGE_UNCERTAINTY     = 0.80

    IMG_SIZE = 256
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for d in [Config.MLP_OUTPUT, Config.VISUAL_OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n  Device          : {Config.DEVICE}")
print(f"  Feature dim     : {Config.TOTAL_FEATURE_DIM}  (384 SSL + 60 Biomarkers)")
print(f"  MLP input dim   : {Config.MLP_INPUT_DIM}  (444 + 8 one-hot cluster)")
print(f"  Hidden dims     : {Config.HIDDEN_DIMS}")
print(f"  Clusters        : {Config.NUM_CLUSTERS}")
print(f"  Max epochs      : {Config.MAX_EPOCHS}  |  Patience: {Config.PATIENCE}")

# ---------------------------------------------------------------------------
# Bring Phase-3 helpers into scope without modifying them
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase3_clustering'))
from feature_extraction import (ViTEncoder, load_ssl_model,
                                 extract_biomarkers, pad_to_square)

# ==========================================
# MLP MODEL DEFINITION
# ==========================================
class MLPCalibrator(nn.Module):
    """
    Multi-Layer Perceptron probability calibrator.
    Replaces cluster-static probabilities with polyp-specific instance probabilities.

    Input : [444-dim raw feature fingerprint  |  8-dim one-hot cluster ID]
    Output: scalar logit  →  sigmoid gives P(Neoplastic / High-Risk)
    """

    def __init__(self, input_dim=452, hidden_dims=None, dropout_rates=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.35, 0.30, 0.20]

        assert len(dropout_rates) == len(hidden_dims), \
            "dropout_rates and hidden_dims must have equal length"

        blocks = []
        in_dim = input_dim
        for h_dim, dr in zip(hidden_dims, dropout_rates):
            blocks += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dr),
            ]
            in_dim = h_dim

        blocks.append(nn.Linear(in_dim, 1))   # final logit
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x).squeeze(-1)    # shape: (B,)

    def predict_probability(self, x):
        """Return calibrated probability in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ==========================================
# GROUND-TRUTH LABEL LOADING
# ==========================================
def load_neopolyp_labels():
    """
    Load or create binary ground-truth labels from the NeoPolyp colour masks.
    RED dominant  → 1 (Neoplastic / High-Risk)
    GREEN dominant → 0 (Non-neoplastic / Low-Risk)
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING NEOPOLYP GROUND-TRUTH LABELS")
    print("=" * 80)

    label_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_labels.json'

    if label_path.exists():
        with open(label_path) as f:
            labels = json.load(f)
        print(f"  Loaded {len(labels):,} labels from cache")
    else:
        print("  Cache not found – deriving labels from colour masks …")
        Config.NEOPOLYP_OUTPUT.mkdir(parents=True, exist_ok=True)
        labels = {}

        mask_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            mask_files.extend(Config.TRAIN_MASKS.glob(ext))

        print(f"  Found {len(mask_files)} mask files")
        if not mask_files:
            print("  ERROR: No mask files found")
            return {}

        for mp in mask_files:
            mask = cv2.imread(str(mp))
            if mask is None:
                continue
            red_px   = int(np.sum((mask[:, :, 2] > 200) &
                                  (mask[:, :, 1] <  50) &
                                  (mask[:, :, 0] <  50)))
            green_px = int(np.sum((mask[:, :, 1] > 200) &
                                  (mask[:, :, 2] <  50) &
                                  (mask[:, :, 0] <  50)))
            labels[mp.name] = 1 if red_px >= green_px else 0

        with open(label_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"  Created {len(labels)} labels and cached to disk")

    high = sum(v == 1 for v in labels.values())
    low  = sum(v == 0 for v in labels.values())
    print(f"\n  NEOPLASTIC  (High-Risk / Remove) : {high:,}  ({high/max(len(labels),1)*100:.1f}%)")
    print(f"  NON-NEOPLASTIC (Low-Risk / Leave) : {low:,}   ({low/max(len(labels),1)*100:.1f}%)")
    return labels


# ==========================================
# FEATURE EXTRACTION FOR NEOPOLYP IMAGES
# ==========================================
def extract_neopolyp_features(labels_dict, kmeans, kmeans_scaler):
    """
    For every labeled NeoPolyp image:
      1. Extract raw 444-dim fact vector  (384 SSL + 60 biomarkers)
      2. Predict cluster ID  (using Phase-3 K-Means + scaler)
      3. One-hot encode cluster
      4. Build MLP input = [444-dim | 8-dim one-hot]

    Returns
    -------
    records : list of dicts  (one per image, full feature data for CSV/TXT output)
    X       : np.ndarray  shape (N, 452)  – MLP inputs
    y       : np.ndarray  shape (N,)      – binary labels
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "EXTRACTING FEATURES FROM NEOPOLYP DATASET")
    print("=" * 80)

    ssl_model = load_ssl_model()
    if ssl_model is None:
        print("  ERROR: SSL model not available. Run Phase 2 first.")
        return None, None, None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    records   = []
    X_list    = []
    y_list    = []
    failed    = 0

    # Biomarker field names (in extraction order)
    bio_names = (
        [f'LAB_L_bin{i}' for i in range(3)] +
        [f'LAB_a_bin{i}' for i in range(3)] +
        [f'LAB_b_bin{i}' for i in range(3)] +
        [f'Sat_bin{i}'   for i in range(16)] +
        [f'Haralick_{i}' for i in range(13)] +
        [f'LBP_bin{i}'   for i in range(19)] +
        ['Texture_Complexity', 'Relative_Area', 'Compactness']
    )

    for img_name, label in tqdm(labels_dict.items(),
                                 desc="  Extracting features", unit="img"):
        img_path = Config.TRAIN_IMAGES / img_name
        if not img_path.exists():
            failed += 1
            continue

        try:
            image     = Image.open(img_path).convert('RGB')
            image     = pad_to_square(image, Config.IMG_SIZE)
            image_np  = np.array(image)

            # Stream A – SSL features (384-dim)
            img_t = transform(image).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                ssl_feats = ssl_model(img_t).cpu().numpy().flatten()  # (384,)

            # Stream B – Biomarkers (60-dim)
            biomarkers = extract_biomarkers(image_np)                 # (60,)

            # Full 444-dim fact vector (raw, unscaled)
            fact_vec = np.concatenate([ssl_feats, biomarkers])        # (444,)

            # Cluster assignment using Phase-3 K-Means
            fv_scaled  = kmeans_scaler.transform(fact_vec.reshape(1, -1))
            cluster_id = int(kmeans.predict(fv_scaled)[0])

            # One-hot encode cluster  (8-dim)
            cluster_oh = np.zeros(Config.NUM_CLUSTERS, dtype=np.float32)
            cluster_oh[cluster_id] = 1.0

            # MLP input vector (452-dim)
            mlp_input = np.concatenate([fact_vec, cluster_oh]).astype(np.float32)

            X_list.append(mlp_input)
            y_list.append(float(label))

            # Build record for analysis output
            rec = {
                'image_name'   : img_name,
                'ground_truth' : int(label),
                'gt_text'      : 'Neoplastic (High-Risk)'
                                  if label == 1
                                  else 'Non-neoplastic (Low-Risk)',
                'cluster_id'   : cluster_id,
                'ssl_feat_l2'  : float(np.linalg.norm(ssl_feats)),
            }
            for bn, bv in zip(bio_names, biomarkers.tolist()):
                rec[bn] = round(bv, 6)
            records.append(rec)

        except Exception as e:
            failed += 1

    print(f"\n  Processed : {len(X_list):,}  images  ({failed} failed)")

    if not X_list:
        print("  ERROR: No features could be extracted.")
        return None, None, None

    X = np.array(X_list)   # (N, 452)
    y = np.array(y_list)   # (N,)

    print(f"  Feature matrix : {X.shape}")
    print(f"  Label balance  : {int(y.sum())} high-risk  |  {int((1-y).sum())} low-risk")

    return records, X, y


# ==========================================
# CLUSTER-STATIC PROBABILITY LOOKUP
# ==========================================
def compute_cluster_static_probs(y, cluster_ids):
    """
    Compute the 'static' probability for each cluster:
        P_static(cluster_k) = count(high-risk in cluster k) / count(all in cluster k)
    This is what the legacy system would have assigned to every polyp in that cluster.
    """
    static_map = {}
    for k in range(Config.NUM_CLUSTERS):
        mask = cluster_ids == k
        if mask.sum() == 0:
            static_map[k] = 0.5   # no data → uncertain
        else:
            static_map[k] = float(y[mask].mean())
    return static_map   # { cluster_id : static_prob }


# ==========================================
# TRAIN MLP
# ==========================================
def train_mlp(X_train_raw, y_train, X_val_raw, y_val):
    """
    Fit a StandardScaler on the 444-dim feature portion of X_train,
    (NOT the 8-dim cluster one-hot which is already in [0,1]),
    then train the MLP with BCEWithLogitsLoss + class-imbalance weighting.

    Returns
    -------
    model   : trained MLPCalibrator (CPU)
    scaler  : fitted StandardScaler for the 444-dim feature block
    history : dict with 'train_loss', 'val_loss', 'val_auc' lists
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "TRAINING MLP CALIBRATOR")
    print("=" * 80)

    # ── Normalise only the 444 feature columns; leave the 8 one-hot cols alone ──
    feat_train = X_train_raw[:, :Config.TOTAL_FEATURE_DIM]
    oh_train   = X_train_raw[:, Config.TOTAL_FEATURE_DIM:]

    feat_val   = X_val_raw[:, :Config.TOTAL_FEATURE_DIM]
    oh_val     = X_val_raw[:, Config.TOTAL_FEATURE_DIM:]

    scaler = StandardScaler()
    feat_train_sc = scaler.fit_transform(feat_train)
    feat_val_sc   = scaler.transform(feat_val)

    X_train_sc = np.hstack([feat_train_sc, oh_train]).astype(np.float32)
    X_val_sc   = np.hstack([feat_val_sc,   oh_val  ]).astype(np.float32)

    # ── Class-imbalance weight ───────────────────────────────────────────────
    n_pos     = float(y_train.sum())
    n_neg     = float(len(y_train) - n_pos)
    pos_w_val = max(n_neg / (n_pos + 1e-7), 1.0)     # ≥ 1
    pos_weight = torch.tensor([pos_w_val])

    print(f"  Train: {len(y_train)} | Val: {len(y_val)}")
    print(f"  Class imbalance weight (pos/high-risk) : {pos_w_val:.3f}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train_sc),
                              torch.from_numpy(y_train.astype(np.float32)))
    val_ds   = TensorDataset(torch.from_numpy(X_val_sc),
                              torch.from_numpy(y_val.astype(np.float32)))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  drop_last=False, num_workers=8)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=8)

    # ── Model + optimiser ────────────────────────────────────────────────────
    model     = MLPCalibrator(input_dim=Config.MLP_INPUT_DIM,
                               hidden_dims=Config.HIDDEN_DIMS,
                               dropout_rates=Config.DROPOUT_RATES).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(Config.DEVICE))
    optimiser = torch.optim.Adam(model.parameters(),
                                  lr=Config.LEARNING_RATE,
                                  weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=8, verbose=False)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_loss   = float('inf')
    best_state_dict = None
    patience_ctr    = 0

    print(f"\n  Training …  (max {Config.MAX_EPOCHS} epochs, "
          f"early-stop patience {Config.PATIENCE})")

    for epoch in range(1, Config.MAX_EPOCHS + 1):

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            train_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_losses  = []
        y_val_true  = []
        y_val_prob  = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
                logits = model(xb)
                loss   = criterion(logits, yb)
                val_losses.append(loss.item())
                probs  = torch.sigmoid(logits).cpu().numpy()
                y_val_true.extend(yb.cpu().numpy().tolist())
                y_val_prob.extend(probs.tolist())

        t_loss   = float(np.mean(train_losses))
        v_loss   = float(np.mean(val_losses))
        try:
            v_auc = roc_auc_score(y_val_true, y_val_prob)
        except Exception:
            v_auc = 0.5

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_auc'].append(v_auc)

        scheduler.step(v_loss)

        # ── Print progress ───────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{Config.MAX_EPOCHS}  |  "
                  f"TrainLoss: {t_loss:.4f}  |  "
                  f"ValLoss: {v_loss:.4f}  |  "
                  f"ValAUC: {v_auc:.4f}  |  "
                  f"LR: {lr_now:.2e}")

        # ── Early stopping ───────────────────────────────────────────────────
        if v_loss < best_val_loss - 1e-5:
            best_val_loss   = v_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr    = 0
        else:
            patience_ctr += 1
            if patience_ctr >= Config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no val-loss improvement for {Config.PATIENCE} epochs)")
                break

    # ── Restore best weights ─────────────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n  Best model restored  (val_loss = {best_val_loss:.4f})")

    model = model.cpu()
    return model, scaler, history


# ==========================================
# EVALUATE MODEL
# ==========================================
def evaluate_model(model, scaler, X_raw, y, split_name="Test"):
    """
    Evaluate MLP on a feature matrix (applies the MLP scaler internally).
    Returns a dict of metrics + probability array.
    """
    feat = X_raw[:, :Config.TOTAL_FEATURE_DIM]
    oh   = X_raw[:, Config.TOTAL_FEATURE_DIM:]

    feat_sc = scaler.transform(feat)
    X_sc    = np.hstack([feat_sc, oh]).astype(np.float32)

    model.eval()
    with torch.no_grad():
        probs = model.predict_probability(
            torch.from_numpy(X_sc)
        ).numpy()                 # (N,)

    preds = (probs >= 0.5).astype(int)

    metrics = {
        'accuracy'  : float(accuracy_score(y, preds)),
        'precision' : float(precision_score(y, preds, zero_division=0)),
        'recall'    : float(recall_score(y, preds, zero_division=0)),
        'f1'        : float(f1_score(y, preds, zero_division=0)),
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y, probs))
    except Exception:
        metrics['roc_auc'] = 0.5

    print(f"\n  [{split_name}]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1']:.4f}")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")

    print(f"\n  Classification Report ({split_name}):")
    report = classification_report(y, preds,
                                   target_names=['Non-neoplastic', 'Neoplastic'],
                                   zero_division=0)
    for line in report.split('\n'):
        print(f"    {line}")

    cm = confusion_matrix(y, preds)
    print(f"\n  Confusion Matrix ({split_name}):")
    print(f"    {'':20s}  Pred: Low  Pred: High")
    print(f"    Actual Low  (0) :   {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"    Actual High (1) :   {cm[1,0]:5d}       {cm[1,1]:5d}")

    return metrics, probs


# ==========================================
# SAVE ARTIFACTS
# ==========================================
def save_artifacts(model, scaler, history, train_metrics, val_metrics, best_epoch):
    """Save MLP weights, scaler, and metadata."""
    print("\n" + "=" * 80)
    print(" " * 30 + "SAVING ARTIFACTS")
    print("=" * 80)

    # Model weights
    model_path = Config.MLP_OUTPUT / 'mlp_model.pth'
    torch.save(model.state_dict(), str(model_path))
    print(f"  Saved model weights  : {model_path.name}")

    # Scaler
    scaler_path = Config.MLP_OUTPUT / 'mlp_scaler.pkl'
    joblib.dump(scaler, str(scaler_path))
    print(f"  Saved MLP scaler     : {scaler_path.name}")

    # Metadata (architecture config + training stats)
    metadata = {
        'architecture': {
            'input_dim'    : Config.MLP_INPUT_DIM,
            'feature_dim'  : Config.TOTAL_FEATURE_DIM,
            'cluster_dim'  : Config.NUM_CLUSTERS,
            'hidden_dims'  : Config.HIDDEN_DIMS,
            'dropout_rates': Config.DROPOUT_RATES,
        },
        'training': {
            'learning_rate' : Config.LEARNING_RATE,
            'weight_decay'  : Config.WEIGHT_DECAY,
            'batch_size'    : Config.BATCH_SIZE,
            'max_epochs'    : Config.MAX_EPOCHS,
            'best_epoch'    : best_epoch,
            'patience'      : Config.PATIENCE,
            'val_split'     : Config.VAL_SPLIT,
        },
        'performance': {
            'train': train_metrics,
            'val'  : val_metrics,
        },
        'asge_thresholds': {
            'high_confidence': Config.ASGE_HIGH_CONFIDENCE,
            'uncertainty'    : Config.ASGE_UNCERTAINTY,
        },
        'timestamp': datetime.now().isoformat(),
    }

    meta_path = Config.MLP_OUTPUT / 'mlp_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata       : {meta_path.name}")

    return str(model_path), str(scaler_path)


# ==========================================
# GENERATE ANALYSIS OUTPUT FILES
# ==========================================
def generate_analysis_outputs(records, y_all, cluster_ids_all,
                               mlp_probs_all, static_prob_map):
    """
    Generate two output files:

    1. ground_truth_analysis.txt  – human-readable per-image GT + feature report
    2. polyp_characteristics.csv  – full per-image feature table with comparisons
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "GENERATING GROUND-TRUTH ANALYSIS OUTPUT FILES")
    print("=" * 80)

    # ── 1. polyp_characteristics.csv ────────────────────────────────────────
    csv_path  = Config.MLP_OUTPUT / 'polyp_characteristics.csv'
    bio_names = [k for k in records[0].keys()
                 if k not in ('image_name', 'ground_truth', 'gt_text',
                               'cluster_id', 'ssl_feat_l2')]

    csv_cols = (['image_name', 'ground_truth', 'gt_text', 'cluster_id',
                 'cluster_static_prob', 'mlp_instance_prob',
                 'mlp_decision', 'cluster_decision',
                 'ssl_feature_l2_norm'] + bio_names)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for i, rec in enumerate(records):
            c_id     = cluster_ids_all[i]
            s_prob   = static_prob_map.get(c_id, 0.5)
            m_prob   = float(mlp_probs_all[i])
            row = {
                'image_name'           : rec['image_name'],
                'ground_truth'         : rec['ground_truth'],
                'gt_text'              : rec['gt_text'],
                'cluster_id'           : c_id,
                'cluster_static_prob'  : round(s_prob, 4),
                'mlp_instance_prob'    : round(m_prob, 4),
                'mlp_decision'         : _asge_decision(m_prob),
                'cluster_decision'     : _asge_decision(s_prob),
                'ssl_feature_l2_norm'  : round(rec['ssl_feat_l2'], 4),
            }
            for bn in bio_names:
                row[bn] = rec.get(bn, '')
            writer.writerow(row)

    print(f"  Saved CSV  : {csv_path}")

    # ── 2. ground_truth_analysis.txt ────────────────────────────────────────
    txt_path = Config.MLP_OUTPUT / 'ground_truth_analysis.txt'

    # Pre-compute cluster-level stats for the report
    cluster_stats = {}
    for k in range(Config.NUM_CLUSTERS):
        idx = [i for i, r in enumerate(records) if r['cluster_id'] == k]
        if not idx:
            cluster_stats[k] = None
            continue
        ys   = [y_all[i]             for i in idx]
        mps  = [float(mlp_probs_all[i]) for i in idx]
        s_p  = static_prob_map.get(k, 0.5)
        cluster_stats[k] = {
            'n'            : len(idx),
            'n_high'       : sum(ys),
            'n_low'        : len(ys) - sum(ys),
            'static_prob'  : round(s_p, 4),
            'mlp_mean'     : round(float(np.mean(mps)), 4),
            'mlp_std'      : round(float(np.std(mps)),  4),
            'mlp_min'      : round(float(np.min(mps)),  4),
            'mlp_max'      : round(float(np.max(mps)),  4),
        }

    with open(txt_path, 'w', encoding='utf-8') as f:
        ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

        f.write("=" * 80 + "\n")
        f.write("     GROUND-TRUTH ANALYSIS & POLYP CHARACTERISTICS REPORT\n")
        f.write("     Phase 4.5 – MLP Probability Calibrator\n")
        f.write(f"     Generated : {ts}\n")
        f.write("=" * 80 + "\n\n")

        f.write("PURPOSE\n")
        f.write("-------\n")
        f.write("This report shows (a) ground-truth labels extracted from NeoPolyp\n")
        f.write("colour masks, (b) per-polyp biomarker characteristics, (c) cluster\n")
        f.write("assignments, and (d) a side-by-side comparison of the legacy\n")
        f.write("cluster-static probability vs. the new MLP instance-level\n")
        f.write("probability for each polyp image.\n\n")

        f.write("GROUND-TRUTH ENCODING (from mask colour)\n")
        f.write("-" * 45 + "\n")
        f.write("  RED   dominant pixels  -->  label 1  (Neoplastic  / High-Risk / Remove)\n")
        f.write("  GREEN dominant pixels  -->  label 0  (Non-neoplastic / Low-Risk / Leave)\n\n")

        total  = len(records)
        n_hi   = int(sum(y_all))
        n_lo   = total - n_hi
        f.write("DATASET SUMMARY\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Total images   : {total:,}\n")
        f.write(f"  Neoplastic     : {n_hi:,}  ({n_hi/max(total,1)*100:.1f}%)\n")
        f.write(f"  Non-neoplastic : {n_lo:,}  ({n_lo/max(total,1)*100:.1f}%)\n\n")

        f.write("CLUSTER-LEVEL COMPARISON  (Static vs MLP)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'Cluster':>7}  {'N':>5}  {'Hi':>5}  {'Lo':>5}  "
                f"{'Static%':>9}  {'MLP Mean':>10}  {'MLP Std':>9}  "
                f"{'MLP Min':>9}  {'MLP Max':>9}\n")
        f.write("  " + "-" * 76 + "\n")
        for k in range(Config.NUM_CLUSTERS):
            cs = cluster_stats.get(k)
            if cs is None:
                f.write(f"  {k:>7}  {'(empty)':>5}\n")
                continue
            f.write(f"  {k:>7}  {cs['n']:>5}  {cs['n_high']:>5}  {cs['n_low']:>5}  "
                    f"  {cs['static_prob']*100:>7.2f}%  "
                    f"  {cs['mlp_mean']*100:>8.2f}%  "
                    f"  {cs['mlp_std']*100:>7.2f}%  "
                    f"  {cs['mlp_min']*100:>7.2f}%  "
                    f"  {cs['mlp_max']*100:>7.2f}%\n")
        f.write("\n")
        f.write("INTERPRETATION:\n")
        f.write("  Static%  = SAME probability given to every polyp in that cluster (old way)\n")
        f.write("  MLP Mean = Average MLP probability for the cluster\n")
        f.write("  MLP Std  = Spread of MLP probabilities within the cluster\n")
        f.write("           A non-zero Std confirms polyps NOW receive unique probabilities.\n\n")

        f.write("PER-IMAGE DETAILS  (first 200 shown for brevity)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'Image':<50}  {'GT':>4}  {'Cluster':>7}  "
                f"{'Static%':>8}  {'MLP%':>7}  {'ASGE Decision'}\n")
        f.write("  " + "-" * 100 + "\n")

        for i, rec in enumerate(records[:200]):
            c_id   = cluster_ids_all[i]
            s_prob = static_prob_map.get(c_id, 0.5)
            m_prob = float(mlp_probs_all[i])
            gt_ch  = 'H' if rec['ground_truth'] == 1 else 'L'
            f.write(f"  {rec['image_name']:<50}  {gt_ch:>4}  {c_id:>7}  "
                    f"  {s_prob*100:>6.2f}%  {m_prob*100:>5.2f}%  "
                    f"{_asge_decision(m_prob)}\n")

        if len(records) > 200:
            f.write(f"\n  ... ({len(records) - 200} more rows in polyp_characteristics.csv)\n")

        f.write("\n")
        f.write("KEY BIOMARKER STATISTICS ACROSS CLASSES\n")
        f.write("=" * 80 + "\n")
        _write_biomarker_stats(f, records, y_all)

        f.write("\nEND OF REPORT\n")

    print(f"  Saved TXT  : {txt_path}")
    return str(csv_path), str(txt_path)


def _asge_decision(prob):
    if prob >= Config.ASGE_HIGH_CONFIDENCE:
        return 'HIGH RISK    (Resect & Discard)'
    elif prob >= Config.ASGE_UNCERTAINTY:
        return 'UNCERTAIN    (Require Biopsy)'
    else:
        return 'LOW RISK     (Surveillance)'


def _write_biomarker_stats(f, records, y_all):
    """Write mean biomarker values split by ground-truth class."""
    bio_fields = ['Texture_Complexity', 'Relative_Area', 'Compactness',
                  'LAB_a_bin0', 'LAB_a_bin1', 'LAB_a_bin2',
                  'Haralick_0', 'Haralick_1', 'Haralick_2',
                  'ssl_feat_l2']

    f.write(f"\n  {'Biomarker':<28}  {'Non-neoplastic Mean':>20}  {'Neoplastic Mean':>16}\n")
    f.write("  " + "-" * 68 + "\n")

    for field in bio_fields:
        vals_lo = [rec.get(field, 0) for i, rec in enumerate(records) if y_all[i] == 0]
        vals_hi = [rec.get(field, 0) for i, rec in enumerate(records) if y_all[i] == 1]
        m_lo = float(np.mean(vals_lo)) if vals_lo else 0.0
        m_hi = float(np.mean(vals_hi)) if vals_hi else 0.0
        f.write(f"  {field:<28}  {m_lo:>20.6f}  {m_hi:>16.6f}\n")

    f.write("\n  (Full 60-dim biomarker values available in polyp_characteristics.csv)\n")


# ==========================================
# VISUALIZATIONS
# ==========================================
def generate_visualizations(history, y_all, cluster_ids_all,
                             mlp_probs_all, static_prob_map):
    print("\n" + "=" * 80)
    print(" " * 30 + "GENERATING VISUALIZATIONS")
    print("=" * 80)

    # ── 1. Training curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss',  color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',    color='orange')
    axes[0].set_title('MLP Training / Validation Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch');  axes[0].set_ylabel('BCE Loss')
    axes[0].legend();             axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['val_auc'], label='Val ROC-AUC', color='green')
    axes[1].set_title('Validation ROC-AUC over Training', fontweight='bold')
    axes[1].set_xlabel('Epoch');  axes[1].set_ylabel('ROC-AUC')
    axes[1].set_ylim([0, 1]);     axes[1].legend();  axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = Config.VISUAL_OUTPUT / 'training_curves.png'
    plt.savefig(str(p), dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved : training_curves.png")

    # ── 2. Calibration comparison (MLP vs static, per polyp) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    static_probs = np.array([static_prob_map.get(int(c), 0.5)
                              for c in cluster_ids_all])

    # Side-by-side scatter (ground-truth color coded)
    colors = np.where(y_all == 1, 'red', 'green')
    axes[0].scatter(static_probs, mlp_probs_all, c=colors, alpha=0.5, s=20)
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Identity')
    axes[0].axhline(Config.ASGE_HIGH_CONFIDENCE, color='red',    linestyle=':', linewidth=1.5,
                    label=f'ASGE High ({Config.ASGE_HIGH_CONFIDENCE})')
    axes[0].axhline(Config.ASGE_UNCERTAINTY,     color='orange', linestyle=':', linewidth=1.5,
                    label=f'ASGE Unc. ({Config.ASGE_UNCERTAINTY})')
    axes[0].set_title('Static Cluster Prob vs MLP Instance Prob\n(Red=Neoplastic, Green=Non-neoplastic)',
                       fontweight='bold')
    axes[0].set_xlabel('Cluster-Static Probability')
    axes[0].set_ylabel('MLP Instance Probability')
    axes[0].legend(fontsize=8);  axes[0].grid(alpha=0.3)

    # Per-cluster variance comparison
    cluster_list  = list(range(Config.NUM_CLUSTERS))
    static_stds   = [0.0] * Config.NUM_CLUSTERS   # always 0 – static
    mlp_stds = []
    for k in cluster_list:
        idx = np.where(cluster_ids_all == k)[0]
        mlp_stds.append(float(np.std(mlp_probs_all[idx])) if len(idx) > 0 else 0.0)

    x = np.arange(len(cluster_list))
    axes[1].bar(x - 0.2, static_stds, 0.35, label='Static STD (always 0)', color='gray',  alpha=0.8)
    axes[1].bar(x + 0.2, mlp_stds,    0.35, label='MLP STD (>0 = instance-level)', color='steelblue', alpha=0.8)
    axes[1].set_title('Probability Spread Within Each Cluster\n(non-zero MLP STD = unique probabilities)',
                       fontweight='bold')
    axes[1].set_xlabel('Cluster ID');  axes[1].set_ylabel('Std Dev of Probability')
    axes[1].set_xticks(x);  axes[1].set_xticklabels([f'C{k}' for k in cluster_list])
    axes[1].legend();  axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = Config.VISUAL_OUTPUT / 'calibration_comparison.png'
    plt.savefig(str(p), dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved : calibration_comparison.png")

    # ── 3. Probability distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, probs, title in [
        (axes[0], static_probs, 'Cluster-Static Probability Distribution'),
        (axes[1], mlp_probs_all, 'MLP Instance Probability Distribution'),
    ]:
        ax.hist(probs[y_all == 0], bins=30, alpha=0.6, color='green',
                label='Non-neoplastic (GT=0)', density=True)
        ax.hist(probs[y_all == 1], bins=30, alpha=0.6, color='red',
                label='Neoplastic (GT=1)',     density=True)
        ax.axvline(Config.ASGE_HIGH_CONFIDENCE, color='darkred',
                    linestyle='--', linewidth=1.5,
                    label=f'ASGE High ({Config.ASGE_HIGH_CONFIDENCE})')
        ax.axvline(Config.ASGE_UNCERTAINTY, color='orange',
                    linestyle='--', linewidth=1.5,
                    label=f'Uncertainty ({Config.ASGE_UNCERTAINTY})')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Probability');  ax.set_ylabel('Density')
        ax.legend(fontsize=8);         ax.grid(alpha=0.3)

    plt.tight_layout()
    p = Config.VISUAL_OUTPUT / 'probability_distribution.png'
    plt.savefig(str(p), dpi=150, bbox_inches='tight');  plt.close()
    print(f"  Saved : probability_distribution.png")


# ==========================================
# LOAD CLUSTERING ARTIFACTS (Phase 3)
# ==========================================
def load_clustering_artifacts():
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING PHASE-3 CLUSTERING ARTIFACTS")
    print("=" * 80)

    kmeans_path = Config.FEATURES_OUTPUT / 'kmeans_model.pkl'
    scaler_path = Config.FEATURES_OUTPUT / 'feature_scaler.pkl'

    if not kmeans_path.exists():
        print(f"  ERROR: K-Means model not found at {kmeans_path}")
        print("         Run Phase 3 first.")
        return None, None

    if not scaler_path.exists():
        print(f"  ERROR: Feature scaler not found at {scaler_path}")
        print("         Run Phase 3 first.")
        return None, None

    kmeans = joblib.load(str(kmeans_path))
    scaler = joblib.load(str(scaler_path))
    print(f"  K-Means loaded  ({kmeans.n_clusters} clusters)")
    print(f"  Feature scaler loaded")
    return kmeans, scaler


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"\n  Started : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")

    # ── Step 1: Load Phase-3 clustering artifacts ────────────────────────────
    kmeans, kmeans_scaler = load_clustering_artifacts()
    if kmeans is None:
        return

    # ── Step 2: Load GT labels ───────────────────────────────────────────────
    labels_dict = load_neopolyp_labels()
    if not labels_dict:
        print("  ERROR: No ground-truth labels available.")
        return

    # ── Step 3: Extract features + cluster assignments ───────────────────────
    records, X_all, y_all = extract_neopolyp_features(labels_dict, kmeans, kmeans_scaler)
    if X_all is None:
        return

    # Cluster IDs are stored in records
    cluster_ids_all = np.array([r['cluster_id'] for r in records])

    # ── Step 4: Compute cluster-static probabilities (legacy baseline) ────────
    # use only the 444-dim feature block – cluster_ids already known
    static_prob_map = compute_cluster_static_probs(y_all, cluster_ids_all)

    print("\n  Legacy cluster-static probabilities:")
    for k, p in static_prob_map.items():
        n_k = int((cluster_ids_all == k).sum())
        print(f"    Cluster {k}: {n_k:4d} samples  ->  static prob = {p*100:.1f}%")

    # ── Step 5: Train / val split ────────────────────────────────────────────
    idx = np.arange(len(y_all))
    tr_idx, va_idx = train_test_split(idx, test_size=Config.VAL_SPLIT,
                                       random_state=Config.RANDOM_SEED,
                                       stratify=y_all if len(np.unique(y_all)) > 1 else None)

    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val,   y_val   = X_all[va_idx], y_all[va_idx]

    print(f"\n  Train set: {len(y_train)} samples  |  Val set: {len(y_val)} samples")

    # ── Step 6: Train MLP ────────────────────────────────────────────────────
    model, mlp_scaler, history = train_mlp(X_train, y_train, X_val, y_val)

    best_epoch = int(np.argmin(history['val_loss'])) + 1

    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL EVALUATION")
    print("=" * 80)

    train_metrics, _         = evaluate_model(model, mlp_scaler, X_train, y_train, "Train")
    val_metrics,   val_probs = evaluate_model(model, mlp_scaler, X_val,   y_val,   "Validation")

    # Full-dataset probabilities for analysis files
    _, all_mlp_probs = evaluate_model(model, mlp_scaler, X_all, y_all, "Full Dataset")

    # ── Step 7: Save weights + metadata ─────────────────────────────────────
    save_artifacts(model, mlp_scaler, history, train_metrics, val_metrics, best_epoch)

    # ── Step 8: Generate analysis output files ───────────────────────────────
    generate_analysis_outputs(records, y_all, cluster_ids_all,
                               all_mlp_probs, static_prob_map)

    # ── Step 9: Visualizations ───────────────────────────────────────────────
    generate_visualizations(history, y_all, cluster_ids_all,
                             all_mlp_probs, static_prob_map)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" " * 25 + "PHASE 4.5 COMPLETE!")
    print("=" * 80)
    print(f"\n  MLP Architecture  : {Config.MLP_INPUT_DIM} → "
          f"{' → '.join(str(h) for h in Config.HIDDEN_DIMS)} → 1")
    print(f"  Input encoding    : 444-dim features  +  8-dim one-hot cluster ID")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Val ROC-AUC       : {val_metrics['roc_auc']:.4f}")
    print(f"  Val F1-Score      : {val_metrics['f1']:.4f}")

    print(f"\n  KEY BENEFIT:")
    print(f"  Before (cluster-static): every polyp in cluster k got the SAME probability")
    print(f"  After  (MLP instance)  : every polyp gets a UNIQUE probability based on its")
    print(f"                           444-dim fingerprint (SSL + biomarkers) + cluster context")

    print(f"\n  Output files:")
    print(f"    {Config.MLP_OUTPUT / 'mlp_model.pth'}")
    print(f"    {Config.MLP_OUTPUT / 'mlp_scaler.pkl'}")
    print(f"    {Config.MLP_OUTPUT / 'mlp_metadata.json'}")
    print(f"    {Config.MLP_OUTPUT / 'ground_truth_analysis.txt'}")
    print(f"    {Config.MLP_OUTPUT / 'polyp_characteristics.csv'}")
    print(f"    {Config.VISUAL_OUTPUT / 'training_curves.png'}")
    print(f"    {Config.VISUAL_OUTPUT / 'calibration_comparison.png'}")
    print(f"    {Config.VISUAL_OUTPUT / 'probability_distribution.png'}")

    print(f"\n  Next step: Run Inference (Phase Inf) – the MLP will now provide")
    print(f"             instance-level probabilities instead of cluster-static ones.")


if __name__ == '__main__':
    main()
