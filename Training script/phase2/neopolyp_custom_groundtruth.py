# -*- coding: utf-8 -*-
"""
PHASE 2C: NeoPolyp Custom Ground Truth Generation
==================================================
Generates purely data-driven ground truth thresholds from the NeoPolyp dataset
by extracting dense visual feature vectors via the SimCLR-trained ViT encoder
from Phase 2A.  No external clinical guidelines (ASGE) are used at any point.

Pipeline Position
-----------------
  Phase 2A  →  ssl_training.py          (train ViT-Small SimCLR)
  Phase 2B  →  neopolyp_preparation.py  (validate colour-coded masks)
  Phase 2C  →  neopolyp_custom_groundtruth.py  ← THIS FILE
  Phase 3   →  feature_extraction.py    (cluster all unlabelled ROIs)
  Phase 4   →  mixture_of_experts.py    (train per-cluster Decision Trees)

To add this phase to the master pipeline runner without editing master_run.py,
insert the following entry into the PHASES list in master_run.py between the
'2b' and '3' entries:

    {
        'id'         : '2c',
        'name'       : 'NeoPolyp Custom Ground Truth Generation',
        'script'     : 'phase2/neopolyp_custom_groundtruth.py',
        'required'   : True,
        'description': (
            'Extract 444-dim ViT+biomarker features from NeoPolyp, '
            'compute per-class statistics, and derive data-driven risk '
            'thresholds (replaces ASGE for probability calibration).'
        ),
    },

Approach
--------
1. Pass every NeoPolyp training image through the Phase-2A SSL encoder.
2. Extract a 444-dim Fact Vector per image:
     Stream A — 384-dim ViT-Small CLS token  (Neural / SSL)
     Stream B —  60-dim symbolic biomarkers  (colour, texture, shape)
3. Parse colour-coded ground truth masks:
     RED  pixels  →  High Risk  (label = 1)
     GREEN pixels →  Low Risk   (label = 0)
4. For each of the 444 features compute:
     • Per-class descriptive statistics (mean, std, median, IQR, min, max)
     • Cohen's d  (signed effect size: positive ↑ = higher in High Risk)
     • Youden's J optimal threshold  (maximises sensitivity + specificity − 1)
     • Welch t-test p-value
5. Rank features by |Cohen's d|  →  Top-N discriminative "ground truth rules".
6. Build a weighted composite risk score from the top rules and calibrate
   high-confidence and uncertainty probability thresholds from the data.

Outputs  (thesis_outputs/custom_groundtruth/)
--------------------------------------------
  neopolyp_feature_log.csv   — one row per image: label + risk score + all 444 features
  custom_groundtruths.json   — ground truth rules + calibration thresholds
  feature_statistics.json    — full per-feature statistics for all 444 dimensions
  gt_top10_features.png      — box-plot comparison of the 10 most discriminative features
  gt_risk_score_dist.png     — composite risk score histogram per class
  gt_cohen_d_ranking.png     — horizontal bar chart of top-20 effect sizes
"""

# =============================================================================
# SYSTEM & ENV SETUP
# =============================================================================
import os

# Prevent Windows OMP deadlock (same guards as Phase 3 / Phase 4)
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import torch
import torch.nn as nn
from torchvision import transforms
import timm

print("=" * 80)
print(" " * 10 + "PHASE 2C: NEOPOLYP CUSTOM GROUND TRUTH GENERATION")
print(" " * 6 + "(Data-Driven Risk Thresholds via SimCLR ViT — No ASGE)")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # ---- Path resolution ----------------------------------------------------
    # This file lives at  Training script/phase2/neopolyp_custom_groundtruth.py
    # → parent            = Training script/phase2/
    # → parent.parent     = Training script/
    # → parent.parent.parent = Thesis V5/   (THESIS_ROOT)
    THESIS_ROOT     = Path(__file__).parent.parent.parent.absolute()

    # ---- NeoPolyp dataset ---------------------------------------------------
    NEOPOLYP_ROOT   = THESIS_ROOT / 'NeSy' / 'Neo polyp Dataset'
    TRAIN_IMAGES    = NEOPOLYP_ROOT / 'train'    / 'train'
    TRAIN_MASKS     = NEOPOLYP_ROOT / 'train_gt' / 'train_gt'

    # ---- Phase 2A SSL checkpoint -------------------------------------------
    OUTPUT_ROOT     = THESIS_ROOT / 'thesis_outputs'
    SSL_OUTPUT      = OUTPUT_ROOT  / 'ssl_outputs'

    # ---- This phase's outputs -----------------------------------------------
    GT_OUTPUT       = OUTPUT_ROOT  / 'custom_groundtruth'
    VISUAL_OUTPUT   = OUTPUT_ROOT  / 'visualizations'

    # ---- Image resolution (must match Phase 2A training size) ---------------
    IMG_SIZE        = 256

    # ---- Hardware -----------------------------------------------------------
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Feature dimensions (must match Phase 3 layout exactly) -------------
    SSL_FEATURE_DIM   = 384   # ViT-Small CLS token
    BIOMARKER_DIM     =  60   # Symbolic biomarkers
    TOTAL_FEATURE_DIM = 444   # 384 + 60

    # ---- Mask colour thresholds (OpenCV BGR space) --------------------------
    # Red  mask: R > 180, G < 60, B < 60
    # Green mask: G > 180, R < 60, B < 60
    RED_R_MIN   = 180
    RED_G_MAX   =  60
    RED_B_MAX   =  60
    GREEN_G_MIN = 180
    GREEN_R_MAX =  60
    GREEN_B_MAX =  60

    # ---- Statistical analysis parameters ------------------------------------
    TOP_N_DISCRIMINATIVE = 20     # Maximum rules to include in ground truth output
    MIN_COHEN_D          = 0.30   # Minimum |d| to qualify as a meaningful rule
    N_THRESHOLD_STEPS    = 1000   # Youden-J threshold search resolution


# Create output directories
Config.GT_OUTPUT.mkdir(parents=True, exist_ok=True)
Config.VISUAL_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n  Thesis Root : {Config.THESIS_ROOT}")
print(f"  NeoPolyp    : {Config.TRAIN_IMAGES}")
print(f"  SSL Output  : {Config.SSL_OUTPUT}")
print(f"  GT Output   : {Config.GT_OUTPUT}")
print(f"  Device      : {Config.DEVICE}")

# =============================================================================
# FEATURE NAMES
# (must align with Phase 3's extract_biomarkers() layout: 9+16+13+19+3 = 60)
# =============================================================================
_BIOMARKER_NAMES = (
    # CIE-LAB histograms  — 3 bins × 3 channels = 9
    ['lab_L_dark',      'lab_L_mid',      'lab_L_bright'     ] +
    ['lab_a_green_dir', 'lab_a_neutral',  'lab_a_red_dir'    ] +
    ['lab_b_blue_dir',  'lab_b_neutral',  'lab_b_yellow_dir' ] +
    # Saturation histogram — 16 bins
    [f'sat_bin_{i}'              for i in range(16)           ] +
    # Haralick GLCM — 4 directions × 3 properties + 1 energy = 13
    [f'haralick_contrast_{a}'     for a in ['0', '45', '90', '135']] +
    [f'haralick_dissimilarity_{a}' for a in ['0', '45', '90', '135']] +
    [f'haralick_homogeneity_{a}'  for a in ['0', '45', '90', '135']] +
    ['haralick_energy_mean'                                   ] +
    # LBP histogram — 19 uniform-pattern bins
    [f'lbp_bin_{i}'              for i in range(19)           ] +
    # Shape — 3
    ['shape_edge_density', 'shape_relative_area', 'shape_compactness']
)

assert len(_BIOMARKER_NAMES) == 60, (
    f"Biomarker name list length mismatch: expected 60, got {len(_BIOMARKER_NAMES)}"
)

# All 444 feature names in order  [SSL_0 … SSL_383 | biomarker_0 … biomarker_59]
ALL_FEATURE_NAMES = [f'ssl_feat_{i}' for i in range(Config.SSL_FEATURE_DIM)] + _BIOMARKER_NAMES


# =============================================================================
# VIT ENCODER
# (Identical architecture to Phase 2A / Phase 3 — reimplemented here to avoid
#  importing from those files, which execute code at module level.)
# =============================================================================
class ViTEncoder(nn.Module):
    """ViT-Small with 256×256 input, no classification head (returns 384-dim CLS)."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0,   # return CLS token directly
            img_size=Config.IMG_SIZE,
        )

    def forward(self, x):
        return self.backbone(x)   # → (B, 384)


def load_ssl_model() -> nn.Module:
    """
    Load the Phase 2A checkpoint from thesis_outputs/ssl_outputs/.
    Mirrors Phase 3's loading logic exactly.
    Falls back to an untrained model with a warning if no checkpoint is found.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "LOADING SSL ENCODER  (Phase 2A)")
    print("=" * 80)

    candidates = [
        Config.SSL_OUTPUT / 'ssl_encoder_final.pth',
        Config.SSL_OUTPUT / 'ssl_model_final.pth',
        Config.SSL_OUTPUT / 'best_ssl_model.pth',
    ]

    model = ViTEncoder().to(Config.DEVICE)

    for ckpt_path in candidates:
        if not ckpt_path.exists():
            continue
        try:
            state = torch.load(str(ckpt_path), map_location=Config.DEVICE)
            if isinstance(state, dict):
                # Full model saved (keys start with 'backbone.')
                if any(k.startswith('backbone.') for k in state.keys()):
                    backbone_state = {
                        k.replace('backbone.', ''): v
                        for k, v in state.items()
                        if k.startswith('backbone.')
                    }
                    model.backbone.load_state_dict(backbone_state)
                else:
                    model.backbone.load_state_dict(state)
            else:
                model.load_state_dict(state)
            print(f"  [OK] SSL checkpoint loaded: {ckpt_path.name}")
            model.eval()
            return model
        except Exception as exc:
            print(f"  [WARN] Could not load {ckpt_path.name}: {exc}")
            continue

    print("  [WARN] No Phase 2A SSL checkpoint found in:")
    print(f"         {Config.SSL_OUTPUT}")
    print("  [WARN] Using randomly-initialised ViT — run Phase 2A first for")
    print("         meaningful SSL features.  Biomarker features will still")
    print("         be computed correctly.")
    model.eval()
    return model


# =============================================================================
# IMAGE UTILITIES
# =============================================================================
def pad_to_square(image: Image.Image, target_size: int = 256) -> Image.Image:
    """
    Letterbox-pad to a square canvas while preserving aspect ratio.
    Identical to Phase 3's pad_to_square() function.
    """
    ratio    = float(target_size) / max(image.size)
    new_wh   = tuple(int(d * ratio) for d in image.size)
    resized  = image.resize(new_wh, Image.LANCZOS)
    canvas   = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(resized, ((target_size - new_wh[0]) // 2,
                            (target_size - new_wh[1]) // 2))
    return canvas


def build_transform(img_size: int = 256) -> transforms.Compose:
    """Inference-only transform (no augmentation); matches Phase 3."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# MASK PARSING
# =============================================================================
def parse_mask_label(mask_path: Path):
    """
    Read a colour-coded NeoPolyp ground truth mask and determine risk label.

    Parameters
    ----------
    mask_path : Path

    Returns
    -------
    label       : int   1 = High Risk (red),  0 = Low Risk (green),  -1 = invalid
    red_ratio   : float fraction of pixels classified as red
    green_ratio : float fraction of pixels classified as green
    """
    bgr = cv2.imread(str(mask_path))
    if bgr is None:
        return -1, 0.0, 0.0

    total = bgr.shape[0] * bgr.shape[1]

    # OpenCV BGR channels: index 2 = R, index 1 = G, index 0 = B
    r_ch, g_ch, b_ch = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]

    red_px   = int(np.sum((r_ch > Config.RED_R_MIN)   &
                           (g_ch < Config.RED_G_MAX)   &
                           (b_ch < Config.RED_B_MAX)))
    green_px = int(np.sum((g_ch > Config.GREEN_G_MIN) &
                           (r_ch < Config.GREEN_R_MAX) &
                           (b_ch < Config.GREEN_B_MAX)))

    red_ratio   = red_px   / total
    green_ratio = green_px / total

    if red_px == 0 and green_px == 0:
        # No primary colour found — fall back to dominant channel average
        if np.mean(r_ch) >= np.mean(g_ch):
            return 1, red_ratio, green_ratio   # Inferred high-risk
        else:
            return 0, red_ratio, green_ratio   # Inferred low-risk

    # Safety rule: when ambiguous, prefer High Risk label
    label = 1 if red_px >= green_px else 0
    return label, red_ratio, green_ratio


# =============================================================================
# BIOMARKER EXTRACTION  (Stream B — Symbolic)
# Reimplemented from Phase 3 without importing that module.
# The vector layout MUST match feature_extraction.py:extract_biomarkers() exactly.
# =============================================================================
def _cielab_histograms(img: np.ndarray) -> np.ndarray:
    """9-dim CIE-LAB colour histograms (3 bins per channel × 3 channels)."""
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    feat = []
    for ch in cv2.split(lab):
        h = cv2.calcHist([ch], [0], None, [3], [0, 256]).flatten()
        feat.append(h / (h.sum() + 1e-7))
    return np.concatenate(feat)    # (9,)


def _saturation_histogram(img: np.ndarray) -> np.ndarray:
    """16-dim HSV saturation histogram."""
    s = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[1]
    h = cv2.calcHist([s], [0], None, [16], [0, 256]).flatten()
    return h / (h.sum() + 1e-7)    # (16,)


def _haralick_features(img: np.ndarray) -> np.ndarray:
    """13-dim Haralick texture features from GLCM (4 directions, 16 grey levels)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g16  = (gray // 16).astype(np.uint8)   # reduce to 16 levels
    try:
        glcm = graycomatrix(
            g16, distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16, symmetric=True, normed=True,
        )
        contrast     = graycoprops(glcm, 'contrast').flatten()       # 4
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()  # 4
        homogeneity  = graycoprops(glcm, 'homogeneity').flatten()    # 4
        energy       = graycoprops(glcm, 'energy').flatten()         # 4 → take mean

        return np.concatenate(
            [contrast, dissimilarity, homogeneity, [energy.mean()]]
        )[:13].astype(np.float32)                                    # (13,)
    except Exception:
        return np.zeros(13, dtype=np.float32)


def _lbp_histogram(img: np.ndarray) -> np.ndarray:
    """19-dim LBP (P=18, R=2, uniform) histogram."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=18, R=2, method='uniform')
    h, _ = np.histogram(lbp.ravel(), bins=20, range=(0, 20))
    h    = h.astype(np.float32)
    return (h / (h.sum() + 1e-7))[:19]                              # (19,)


def _shape_features(img: np.ndarray) -> np.ndarray:
    """3-dim shape features: edge density, relative area, compactness."""
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-9)

    _, s_ch, v_ch = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    rough_mask    = ((s_ch > 30) | (v_ch < 200)).astype(np.uint8)
    rel_area      = float(np.mean(rough_mask))

    contours, _ = cv2.findContours(
        rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    compactness = 0.0
    if contours:
        lc         = max(contours, key=cv2.contourArea)
        area_cnt   = cv2.contourArea(lc)
        perimeter  = cv2.arcLength(lc, True)
        compactness = (4 * np.pi * area_cnt) / (perimeter ** 2 + 1e-7)

    return np.array([edge_density, rel_area, compactness], dtype=np.float32)


def extract_biomarkers(img_np: np.ndarray) -> np.ndarray:
    """
    Build the 60-dimensional symbolic biomarker vector.
    Layout: 9 CIE-LAB + 16 saturation + 13 Haralick + 19 LBP + 3 shape.
    Must remain identical to Phase 3's extract_biomarkers().
    """
    vec = np.concatenate([
        _cielab_histograms(img_np),     #  9
        _saturation_histogram(img_np),  # 16
        _haralick_features(img_np),     # 13
        _lbp_histogram(img_np),         # 19
        _shape_features(img_np),        #  3
    ]).astype(np.float32)
    assert len(vec) == 60, f"Biomarker vector length mismatch: {len(vec)}"
    return vec


# =============================================================================
# DUAL-STREAM FEATURE EXTRACTION  (444-dim Fact Vector)
# =============================================================================
@torch.no_grad()
def extract_fact_vector(image_path: Path,
                         ssl_model: nn.Module,
                         transform: transforms.Compose) -> np.ndarray:
    """
    Extract a 444-dim Fact Vector for a single image.
      [0:384]  Stream A — ViT-Small SSL features (neural)
      [384:]   Stream B — symbolic biomarkers
    """
    img_pil   = Image.open(image_path).convert('RGB')
    img_pil   = pad_to_square(img_pil, Config.IMG_SIZE)
    img_np    = np.array(img_pil)

    # Stream A: Neural (SSL ViT)
    tensor    = transform(img_pil).unsqueeze(0).to(Config.DEVICE)
    ssl_feats = ssl_model(tensor).cpu().numpy().flatten()   # (384,)

    # Stream B: Symbolic biomarkers
    biomarkers = extract_biomarkers(img_np)                 # (60,)

    fact_vec   = np.concatenate([ssl_feats, biomarkers])    # (444,)
    return fact_vec


# =============================================================================
# DATASET COLLECTION
# =============================================================================
def collect_dataset():
    """
    Match NeoPolyp training images to their colour-coded ground truth masks.
    Returns list of (image_path, mask_path, label, red_ratio, green_ratio).
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "COLLECTING NEOPOLYP DATASET")
    print("=" * 80)

    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files: list = []
    for ext in img_exts:
        image_files.extend(Config.TRAIN_IMAGES.glob(ext))
    image_files = sorted(set(image_files))
    print(f"  Found {len(image_files):,} images in {Config.TRAIN_IMAGES}")

    records = []
    skipped = 0

    for img_path in image_files:
        # Find matching mask — same stem, any supported extension
        mask_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = Config.TRAIN_MASKS / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            skipped += 1
            continue

        label, red_ratio, green_ratio = parse_mask_label(mask_path)
        if label == -1:
            skipped += 1
            continue

        records.append((img_path, mask_path, label, red_ratio, green_ratio))

    n_high = sum(1 for r in records if r[2] == 1)
    n_low  = sum(1 for r in records if r[2] == 0)

    print(f"  Valid pairs : {len(records):,}  (skipped / invalid: {skipped})")
    print(f"  High Risk   : {n_high:,}  ({n_high / max(len(records), 1) * 100:.1f}%)")
    print(f"  Low Risk    : {n_low:,}  ({n_low  / max(len(records), 1) * 100:.1f}%)")

    return records


# =============================================================================
# BATCH FEATURE EXTRACTION
# =============================================================================
def extract_all_features(records, ssl_model, transform):
    """
    Run dual-stream extraction on every collected record.

    Returns
    -------
    features_matrix : np.ndarray  shape (N, 444)
    labels          : np.ndarray  shape (N,)   int32, {0, 1}
    image_names     : list[str]   length N
    meta_list       : list[dict]  per-image metadata
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "EXTRACTING DUAL-STREAM FEATURES  (444-dim)")
    print("=" * 80)
    print(f"  {len(records):,} images on {Config.DEVICE}\n")

    features_list, labels_list, names_list, meta_list = [], [], [], []
    failed: list = []

    for img_path, mask_path, label, red_ratio, green_ratio in tqdm(
        records, ncols=90, desc="Extracting"
    ):
        try:
            feat_vec = extract_fact_vector(img_path, ssl_model, transform)
            features_list.append(feat_vec)
            labels_list.append(label)
            names_list.append(img_path.name)
            meta_list.append({
                'image_name'      : img_path.name,
                'label'           : int(label),
                'risk_category'   : 'High Risk' if label == 1 else 'Low Risk',
                'red_mask_ratio'  : float(red_ratio),
                'green_mask_ratio': float(green_ratio),
            })
        except Exception as exc:
            failed.append((img_path.name, str(exc)))

    if failed:
        print(f"\n  [WARN] Failed on {len(failed)} image(s):")
        for name, err in failed[:5]:
            print(f"    {name}: {err}")

    features_matrix = np.array(features_list, dtype=np.float32)  # (N, 444)
    labels          = np.array(labels_list,   dtype=np.int32)     # (N,)

    print(f"\n  Feature matrix : {features_matrix.shape}")
    print(f"  High Risk      : {int(labels.sum())}")
    print(f"  Low Risk       : {int((labels == 0).sum())}")

    return features_matrix, labels, names_list, meta_list


# =============================================================================
# STATISTICAL ANALYSIS  — per feature, per class
# =============================================================================
def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Signed Cohen's d.
    Positive ↑  means feature is on average HIGHER in group `a` than `b`.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-9))


def _youden_threshold(low_vals: np.ndarray,
                      high_vals: np.ndarray,
                      n_steps: int = 1000):
    """
    Scan `n_steps` candidate thresholds and return the one that maximises
    Youden's J  =  Sensitivity + Specificity − 1.

    Returns
    -------
    best_t   : float  optimal threshold value
    direction: str    'above' → feature > threshold predicts High Risk
                      'below' → feature < threshold predicts High Risk
    j_score  : float  Youden's J at best_t  (0 = random, 1 = perfect)
    """
    all_vals = np.concatenate([low_vals, high_vals])
    lo, hi   = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    if lo == hi:
        return float(lo), 'above', 0.0

    thresholds = np.linspace(lo, hi, n_steps)
    n_low, n_high = len(low_vals), len(high_vals)
    best_j, best_t, best_dir = -np.inf, float(lo), 'above'

    for t in thresholds:
        # Direction A: feature > t  →  High Risk
        sens_a = np.sum(high_vals > t) / (n_high + 1e-9)
        spec_a = np.sum(low_vals  <= t) / (n_low  + 1e-9)
        j_a    = sens_a + spec_a - 1

        # Direction B: feature < t  →  High Risk
        sens_b = np.sum(high_vals < t) / (n_high + 1e-9)
        spec_b = np.sum(low_vals  >= t) / (n_low  + 1e-9)
        j_b    = sens_b + spec_b - 1

        if j_a >= j_b and j_a > best_j:
            best_j, best_t, best_dir = j_a, float(t), 'above'
        elif j_b > j_a and j_b > best_j:
            best_j, best_t, best_dir = j_b, float(t), 'below'

    return best_t, best_dir, float(best_j)


def compute_class_statistics(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    For every one of the 444 features compute descriptive statistics
    per class (High / Low Risk) plus inter-class analysis.

    Returns
    -------
    stats : dict   keyed by feature name, value is a nested dict
    """
    high_feats = features[labels == 1]   # (N_high, 444)
    low_feats  = features[labels == 0]   # (N_low,  444)
    n_feats    = features.shape[1]

    print(f"\n  Computing statistics for {n_feats} features …")

    stats: dict = {}

    for i in tqdm(range(n_feats), ncols=80, leave=False, desc="Stats"):
        h = high_feats[:, i]
        l = low_feats[:, i]

        t_stat, p_val = scipy_stats.ttest_ind(h, l, equal_var=False)
        d             = _cohen_d(h, l)
        threshold, direction, j_score = _youden_threshold(
            l, h, Config.N_THRESHOLD_STEPS
        )

        stats[ALL_FEATURE_NAMES[i]] = {
            'feature_index': int(i),
            'high_risk': {
                'n'     : int(len(h)),
                'mean'  : float(np.mean(h)),
                'std'   : float(np.std(h,  ddof=1)),
                'median': float(np.median(h)),
                'q25'   : float(np.percentile(h, 25)),
                'q75'   : float(np.percentile(h, 75)),
                'min'   : float(np.min(h)),
                'max'   : float(np.max(h)),
            },
            'low_risk': {
                'n'     : int(len(l)),
                'mean'  : float(np.mean(l)),
                'std'   : float(np.std(l,  ddof=1)),
                'median': float(np.median(l)),
                'q25'   : float(np.percentile(l, 25)),
                'q75'   : float(np.percentile(l, 75)),
                'min'   : float(np.min(l)),
                'max'   : float(np.max(l)),
            },
            'inter_class': {
                'cohen_d'         : float(d),
                'abs_cohen_d'     : float(abs(d)),
                'welch_t'         : float(t_stat),
                'p_value'         : float(p_val),
                'significant_p05' : bool(p_val < 0.05),
                'youden_threshold': float(threshold),
                'risk_direction'  : direction,
                'youden_j_score'  : float(j_score),
            },
        }

    return stats


# =============================================================================
# GROUND TRUTH RULE GENERATION
# =============================================================================
def generate_ground_truth_rules(class_stats: dict) -> list:
    """
    Select the most discriminative features, ranked by |Cohen's d|,
    and express each as an interpretable numerical threshold rule.

    Features with |Cohen's d| < MIN_COHEN_D are excluded.
    At most TOP_N_DISCRIMINATIVE rules are returned.
    """
    # Sort all features by descending absolute effect size
    ranked = sorted(
        class_stats.items(),
        key=lambda kv: kv[1]['inter_class']['abs_cohen_d'],
        reverse=True,
    )

    rules: list = []
    for name, stat in ranked[:Config.TOP_N_DISCRIMINATIVE]:
        ic  = stat['inter_class']
        hr  = stat['high_risk']
        lr  = stat['low_risk']

        if ic['abs_cohen_d'] < Config.MIN_COHEN_D:
            continue

        direction = ic['risk_direction']
        threshold = ic['youden_threshold']
        op        = '>' if direction == 'above' else '<'
        rule_text = (
            f"if {name} {op} {threshold:.6f}  →  High Risk"
        )

        rules.append({
            'rank'             : len(rules) + 1,
            'feature_name'     : name,
            'feature_index'    : int(stat['feature_index']),
            'rule'             : rule_text,
            'risk_direction'   : direction,
            'operator'         : op,
            'youden_threshold' : float(threshold),
            'youden_j_score'   : float(ic['youden_j_score']),
            'cohen_d'          : float(ic['cohen_d']),
            'abs_cohen_d'      : float(ic['abs_cohen_d']),
            'p_value'          : float(ic['p_value']),
            'significant_p05'  : bool(ic['significant_p05']),
            'high_risk_mean'   : float(hr['mean']),
            'low_risk_mean'    : float(lr['mean']),
            'high_risk_std'    : float(hr['std']),
            'low_risk_std'     : float(lr['std']),
            'mean_difference'  : float(hr['mean'] - lr['mean']),
        })

    return rules


# =============================================================================
# PROBABILITY CALIBRATION THRESHOLDS
# =============================================================================
def derive_calibration_thresholds(features: np.ndarray,
                                   labels: np.ndarray,
                                   rules: list) -> dict:
    """
    Build a normalised composite risk score from the top discriminative rules
    and scan for two probability calibration thresholds:

      high_confidence_threshold
          Risk score above which a polyp is classified as High Risk with high
          specificity (≥ 0.90).  Analogous to ASGE's 0.90 cutoff, but derived
          purely from the NeoPolyp data distribution.

      uncertainty_threshold
          Risk score above which a polyp warrants further review.
          Found by targeting sensitivity ≥ 0.90.

    The risk score and both thresholds are included in the output JSON so that
    Phase 4 (mixture_of_experts.py) and the inference script can optionally
    load this file to replace any hard-coded clinical guideline values.
    """
    if not rules:
        return {
            'high_confidence_threshold' : 0.80,
            'uncertainty_threshold'     : 0.50,
            'method'                    : 'fallback — no discriminative rules derived',
            'scores'                    : [0.0] * len(features),
        }

    # --- Compute weighted composite risk score --------------------------------
    # Each rule contributes proportionally to its |Cohen's d|.
    # A rule "fires" when the feature exceeds its Youden threshold in the
    # risk-positive direction.
    scores       = np.zeros(len(features), dtype=np.float32)
    total_weight = sum(r['abs_cohen_d'] for r in rules)

    for rule in rules:
        fi        = rule['feature_index']
        threshold = rule['youden_threshold']
        direction = rule['risk_direction']
        col       = features[:, fi]
        fires     = (col > threshold) if direction == 'above' else (col < threshold)
        scores   += rule['abs_cohen_d'] * fires.astype(np.float32)

    # Normalise to [0, 1]
    scores = scores / (total_weight + 1e-9)

    # --- Scan thresholds via ROC-style analysis --------------------------------
    candidates = np.linspace(0.0, 1.0, 1001)
    best_hc = {'threshold': 0.90, 'accuracy': 0.0, 'specificity': 0.0,
                'sensitivity': 0.0}
    best_uc = {'threshold': 0.50, 'accuracy': 0.0, 'specificity': 0.0,
                'sensitivity': 0.0}

    for t in candidates:
        pred  = (scores >= t).astype(int)
        acc   = float(np.mean(pred == labels))
        tp    = float(np.sum((pred == 1) & (labels == 1)))
        tn    = float(np.sum((pred == 0) & (labels == 0)))
        fp    = float(np.sum((pred == 1) & (labels == 0)))
        fn    = float(np.sum((pred == 0) & (labels == 1)))
        sens  = tp / (tp + fn + 1e-9)
        spec  = tn / (tn + fp + 1e-9)

        # High confidence: maximise accuracy subject to specificity ≥ 0.90
        if spec >= 0.90 and acc > best_hc['accuracy']:
            best_hc = {'threshold': float(t), 'accuracy': acc,
                       'specificity': float(spec), 'sensitivity': float(sens)}

        # Uncertainty boundary: maximise accuracy subject to sensitivity ≥ 0.90
        if sens >= 0.90 and acc > best_uc['accuracy']:
            best_uc = {'threshold': float(t), 'accuracy': acc,
                       'specificity': float(spec), 'sensitivity': float(sens)}

    # --- Per-class score statistics -------------------------------------------
    h_scores = scores[labels == 1]
    l_scores = scores[labels == 0]

    return {
        'high_confidence_threshold'   : best_hc['threshold'],
        'high_confidence_specificity' : best_hc['specificity'],
        'high_confidence_sensitivity' : best_hc['sensitivity'],
        'high_confidence_accuracy'    : best_hc['accuracy'],
        'uncertainty_threshold'       : best_uc['threshold'],
        'uncertainty_specificity'     : best_uc['specificity'],
        'uncertainty_sensitivity'     : best_uc['sensitivity'],
        'uncertainty_accuracy'        : best_uc['accuracy'],
        'score_high_risk_mean'        : float(np.mean(h_scores)) if len(h_scores) else 0.0,
        'score_high_risk_std'         : float(np.std(h_scores))  if len(h_scores) else 0.0,
        'score_low_risk_mean'         : float(np.mean(l_scores)) if len(l_scores) else 0.0,
        'score_low_risk_std'          : float(np.std(l_scores))  if len(l_scores) else 0.0,
        'n_rules_used'                : len(rules),
        'method': (
            'Weighted composite risk score from top discriminative visual features '
            '(weights = |Cohen\'s d|, normalised). '
            'High-confidence threshold targets specificity >= 0.90. '
            'Uncertainty threshold targets sensitivity >= 0.90. '
            'No external ASGE values used.'
        ),
        'scores': scores.tolist(),   # stored for CSV output; stripped from JSON
    }


# =============================================================================
# OUTPUTS — CSV
# =============================================================================
def save_feature_log_csv(features_matrix: np.ndarray,
                          labels: np.ndarray,
                          image_names: list,
                          meta_list: list,
                          calibration: dict,
                          output_path: Path) -> None:
    """
    Save one row per NeoPolyp image containing:
      image_name | label | risk_category | composite_risk_score |
      red_mask_ratio | green_mask_ratio |
      ssl_feat_0 … ssl_feat_383 | lab_L_dark … shape_compactness
    """
    scores = np.array(calibration.get('scores', [0.0] * len(features_matrix)),
                      dtype=np.float32)
    header = (
        ['image_name', 'label', 'risk_category',
         'composite_risk_score', 'red_mask_ratio', 'green_mask_ratio']
        + ALL_FEATURE_NAMES
    )

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, name in enumerate(image_names):
            m     = meta_list[i]
            row   = (
                [name,
                 m['label'],
                 m['risk_category'],
                 f"{scores[i]:.6f}",
                 f"{m['red_mask_ratio']:.6f}",
                 f"{m['green_mask_ratio']:.6f}"]
                + [f"{v:.8f}" for v in features_matrix[i].tolist()]
            )
            writer.writerow(row)

    print(f"  [OK] Feature log CSV  → {output_path}")
    print(f"       {len(image_names):,} rows  ×  {len(header):,} columns")


# =============================================================================
# OUTPUTS — JSON
# =============================================================================
def save_custom_groundtruths_json(metadata: dict,
                                   rules: list,
                                   calibration: dict,
                                   output_path: Path) -> None:
    """
    Primary output for downstream phases.
    Strip the large 'scores' list before serialising so the file stays compact.
    Phase 4 / inference can load this file to replace hard-coded ASGE thresholds.
    """
    calib_clean = {k: v for k, v in calibration.items() if k != 'scores'}

    payload = {
        'schema_version'       : '1.0',
        'generated_at'         : metadata['generated_at'],
        'generated_by'         : 'Phase 2C — NeoPolyp Custom Ground Truth',
        'pipeline_integration' : (
            "Insert Phase '2c' entry (phase2/neopolyp_custom_groundtruth.py) "
            "into master_run.py PHASES list between '2b' and '3' to run this "
            "automatically within the full pipeline."
        ),
        'design_note': (
            'Thresholds are derived purely from visual statistics of the NeoPolyp '
            'training set (Red mask = High Risk, Green mask = Low Risk). '
            'No external ASGE guidelines are encoded — the data speaks for itself.'
        ),
        'dataset_summary': {
            'total_images'  : int(metadata['total_images']),
            'n_high_risk'   : int(metadata['n_high_risk']),
            'n_low_risk'    : int(metadata['n_low_risk']),
            'high_risk_pct' : float(metadata['high_risk_pct']),
            'feature_dim'   : int(metadata['feature_dim']),
            'ssl_dim'       : Config.SSL_FEATURE_DIM,
            'biomarker_dim' : Config.BIOMARKER_DIM,
        },
        'calibration_thresholds' : calib_clean,
        'top_ground_truth_rules' : rules,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"  [OK] Custom ground truths JSON → {output_path}")


def save_feature_statistics_json(class_stats: dict, output_path: Path) -> None:
    """Full per-feature statistics (all 444 features × both classes)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_stats, f, indent=2)
    print(f"  [OK] Feature statistics JSON   → {output_path}")


# =============================================================================
# VISUALISATIONS
# =============================================================================
def _plot_top10_feature_comparison(class_stats: dict,
                                    rules: list,
                                    features: np.ndarray,
                                    labels: np.ndarray) -> None:
    """Box-plot comparison of the 10 most discriminative features."""
    if not rules:
        return

    top = rules[:10]
    cols = min(5, len(top))
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_flat = np.array(axes).flatten()

    for ax, rule in zip(axes_flat, top):
        fi   = rule['feature_index']
        name = rule['feature_name']
        h    = features[labels == 1, fi]
        l    = features[labels == 0, fi]

        ax.boxplot(
            [l, h],
            labels=['Low Risk\n(Green)', 'High Risk\n(Red)'],
            patch_artist=True,
            boxprops=dict(facecolor='#aec6e8'),
            medianprops=dict(color='navy', linewidth=2),
        )
        ax.axhline(
            rule['youden_threshold'],
            color='crimson', linestyle='--', linewidth=1.5,
            label=f"Threshold\n{rule['youden_threshold']:.4f}",
        )
        short = name if len(name) <= 28 else name[:25] + '…'
        ax.set_title(
            f"#{rule['rank']}  {short}\n|d| = {rule['abs_cohen_d']:.2f}",
            fontsize=8,
        )
        ax.set_ylabel('Feature Value', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper left')

    # Hide unused subplots
    for ax in axes_flat[len(top):]:
        ax.set_visible(False)

    plt.suptitle(
        'Top-10 Discriminative Features — High Risk (Red) vs Low Risk (Green)\n'
        'NeoPolyp Custom Ground Truth  |  No ASGE',
        fontsize=11,
    )
    plt.tight_layout()
    out = Config.VISUAL_OUTPUT / 'gt_top10_features.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Feature box-plot           → {out}")


def _plot_risk_score_distribution(calibration: dict, labels: np.ndarray) -> None:
    """Histogram of per-class composite risk scores + calibration cut-offs."""
    scores = np.array(calibration.get('scores', []))
    if len(scores) == 0:
        return

    hc_thr = calibration.get('high_confidence_threshold', 0.90)
    uc_thr = calibration.get('uncertainty_threshold',     0.50)

    h_scores = scores[labels == 1]
    l_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(l_scores, bins=bins, alpha=0.65, color='#2ca02c',
            edgecolor='white', label='Low Risk  (Green mask)')
    ax.hist(h_scores, bins=bins, alpha=0.65, color='#d62728',
            edgecolor='white', label='High Risk (Red mask)')
    ax.axvline(hc_thr, color='darkred',   linestyle='--', linewidth=2.0,
               label=f'High Confidence ≥ {hc_thr:.2f}')
    ax.axvline(uc_thr, color='darkorange', linestyle=':',  linewidth=2.0,
               label=f'Uncertainty ≥ {uc_thr:.2f}')
    ax.set_xlabel('Composite Risk Score', fontsize=11)
    ax.set_ylabel('Count',               fontsize=11)
    ax.set_title(
        'Data-Driven Risk Score Distribution — NeoPolyp\n'
        'Custom Ground Truth Calibration (No ASGE)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = Config.VISUAL_OUTPUT / 'gt_risk_score_dist.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Risk score distribution    → {out}")


def _plot_cohen_d_ranking(rules: list) -> None:
    """Horizontal bar chart of top-20 feature effect sizes, coloured by direction."""
    if not rules:
        return

    top    = rules[:20]
    names  = [r['feature_name'] for r in top][::-1]
    d_abs  = [r['abs_cohen_d']  for r in top][::-1]
    colors = ['#d73027' if r['cohen_d'] > 0 else '#1a9850' for r in top][::-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(names, d_abs, color=colors, edgecolor='black', linewidth=0.4)
    ax.axvline(0.20, color='#888888', linestyle=':', linewidth=1.0,
               label='Small  (0.20)')
    ax.axvline(0.50, color='#e6a817', linestyle=':', linewidth=1.0,
               label='Medium (0.50)')
    ax.axvline(0.80, color='#cc0000', linestyle=':', linewidth=1.0,
               label='Large  (0.80)')
    ax.set_xlabel("|Cohen's d|  (Effect Size)", fontsize=11)
    ax.set_title(
        "Top Discriminative Features — Effect Size Ranking\n"
        "Red bar = feature higher in High-Risk   |   "
        "Green bar = feature higher in Low-Risk",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = Config.VISUAL_OUTPUT / 'gt_cohen_d_ranking.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Cohen's d ranking plot     → {out}")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_console_summary(rules: list, calibration: dict) -> None:
    print("\n" + "=" * 80)
    print(" " * 10 + "CUSTOM GROUND TRUTH RULES  (Top 10 by |Cohen's d|)")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Feature':<36} {'Op':<3} {'Threshold':>12}"
          f"  {'|d|':>6}  {'J':>6}  {'p':>8}  {'Sig':>4}")
    print("  " + "-" * 76)

    for r in rules[:10]:
        sig = "*" if r['significant_p05'] else " "
        print(
            f"  {r['rank']:<5} {r['feature_name']:<36} "
            f"{r['operator']:<3} {r['youden_threshold']:>12.5f}  "
            f"{r['abs_cohen_d']:>6.3f}  {r['youden_j_score']:>6.3f}  "
            f"{r['p_value']:>8.4f}  {sig:>4}"
        )
    print("  (* = significant at p < 0.05)")

    print("\n" + "=" * 80)
    print(" " * 15 + "DATA-DRIVEN CALIBRATION THRESHOLDS")
    print("=" * 80)
    hc  = calibration.get('high_confidence_threshold', 'N/A')
    uc  = calibration.get('uncertainty_threshold',     'N/A')
    hsp = calibration.get('high_confidence_specificity', 0)
    uss = calibration.get('uncertainty_sensitivity',    0)
    print(f"  High Confidence  :  Risk Score ≥ {hc:.4f}   "
          f"[Specificity = {hsp:.2f}]")
    print(f"  Uncertainty Zone :  Risk Score ≥ {uc:.4f}   "
          f"[Sensitivity = {uss:.2f}]")
    print(f"\n  High Risk Score  :  "
          f"μ = {calibration.get('score_high_risk_mean', 0):.3f}  "
          f"σ = {calibration.get('score_high_risk_std', 0):.3f}")
    print(f"  Low Risk Score   :  "
          f"μ = {calibration.get('score_low_risk_mean', 0):.3f}  "
          f"σ = {calibration.get('score_low_risk_std', 0):.3f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    generated_at = datetime.now().isoformat()

    # -------------------------------------------------------------------------
    # 1. Load the Phase 2A SSL encoder
    # -------------------------------------------------------------------------
    ssl_model = load_ssl_model()
    transform  = build_transform(Config.IMG_SIZE)

    # -------------------------------------------------------------------------
    # 2. Collect dataset (image ↔ mask pairs)
    # -------------------------------------------------------------------------
    records = collect_dataset()

    if len(records) == 0:
        print("\n  [ERROR] No valid image-mask pairs found.")
        print(f"          Images : {Config.TRAIN_IMAGES}")
        print(f"          Masks  : {Config.TRAIN_MASKS}")
        print("  Ensure the NeoPolyp dataset is present and run Phase 2B first.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Extract 444-dim feature vectors for every image
    # -------------------------------------------------------------------------
    features_matrix, labels, image_names, meta_list = extract_all_features(
        records, ssl_model, transform
    )

    n_high = int(labels.sum())
    n_low  = int((labels == 0).sum())

    # -------------------------------------------------------------------------
    # 4. Per-feature statistical analysis (mean, std, Cohen's d, Youden thr.)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" " * 22 + "STATISTICAL ANALYSIS")
    print("=" * 80)
    class_stats = compute_class_statistics(features_matrix, labels)

    # -------------------------------------------------------------------------
    # 5. Generate ground truth rules ranked by effect size
    # -------------------------------------------------------------------------
    rules = generate_ground_truth_rules(class_stats)
    print(f"\n  Derived {len(rules)} ground truth rules "
          f"(|Cohen's d| >= {Config.MIN_COHEN_D})")

    # -------------------------------------------------------------------------
    # 6. Derive data-driven probability calibration thresholds
    # -------------------------------------------------------------------------
    print("\n  Deriving probability calibration thresholds …")
    calibration = derive_calibration_thresholds(features_matrix, labels, rules)

    # -------------------------------------------------------------------------
    # 7. Save all outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" " * 28 + "SAVING OUTPUTS")
    print("=" * 80)

    metadata = {
        'generated_at' : generated_at,
        'total_images' : len(image_names),
        'n_high_risk'  : n_high,
        'n_low_risk'   : n_low,
        'high_risk_pct': round(n_high / max(len(image_names), 1) * 100, 2),
        'feature_dim'  : int(features_matrix.shape[1]),
    }

    # (a) Detailed feature log — CSV
    csv_path = Config.GT_OUTPUT / 'neopolyp_feature_log.csv'
    save_feature_log_csv(
        features_matrix, labels, image_names,
        meta_list, calibration, csv_path
    )

    # (b) Ground truth rules + calibration thresholds — JSON
    json_path = Config.GT_OUTPUT / 'custom_groundtruths.json'
    save_custom_groundtruths_json(metadata, rules, calibration, json_path)

    # (c) Full per-feature statistics — JSON
    stats_path = Config.GT_OUTPUT / 'feature_statistics.json'
    save_feature_statistics_json(class_stats, stats_path)

    # -------------------------------------------------------------------------
    # 8. Visualisations
    # -------------------------------------------------------------------------
    print("\n  Generating visualisations …")
    _plot_top10_feature_comparison(class_stats, rules, features_matrix, labels)
    _plot_risk_score_distribution(calibration, labels)
    _plot_cohen_d_ranking(rules)

    # -------------------------------------------------------------------------
    # 9. Console summary
    # -------------------------------------------------------------------------
    print_console_summary(rules, calibration)

    # -------------------------------------------------------------------------
    # 10. Final status
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" " * 10 + "PHASE 2C COMPLETE — CUSTOM GROUND TRUTHS GENERATED")
    print("=" * 80)

    print(f"\n  Output directory : {Config.GT_OUTPUT}")
    print(f"\n  Files written:")
    print(f"    neopolyp_feature_log.csv   "
          f"{len(image_names):,} images × {features_matrix.shape[1]} features")
    print(f"    custom_groundtruths.json   "
          f"{len(rules)} rules + calibration thresholds")
    print(f"    feature_statistics.json    "
          f"full statistics for all {features_matrix.shape[1]} features")
    print(f"    gt_top10_features.png      top-10 feature box-plots")
    print(f"    gt_risk_score_dist.png     composite risk score distribution")
    print(f"    gt_cohen_d_ranking.png     effect size ranking chart")

    print(f"\n  Integration note:")
    print(f"    Phase 4 (mixture_of_experts.py) and the inference script can")
    print(f"    optionally load  {json_path.name}")
    print(f"    from  {Config.GT_OUTPUT}")
    print(f"    to replace any hard-coded ASGE probability thresholds with these")
    print(f"    data-driven values.")
    print()


if __name__ == '__main__':
    main()
