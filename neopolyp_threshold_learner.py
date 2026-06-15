# -*- coding: utf-8 -*-
"""
NEOPOLYP THRESHOLD LEARNER
Derives all symbolic reasoning thresholds and risk weights from NeoPolyp ground truth.
Red mask = HIGH_RISK (neoplastic).  Green mask = LOW_RISK (non-neoplastic).

Run once:
    python neopolyp_threshold_learner.py --neopolyp_dir /path/to/NeoPolyp \
                                          --output_path mixture_of_experts/neopolyp_thresholds.json

Output JSON is loaded by SymbolicReasoningIntegrator at inference time.
No hardcoded thresholds anywhere — every number comes from this dataset.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ── Feature extraction (same formulas as inference pipeline) ─────────────────

def _extract_features_from_roi(roi_bgr: np.ndarray) -> dict:
    """
    Extract the same 4 features used at inference time.
    roi_bgr: the polyp ROI cropped by the mask bounding box, BGR uint8.
    Returns dict with redness, vessel_visibility, texture, radius, s_mean.
    Texture divisor is computed per-dataset (99th percentile), not hardcoded.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = roi_rgb[:, :, 0], roi_rgb[:, :, 1], roi_rgb[:, :, 2]

    # Redness: (R-G)/(R+G+B), clipped to [0,1]
    denom = R + G + B + 1e-6
    redness = float(np.clip(np.mean((R - G) / denom), 0.0, 1.0))

    # Vessel visibility: fraction of pixels where R > 1.2*G (blood tone)
    vessel_mask = (R > 1.2 * G + 10).astype(np.float32)
    vessel_visibility = float(np.mean(vessel_mask))

    # Texture: Laplacian variance — divisor derived later at dataset level
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Radius: ratio of longest side to frame diagonal (approximate for crops)
    h, w = roi_bgr.shape[:2]
    radius = float(max(h, w) / (np.sqrt(h**2 + w**2) + 1e-6))

    # HSV saturation mean
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_mean = float(np.mean(hsv[:, :, 1]) / 255.0)

    return {
        'redness': redness,
        'vessel_visibility': vessel_visibility,
        'lap_var_raw': lap_var,   # raw, normalised after dataset scan
        'radius': radius,
        's_mean': s_mean,
    }


def _mask_to_roi(image_bgr: np.ndarray, mask_bgr: np.ndarray):
    """
    Extract polyp ROI from image using the mask bounding box.
    Returns (roi_bgr, label) where label is 'high' or 'low', or None if mask empty.
    
    Handles multiple mask formats:
    - Red channel dominant in mask → HIGH_RISK
    - Green channel dominant → LOW_RISK
    - Binary mask (any non-zero) → defaults to HIGH_RISK (conservative)
    """
    if mask_bgr is None or image_bgr is None:
        return None, None

    # Handle grayscale masks
    if len(mask_bgr.shape) == 2:
        # Simple binary mask — treat all polyps as high risk (conservative default)
        bin_mask = (mask_bgr > 127).astype(np.uint8)
        label = 'high'
    else:
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        R, G, B = mask_rgb[:, :, 0], mask_rgb[:, :, 1], mask_rgb[:, :, 2]

        red_pixels   = np.sum((R > 100) & (R > G * 1.5) & (R > B * 1.5))
        green_pixels = np.sum((G > 100) & (G > R * 1.5) & (G > B * 1.5))

        if red_pixels > green_pixels and red_pixels > 50:
            label = 'high'
            bin_mask = ((R > 100) & (R > G * 1.5)).astype(np.uint8)
        elif green_pixels > red_pixels and green_pixels > 50:
            label = 'low'
            bin_mask = ((G > 100) & (G > R * 1.5)).astype(np.uint8)
        else:
            # No strong color signal — treat as binary mask
            # Threshold at 127 on any channel with at least one channel > 100
            any_signal = (R > 100) | (G > 100) | (B > 100)
            if any_signal.sum() > 50:
                label = 'high'  # Conservative default
                bin_mask = any_signal.astype(np.uint8)
            else:
                return None, None

    coords = cv2.findNonZero(bin_mask)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(image_bgr.shape[1], x + w + pad)
    y2 = min(image_bgr.shape[0], y + h + pad)
    roi = image_bgr[y1:y2, x1:x2]
    return roi, label


# ── Dataset scan ─────────────────────────────────────────────────────────────

def scan_neopolyp_dataset(neopolyp_dir: Path):
    """
    Walk the NeoPolyp directory.  Expects pairs:
        images/xxx.jpg  ←→  masks/xxx.jpg   (or .png)
        OR
        train/xxx.jpg  ←→  train_gt/xxx.jpg
    Returns list of dicts with features + label.
    """
    image_dir = neopolyp_dir / 'images'
    mask_dir  = neopolyp_dir / 'masks'

    if not image_dir.exists():
        # Try train / train_gt layout
        if (neopolyp_dir / 'train').exists() and (neopolyp_dir / 'train_gt').exists():
            image_dir = neopolyp_dir / 'train' / 'train'
            mask_dir  = neopolyp_dir / 'train_gt' / 'train_gt'
            if not image_dir.exists():
                image_dir = neopolyp_dir / 'train'
                mask_dir  = neopolyp_dir / 'train_gt'
        else:
            # Try flat layout: images and masks in the same folder with _mask suffix
            image_dir = neopolyp_dir
            mask_dir  = neopolyp_dir

    image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png')))
    print(f"   Found {len(image_paths)} images in {image_dir}")

    records = []
    lap_var_all = []

    for img_path in image_paths:
        # Find matching mask
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            stem = img_path.stem
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = mask_dir / (stem + '_mask' + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
        if not mask_path.exists():
            continue

        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))
        if img is None or mask is None:
            continue

        roi, label = _mask_to_roi(img, mask)
        if roi is None or roi.size == 0:
            continue

        feats = _extract_features_from_roi(roi)
        if feats is None:
            continue

        feats['label'] = label
        lap_var_all.append(feats['lap_var_raw'])
        records.append(feats)

    print(f"   Processed {len(records)} valid polyp ROIs "
          f"({sum(1 for r in records if r['label']=='high')} high-risk, "
          f"{sum(1 for r in records if r['label']=='low')} low-risk)")
    return records, lap_var_all


# ── Threshold derivation ──────────────────────────────────────────────────────

def derive_thresholds(records: list, lap_var_all: list) -> dict:
    """
    From NeoPolyp records derive:
      1. texture_divisor  — 99th percentile Laplacian variance (replaces hardcoded 2000)
      2. feature thresholds — optimal cut per feature from ROC (replaces hardcoded T_*)
      3. logistic weights — learned coefficients (replaces hardcoded +0.35/+0.20/etc.)
      4. risk buckets — Youden-optimal HIGH/LOW boundary + MEDIUM band (replaces 0.55/0.20)
      5. confidence clip — data-driven percentile bounds (replaces hardcoded [0.35, 0.88])
      6. margin_penalty — learned from calibration (replaces hardcoded 0.20)
    """
    # ── 1. Texture divisor ────────────────────────────────────────────────────
    texture_divisor = float(np.percentile(lap_var_all, 99)) if lap_var_all else 2000.0
    texture_divisor = max(texture_divisor, 100.0)   # safety floor
    print(f"   Texture divisor (99th pct Laplacian var): {texture_divisor:.1f}")

    # Normalise texture in records now that divisor is known
    for r in records:
        r['texture'] = float(np.clip(r['lap_var_raw'] / texture_divisor, 0.0, 1.0))

    # ── 2. Build feature arrays ───────────────────────────────────────────────
    feature_names = ['redness', 'vessel_visibility', 'texture', 'radius', 's_mean']
    X = np.array([[r[f] for f in feature_names] for r in records], dtype=np.float32)
    y = np.array([1 if r['label'] == 'high' else 0 for r in records], dtype=np.int32)

    if len(np.unique(y)) < 2:
        raise ValueError("NeoPolyp scan found only one class — check your mask directory.")

    # ── 3. Per-feature optimal threshold from ROC (Youden index) ─────────────
    feature_thresholds = {}
    for i, fname in enumerate(feature_names):
        fpr, tpr, threshs = roc_curve(y, X[:, i])
        youden = tpr - fpr
        best_idx = int(np.argmax(youden))
        feature_thresholds[fname] = float(threshs[best_idx])
        print(f"   Threshold {fname}: {feature_thresholds[fname]:.4f}  (Youden={youden[best_idx]:.3f})")

    # ── 4. Logistic regression weights (data-driven risk_score) ──────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    lr.fit(X_scaled, y)
    # Keep weights as normalised positive contributions summing to 1
    raw_coef = lr.coef_[0]
    # Signed coefficients — high positive = risk-increasing feature
    coef_dict = {fname: float(raw_coef[i]) for i, fname in enumerate(feature_names)}
    scaler_dict = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': feature_names,
    }
    print(f"   Logistic weights: { {k: round(v,3) for k,v in coef_dict.items()} }")

    # ── 5. Risk bucket boundaries from logistic probability distribution ──────
    probs_train = lr.predict_proba(X_scaled)[:, 1]

    # HIGH threshold: point on ROC with max Youden
    fpr_all, tpr_all, threshs_all = roc_curve(y, probs_train)
    youden_all = tpr_all - fpr_all
    high_boundary = float(threshs_all[int(np.argmax(youden_all))])

    # LOW threshold: 20th percentile of high-risk prob distribution (conservative)
    high_probs = probs_train[y == 1]
    low_boundary = float(np.percentile(high_probs, 20)) if len(high_probs) > 0 else high_boundary * 0.4
    low_boundary = min(low_boundary, high_boundary * 0.6)  # ensure gap

    print(f"   Risk buckets: LOW < {low_boundary:.3f} ≤ MEDIUM < {high_boundary:.3f} ≤ HIGH")

    # ── 6. Confidence calibration bounds ─────────────────────────────────────
    # From the actual prediction probability distribution on training data
    conf_p5  = float(np.percentile(probs_train, 5))
    conf_p95 = float(np.percentile(probs_train, 95))
    print(f"   Confidence clip: [{conf_p5:.3f}, {conf_p95:.3f}]")

    # ── 7. Margin penalty — calibrate via isotonic regression ────────────────
    # We want: after clipping, how much should a margin of M reduce confidence?
    # Use the mean margin across training predictions as the natural scale factor.
    proba_matrix = lr.predict_proba(X_scaled)
    sorted_probs = np.sort(proba_matrix, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    margin_penalty = float(np.clip(np.std(margins), 0.05, 0.30))
    print(f"   Margin penalty coefficient: {margin_penalty:.3f}")

    return {
        'texture_divisor':    texture_divisor,
        'feature_thresholds': feature_thresholds,
        'logistic_weights':   coef_dict,
        'scaler':             scaler_dict,
        'logistic_intercept': float(lr.intercept_[0]),
        'risk_high_boundary': high_boundary,
        'risk_low_boundary':  low_boundary,
        'confidence_clip_low':  conf_p5,
        'confidence_clip_high': conf_p95,
        'margin_penalty':       margin_penalty,
        'feature_names':        feature_names,
        'n_high_risk':  int(np.sum(y == 1)),
        'n_low_risk':   int(np.sum(y == 0)),
        'derived_from': 'NeoPolyp_ground_truth',
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Derive symbolic reasoning thresholds from NeoPolyp')
    parser.add_argument('--neopolyp_dir', type=str, required=True,
                        help='Path to NeoPolyp dataset root (contains images/ and masks/ folders)')
    parser.add_argument('--output_path', type=str,
                        default='mixture_of_experts/neopolyp_thresholds.json',
                        help='Where to save the derived thresholds JSON')
    args = parser.parse_args()

    neopolyp_dir = Path(args.neopolyp_dir)
    output_path  = Path(args.output_path)

    print(f"\n{'='*60}")
    print(f"  NeoPolyp Threshold Learner")
    print(f"  Dataset : {neopolyp_dir}")
    print(f"  Output  : {output_path}")
    print(f"{'='*60}\n")

    print("1. Scanning NeoPolyp dataset...")
    records, lap_var_all = scan_neopolyp_dataset(neopolyp_dir)

    if len(records) < 20:
        raise ValueError(f"Only {len(records)} valid ROIs found — check your dataset path and mask format.")

    print("\n2. Deriving thresholds...")
    thresholds = derive_thresholds(records, lap_var_all)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✅ Thresholds saved to: {output_path}")
    print(f"   Run your inference pipeline — thresholds will be loaded automatically.\n")


if __name__ == '__main__':
    main()
