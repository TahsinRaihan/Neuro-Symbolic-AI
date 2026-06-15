# -*- coding: utf-8 -*-
"""
ASGE-COMPLIANT INFERENCE PIPELINE
Complete cascade: Detection → Segmentation → Feature Extraction → Mixture of Experts → ASGE Decision

Implements ASGE PIVI Standards:
- High Confidence (≥0.90): "HIGH RISK - Resect & Discard"
- Uncertainty (0.80-0.90): "UNCERTAIN HIGH RISK - Require Biopsy/Review"
- Low Risk (<0.80): "LOW RISK - Surveillance"

Pipeline Stages:
1. Detection: YOLO/RT-DETR for polyp localization
2. Segmentation: UNet++ for precise boundaries
3. Feature Extraction: Dual-stream (384-dim ViT + 60-dim biomarkers)
4. Clustering: Assign to one of 8 visual prototypes
5. Expert Prediction: Local Decision Tree provides risk probability
6. ASGE Decision: Apply clinical thresholds for final recommendation
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
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
import timm

print("=" * 80)
print(" " * 15 + "ASGE-COMPLIANT INFERENCE PIPELINE")
print(" " * 10 + "(Detection → Segmentation → Mixture of Experts → ASGE Decision)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    DATASET_2_ROOT = THESIS_ROOT / 'NeSy' / 'Dataset 2' / 'images'
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'

    # Model paths
    DETECTION_OUTPUT = OUTPUT_ROOT / 'detection_models'
    SEGMENTATION_OUTPUT = OUTPUT_ROOT / 'segmentation_models'
    SSL_OUTPUT = OUTPUT_ROOT / 'ssl_outputs'
    FEATURES_OUTPUT = OUTPUT_ROOT / 'extracted_features'
    EXPERTS_OUTPUT = OUTPUT_ROOT / 'mixture_of_experts'

    # Inference output
    INFERENCE_OUTPUT = OUTPUT_ROOT / 'inference_results_asge'
    REPORTS_DIR = INFERENCE_OUTPUT / 'individual_reports'
    VISUALIZATION_DIR = INFERENCE_OUTPUT / 'visualizations'
    HIGH_RISK_CASES_DIR = INFERENCE_OUTPUT / 'high_risk_cases'
    UNCERTAIN_CASES_DIR = INFERENCE_OUTPUT / 'uncertain_cases'

    # Detection thresholds
    DETECTION_CONF = 0.35
    DETECTION_NMS_IOU = 0.40   # IoU threshold for cross-model NMS
    DETECTION_MIN_SIZE = 20    # Minimum bbox side length in pixels (filters noise)
    DETECTION_MAX_PER_IMAGE = 5  # Keep only top-N detections by confidence

    # ASGE PIVI Thresholds
    ASGE_HIGH_CONFIDENCE = 0.90
    ASGE_UNCERTAINTY = 0.80

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 256
    NUM_CLUSTERS = 8

# Create directories
for dir_path in [Config.INFERENCE_OUTPUT, Config.REPORTS_DIR, Config.VISUALIZATION_DIR,
                 Config.HIGH_RISK_CASES_DIR, Config.UNCERTAIN_CASES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Device: {Config.DEVICE}")
print(f"   Dataset: {Config.DATASET_2_ROOT}")
print(f"   ASGE High Confidence: ≥{Config.ASGE_HIGH_CONFIDENCE}")
print(f"   ASGE Uncertainty: {Config.ASGE_UNCERTAINTY}-{Config.ASGE_HIGH_CONFIDENCE}")

# ==========================================
# LOAD ALL MODELS
# ==========================================
print("\n" + "=" * 80)
print(" " * 30 + "LOADING MODELS")
print("=" * 80)

# 1. Detection Models
yolo_model = None
rtdetr_model = None

try:
    from ultralytics import YOLO
    yolo_path = Config.DETECTION_OUTPUT / 'yolov8m' / 'weights' / 'best.pt'
    if yolo_path.exists():
        yolo_model = YOLO(str(yolo_path))
        print("✅ YOLOv8 loaded")
    else:
        print(f"⚠️  YOLOv8 not found at {yolo_path}")
except Exception as e:
    print(f"⚠️  YOLOv8 loading failed: {e}")

try:
    from ultralytics import RTDETR
    rtdetr_path = Config.DETECTION_OUTPUT / 'rtdetr_polyp' / 'weights' / 'best.pt'
    if rtdetr_path.exists():
        rtdetr_model = RTDETR(str(rtdetr_path))
        print("✅ RT-DETR loaded")
except Exception as e:
    print(f"⚠️  RT-DETR loading failed: {e}")

# 2. Segmentation Model
segmentation_model = None
try:
    import segmentation_models_pytorch as smp

    seg_paths = [
        Config.SEGMENTATION_OUTPUT / 'unetpp_best.pth',
        Config.SEGMENTATION_OUTPUT / 'unetpp_final.pth',
    ]

    for seg_path in seg_paths:
        if seg_path.exists():
            try:
                segmentation_model = smp.UnetPlusPlus(
                    encoder_name='resnet34',
                    encoder_weights='imagenet',  # Use pretrained ImageNet weights
                    classes=1,
                    activation='sigmoid'
                )

                state_dict = torch.load(str(seg_path), map_location=Config.DEVICE)

                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    segmentation_model.load_state_dict(state_dict['model_state_dict'])
                elif isinstance(state_dict, dict) and 'encoder' in str(state_dict.keys()):
                    segmentation_model.load_state_dict(state_dict)
                else:
                    # Might be just the state dict
                    segmentation_model.load_state_dict(state_dict)

                segmentation_model = segmentation_model.to(Config.DEVICE)
                segmentation_model.eval()
                print("✅ UNet++ segmentation loaded")
                break
            except Exception as e:
                continue

    if segmentation_model is None:
        print("⚠️  Segmentation model files found but loading failed")

except Exception as e:
    print(f"⚠️  Segmentation loading failed: {e}")

# 3. SSL Model
ssl_model = None
try:
    class SSLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model('vit_small_patch16_224',
                                             pretrained=False,
                                             num_classes=0,
                                             img_size=256)

        def forward(self, x):
            return self.backbone(x)

    ssl_model = SSLModel().to(Config.DEVICE)

    # Try different checkpoint locations
    ssl_paths = [
        Config.SSL_OUTPUT / 'ssl_encoder_final.pth',
        Config.SSL_OUTPUT / 'ssl_model_final.pth',
        Config.SSL_OUTPUT / 'best_ssl_model.pth',
    ]

    for ssl_path in ssl_paths:
        if ssl_path.exists():
            try:
                state_dict = torch.load(str(ssl_path), map_location=Config.DEVICE)
                if 'backbone' in str(state_dict.keys()):
                    ssl_model.load_state_dict(state_dict)
                else:
                    ssl_model.backbone.load_state_dict(state_dict)
                ssl_model.eval()
                print("✅ SSL encoder loaded")
                break
            except:
                continue
except Exception as e:
    print(f"⚠️  SSL model loading failed: {e}")

# 4. Clustering Model
kmeans = None
scaler = None
try:
    kmeans_path = Config.FEATURES_OUTPUT / 'kmeans_model.pkl'
    scaler_path = Config.FEATURES_OUTPUT / 'feature_scaler.pkl'

    if kmeans_path.exists() and scaler_path.exists():
        kmeans = joblib.load(str(kmeans_path))
        scaler = joblib.load(str(scaler_path))
        print("✅ K-Means clustering loaded")
except Exception as e:
    print(f"⚠️  Clustering model loading failed: {e}")

# 5. Mixture of Experts
local_experts = {}
try:
    for cluster_id in range(Config.NUM_CLUSTERS):
        expert_path = Config.EXPERTS_OUTPUT / f'expert_cluster_{cluster_id}.pkl'
        if expert_path.exists():
            local_experts[cluster_id] = joblib.load(str(expert_path))

    if len(local_experts) > 0:
        print(f"✅ Mixture of Experts loaded ({len(local_experts)} experts)")
except Exception as e:
    print(f"⚠️  Expert loading failed: {e}")

# ==========================================
# MLP PROBABILITY CALIBRATOR DEFINITION & LOADING
# ==========================================
class MLPCalibrator(torch.nn.Module):
    """
    Phase 4.5 instance-level probability calibrator.
    Input  : 452-dim  =  444 raw features  +  8 one-hot cluster ID
    Output : scalar sigmoid probability -> P(Neoplastic / High-Risk)
    Architecture and hidden-dim list are stored in mlp_metadata.json
    so the same model can be reconstructed at load time.
    """
    def __init__(self, input_dim=452, hidden_dims=None, dropout_rates=None):
        super().__init__()
        if hidden_dims   is None: hidden_dims   = [256, 128, 64]
        if dropout_rates is None: dropout_rates = [0.35, 0.30, 0.20]
        blocks, in_dim = [], input_dim
        for h, dr in zip(hidden_dims, dropout_rates):
            blocks += [torch.nn.Linear(in_dim, h),
                       torch.nn.BatchNorm1d(h),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(dr)]
            in_dim = h
        blocks.append(torch.nn.Linear(in_dim, 1))
        self.network = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x).squeeze(-1)

# Load MLP Calibrator (Phase 4.5)
mlp_model        = None
mlp_feat_scaler  = None
mlp_num_clusters = Config.NUM_CLUSTERS
mlp_feature_dim  = 444   # 384 SSL + 60 Biomarkers

try:
    mlp_output      = Config.OUTPUT_ROOT / 'mlp_calibrator'
    mlp_model_path  = mlp_output / 'mlp_model.pth'
    mlp_scaler_path = mlp_output / 'mlp_scaler.pkl'
    mlp_meta_path   = mlp_output / 'mlp_metadata.json'

    if mlp_model_path.exists() and mlp_scaler_path.exists():
        # Load architecture config from metadata if available
        hidden_dims_cfg   = [256, 128, 64]
        dropout_rates_cfg = [0.35, 0.30, 0.20]
        if mlp_meta_path.exists():
            with open(mlp_meta_path) as _mf:
                _meta = json.load(_mf)
            arc = _meta.get('architecture', {})
            hidden_dims_cfg   = arc.get('hidden_dims',   hidden_dims_cfg)
            dropout_rates_cfg = arc.get('dropout_rates', dropout_rates_cfg)
            mlp_feature_dim   = arc.get('feature_dim',   mlp_feature_dim)
            mlp_num_clusters  = arc.get('cluster_dim',   mlp_num_clusters)

        _input_dim  = mlp_feature_dim + mlp_num_clusters   # 452
        mlp_model   = MLPCalibrator(input_dim=_input_dim,
                                     hidden_dims=hidden_dims_cfg,
                                     dropout_rates=dropout_rates_cfg)
        mlp_model.load_state_dict(
            torch.load(str(mlp_model_path), map_location='cpu'))
        mlp_model.eval()
        mlp_feat_scaler = joblib.load(str(mlp_scaler_path))
        print(f"✅ MLP Calibrator loaded  (input_dim={_input_dim}, "
              f"hidden={hidden_dims_cfg})")
    else:
        print("⚠️  MLP Calibrator not found – will fall back to Decision Trees")
except Exception as _e:
    print(f"⚠️  MLP Calibrator loading failed: {_e}")
    mlp_model       = None
    mlp_feat_scaler = None

# ==========================================
# BIOMARKER EXTRACTION (Updated to match Phase 3)
# ==========================================

def extract_biomarkers(image_np):
    """
    Extract 60-dimensional biomarker vector
    NOW MATCHES Phase 3 with CIELAB, Haralick, and improved features
    """
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

    # Validate input
    if image_np is None or image_np.size == 0:
        raise ValueError("Input image is empty or None")

    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_np.shape}")

    # CIELAB histograms (9 features)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    hist_l = cv2.calcHist([l_channel], [0], None, [3], [0, 256])
    hist_a = cv2.calcHist([a_channel], [0], None, [3], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [3], [0, 256])

    hist_l = hist_l.flatten() / (hist_l.sum() + 1e-7)
    hist_a = hist_a.flatten() / (hist_a.sum() + 1e-7)
    hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)

    # Saturation histogram (16 features)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s_channel, _ = cv2.split(hsv)
    hist_sat = cv2.calcHist([s_channel], [0], None, [16], [0, 256])
    hist_sat = hist_sat.flatten() / (hist_sat.sum() + 1e-7)

    # Haralick features (13 features)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_normalized = (gray / 16).astype(np.uint8)

    try:
        glcm = graycomatrix(gray_normalized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=16, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()

        haralick_features = np.concatenate([
            contrast, dissimilarity, homogeneity, [energy.mean()]
        ])[:13]
    except:
        haralick_features = np.zeros(13, dtype=np.float32)

    # LBP features (19 features)
    lbp = local_binary_pattern(gray, 18, 2, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=20, range=(0, 20))
    hist_lbp = hist_lbp.astype(np.float32) / (hist_lbp.sum() + 1e-7)
    hist_lbp = hist_lbp[:19]

    # Shape features (3 features)
    edges = cv2.Canny(gray, 50, 150)
    texture = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0

    rough_mask = ((s_channel > 30) | (hsv[:,:,2] < 200)).astype(np.uint8)
    relative_area = np.mean(rough_mask)

    contours, _ = cv2.findContours(rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-7) if perimeter > 0 else 0.0
    else:
        compactness = 0.0

    # Combine all features (9+16+13+19+3=60)
    biomarkers = np.concatenate([
        hist_l,              # 3
        hist_a,              # 3
        hist_b,              # 3
        hist_sat,            # 16
        haralick_features,   # 13
        hist_lbp,            # 19
        [texture, relative_area, compactness]  # 3
    ])

    assert len(biomarkers) == 60, f"Expected 60 biomarkers, got {len(biomarkers)}"
    return biomarkers.astype(np.float32)

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
# ASGE-COMPLIANT PREDICTION
# ==========================================
def predict_risk(image_np, ssl_model, kmeans, scaler, local_experts):
    """
    Complete ASGE-compliant risk prediction pipeline
    NOW USES FULL 444-DIM FEATURES (384 SSL + 60 Biomarkers)

    Returns:
        dict with keys:
        - risk_probability: float (0-1)
        - cluster_id: int (0-7)
        - decision: str (HIGH RISK, UNCERTAIN, LOW RISK)
        - recommendation: str (Clinical action)
        - confidence: str (High/Uncertain/Low)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Apply padding (preserve aspect ratio)
    image_pil = Image.fromarray(image_np)
    image_pil = pad_to_square(image_pil, Config.IMG_SIZE)
    image_np_padded = np.array(image_pil)

    # Step 2: Extract SSL features
    image_tensor = transform(image_pil).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        ssl_features = ssl_model(image_tensor).cpu().numpy().flatten()

    # Step 3: Extract biomarkers
    biomarkers = extract_biomarkers(image_np_padded)

    # Step 4: Combine and predict cluster
    fact_vector = np.concatenate([ssl_features, biomarkers])  # Full 444-dim
    fact_vector_scaled = scaler.transform(fact_vector.reshape(1, -1))
    cluster_id = int(kmeans.predict(fact_vector_scaled)[0])

    # Step 5: Get instance-level probability
    # Priority: MLP Calibrator (Phase 4.5) → Decision Tree (Phase 4) → default
    prediction_source = 'unknown'
    try:
        if mlp_model is not None and mlp_feat_scaler is not None:
            # ── MLP path ─────────────────────────────────────────────────────
            # Scale the 444-dim feature block with MLP-specific scaler
            feat_block_sc = mlp_feat_scaler.transform(
                fact_vector.reshape(1, -1))   # (1, 444)

            # One-hot encode cluster ID
            cluster_oh = np.zeros(mlp_num_clusters, dtype=np.float32)
            cluster_oh[cluster_id] = 1.0

            # Concatenate: [scaled 444-dim | 8-dim one-hot] = 452-dim
            mlp_input = np.concatenate(
                [feat_block_sc[0], cluster_oh]).astype(np.float32)    # (452,)

            mlp_tensor = torch.from_numpy(mlp_input).unsqueeze(0)     # (1, 452)
            mlp_model.eval()
            with torch.no_grad():
                logit = mlp_model(mlp_tensor)
                risk_probability = float(torch.sigmoid(logit).item())
            prediction_source = 'MLP_Calibrator'

        else:
            # ── Decision Tree fallback ────────────────────────────────────────
            expert = local_experts.get(cluster_id)
            if expert is None:
                risk_probability  = 0.5
                prediction_source = 'default_uncertain'
            else:
                risk_proba = expert.predict_proba(fact_vector.reshape(1, -1))
                risk_probability  = (float(risk_proba[0, 1])
                                     if risk_proba.shape[1] == 2 else 0.5)
                prediction_source = 'DecisionTree_fallback'

    except Exception as e:
        print(f"   ⚠️  Probability prediction failed: {e}")
        risk_probability  = 0.5
        prediction_source = 'error_fallback'

    # Step 6: Apply ASGE PIVI thresholds
    if risk_probability >= Config.ASGE_HIGH_CONFIDENCE:
        decision       = "HIGH RISK"
        recommendation = "Resect & Discard (ASGE High Confidence)"
        confidence     = "High"
    elif risk_probability >= Config.ASGE_UNCERTAINTY:
        decision       = "UNCERTAIN HIGH RISK"
        recommendation = "Require Biopsy/Pathology Review (Below ASGE Threshold)"
        confidence     = "Uncertain"
    else:
        decision       = "LOW RISK"
        recommendation = "Surveillance (Low Risk)"
        confidence     = "Low"

    # ── (no more exception handler needed – already handled above) ────────────

    return {
        'risk_probability'  : risk_probability,
        'cluster_id'        : cluster_id,
        'decision'          : decision,
        'recommendation'    : recommendation,
        'confidence'        : confidence,
        'prediction_source' : prediction_source,
    }

# ==========================================
# DETECTION & SEGMENTATION
# ==========================================
def _nms(detections, iou_threshold=0.40):
    """Apply Non-Maximum Suppression across all detections."""
    if not detections:
        return []

    boxes  = np.array([d['bbox'] for d in detections], dtype=np.float32)  # (N,4) xyxy
    scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # sort by confidence descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_threshold]

    return [detections[k] for k in keep]


def detect_polyps(image_path):
    """Run detection models to find polyps, then apply cross-model NMS."""
    detections = []

    # YOLO
    if yolo_model is not None:
        try:
            results = yolo_model(str(image_path), conf=Config.DETECTION_CONF, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({
                    'model': 'YOLO',
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        except Exception as e:
            print(f"   ⚠️  YOLO detection failed: {e}")

    # RT-DETR
    if rtdetr_model is not None:
        try:
            results = rtdetr_model(str(image_path), conf=Config.DETECTION_CONF, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({
                    'model': 'RT-DETR',
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        except Exception as e:
            print(f"   ⚠️  RT-DETR detection failed: {e}")

    # Filter out tiny noise detections
    detections = [
        d for d in detections
        if (d['bbox'][2] - d['bbox'][0]) >= Config.DETECTION_MIN_SIZE
        and (d['bbox'][3] - d['bbox'][1]) >= Config.DETECTION_MIN_SIZE
    ]

    # Apply cross-model NMS to remove overlapping duplicates
    detections = _nms(detections, iou_threshold=Config.DETECTION_NMS_IOU)

    # Keep only top-N highest-confidence detections
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    detections = detections[:Config.DETECTION_MAX_PER_IMAGE]

    return detections

def segment_polyp(image_np, bbox):
    """Run segmentation on detected ROI"""
    if segmentation_model is None:
        return None

    try:
        x1, y1, x2, y2 = bbox
        roi = image_np[y1:y2, x1:x2]

        # Prepare for segmentation
        roi_pil = Image.fromarray(roi)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        roi_tensor = transform(roi_pil).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            mask_logits = segmentation_model(roi_tensor)
            mask_pred = (mask_logits > 0.5).float()
            mask_np = mask_pred.cpu().numpy()[0, 0]

        # Resize back to ROI size
        mask_np = cv2.resize(mask_np,
                            (x2-x1, y2-y1),
                            interpolation=cv2.INTER_NEAREST)

        return mask_np.astype(np.uint8)

    except Exception as e:
        print(f"   ⚠️  Segmentation failed: {e}")
        return None

# ==========================================
# PROCESS SINGLE IMAGE
# ==========================================
def process_image(image_path):
    """Complete pipeline for one image"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Detection
    detections = detect_polyps(image_path)

    if len(detections) == 0:
        return {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'polyps_detected': 0,
            'polyps': [],
            'overall_decision': 'NO POLYPS DETECTED',
            'overall_recommendation': 'Continue Screening'
        }

    # Step 2: Process each detection
    polyp_results = []

    for idx, detection in enumerate(detections):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox

        # Clip bounding box to image boundaries (don't skip valid detections)
        img_h, img_w = image_rgb.shape[:2]
        x1_clipped = max(0, min(x1, img_w - 1))
        y1_clipped = max(0, min(y1, img_h - 1))
        x2_clipped = max(0, min(x2, img_w))
        y2_clipped = max(0, min(y2, img_h))

        # Only skip if clipped box is invalid (zero width/height)
        if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
            print(f"   ⚠️  Invalid bbox after clipping: [{x1}, {y1}, {x2}, {y2}] → [{x1_clipped}, {y1_clipped}, {x2_clipped}, {y2_clipped}] - Skipping polyp {idx+1}")
            continue

        # Use clipped coordinates
        bbox = [x1_clipped, y1_clipped, x2_clipped, y2_clipped]
        x1, y1, x2, y2 = bbox

        # Crop ROI
        roi = image_rgb[y1:y2, x1:x2]

        # Validate ROI is not empty
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"   ⚠️  Empty ROI for polyp {idx+1} - Skipping")
            continue

        # Step 3: Segmentation (optional)
        mask = segment_polyp(image_rgb, bbox)

        # Step 4: Risk prediction
        # MLP Calibrator is available even without Decision Tree experts
        core_models_ready = (ssl_model is not None and
                             kmeans is not None and
                             (mlp_model is not None or len(local_experts) > 0))

        if core_models_ready:
            prediction = predict_risk(roi, ssl_model, kmeans, scaler, local_experts)
        else:
            prediction = {
                'risk_probability'  : 0.5,
                'cluster_id'        : -1,
                'decision'          : 'UNCERTAIN',
                'recommendation'    : 'Models not loaded',
                'confidence'        : 'Uncertain',
                'prediction_source' : 'no_models',
            }

        polyp_results.append({
            'polyp_id'          : idx + 1,
            'detection_model'   : detection['model'],
            'detection_confidence': detection['confidence'],
            'bbox'              : bbox,
            'has_mask'          : mask is not None,
            'risk_probability'  : prediction['risk_probability'],
            'cluster_id'        : prediction['cluster_id'],
            'decision'          : prediction['decision'],
            'recommendation'    : prediction['recommendation'],
            'confidence'        : prediction['confidence'],
            'prediction_source' : prediction.get('prediction_source', 'unknown'),
        })

    # Handle case where all detections were invalid
    if len(polyp_results) == 0:
        return {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'polyps_detected': len(detections),
            'polyps': [],
            'overall_decision': 'INVALID DETECTIONS',
            'overall_recommendation': 'All detected polyps had invalid bounding boxes'
        }

    # Overall decision (most severe polyp determines action)
    max_risk = max(p['risk_probability'] for p in polyp_results)

    if max_risk >= Config.ASGE_HIGH_CONFIDENCE:
        overall_decision = "HIGH RISK DETECTED"
        overall_recommendation = "Resect & Discard Protocol"
    elif max_risk >= Config.ASGE_UNCERTAINTY:
        overall_decision = "UNCERTAIN RISK DETECTED"
        overall_recommendation = "Require Biopsy/Pathology Review"
    else:
        overall_decision = "LOW RISK POLYPS"
        overall_recommendation = "Surveillance Protocol"

    return {
        'image_name': image_path.name,
        'image_path': str(image_path),
        'polyps_detected': len(detections),
        'polyps': polyp_results,
        'overall_decision': overall_decision,
        'overall_recommendation': overall_recommendation,
        'max_risk_probability': float(max_risk)
    }

# ==========================================
# VISUALIZATION
# ==========================================
def visualize_result(image_path, result):
    """Create visualization of inference result"""
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)

    # Draw detections
    for polyp in result['polyps']:
        x1, y1, x2, y2 = polyp['bbox']
        decision = polyp['decision']
        risk_prob = polyp['risk_probability']

        # Color based on decision
        if decision == 'HIGH RISK':
            color = 'red'
        elif decision == 'UNCERTAIN HIGH RISK':
            color = 'orange'
        else:
            color = 'green'

        # Draw box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # Add label inside the box (top-left corner) so it never goes off-canvas
        label = f"{decision}  P={risk_prob:.2f}  Cluster {polyp['cluster_id']}"
        ax.text(x1 + 4, y1 + 14, label,
               bbox=dict(facecolor=color, alpha=0.8, pad=2, boxstyle='round,pad=0.2'),
               fontsize=8, color='white', weight='bold',
               verticalalignment='top')

    # Title
    ax.set_title(f"{result['overall_decision']}\n{result['overall_recommendation']}",
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    # Save
    viz_path = Config.VISUALIZATION_DIR / f"{image_path.stem}_result.png"
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    plt.close()

    return str(viz_path)

# ==========================================
# MAIN INFERENCE
# ==========================================
def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STARTING INFERENCE")
    print("=" * 80)

    # Collect images
    image_paths = []
    for ext in ['**/*.jpg', '**/*.jpeg', '**/*.png']:
        image_paths.extend(Config.DATASET_2_ROOT.glob(ext))

    image_paths = sorted(list(set(image_paths)))
    print(f"   Found {len(image_paths):,} images")

    # Process all images
    all_results = []
    statistics = defaultdict(int)

    for img_path in tqdm(image_paths, desc="Processing images"):
        result = process_image(img_path)

        if result is None:
            statistics['failed'] += 1
            continue

        all_results.append(result)

        # Update statistics
        statistics['processed'] += 1
        statistics['total_polyps'] += result['polyps_detected']

        if result['polyps_detected'] > 0:
            statistics['images_with_polyps'] += 1

            # Count by decision
            for polyp in result['polyps']:
                decision = polyp['decision']
                if decision == 'HIGH RISK':
                    statistics['high_risk_polyps'] += 1
                elif decision == 'UNCERTAIN HIGH RISK':
                    statistics['uncertain_polyps'] += 1
                else:
                    statistics['low_risk_polyps'] += 1

        # Save individual report
        report_path = Config.REPORTS_DIR / f"{img_path.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Visualize
        if result['polyps_detected'] > 0:
            visualize_result(img_path, result)

        # Save high-risk and uncertain cases
        if result.get('max_risk_probability', 0) >= Config.ASGE_HIGH_CONFIDENCE:
            shutil.copy(img_path, Config.HIGH_RISK_CASES_DIR / img_path.name)
        elif result.get('max_risk_probability', 0) >= Config.ASGE_UNCERTAINTY:
            shutil.copy(img_path, Config.UNCERTAIN_CASES_DIR / img_path.name)

    # Save master report
    master_report = {
        'timestamp': datetime.now().isoformat(),
        'statistics': dict(statistics),
        'asge_thresholds': {
            'high_confidence': Config.ASGE_HIGH_CONFIDENCE,
            'uncertainty': Config.ASGE_UNCERTAINTY
        },
        'results': all_results
    }

    master_path = Config.INFERENCE_OUTPUT / 'master_report_asge.json'
    with open(master_path, 'w') as f:
        json.dump(master_report, f, indent=2)

    print(f"\n💾 Master report saved: {master_path}")

    # Print final statistics
    print("\n" + "=" * 80)
    print(" " * 25 + "INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Images processed: {statistics['processed']:,}")
    print(f"   Images with polyps: {statistics['images_with_polyps']:,}")
    print(f"   Total polyps detected: {statistics['total_polyps']:,}")
    print(f"\n📊 ASGE Risk Distribution:")
    print(f"   HIGH RISK (≥{Config.ASGE_HIGH_CONFIDENCE}): {statistics['high_risk_polyps']:,} polyps")
    print(f"   UNCERTAIN ({Config.ASGE_UNCERTAINTY}-{Config.ASGE_HIGH_CONFIDENCE}): {statistics['uncertain_polyps']:,} polyps")
    print(f"   LOW RISK (<{Config.ASGE_UNCERTAINTY}): {statistics['low_risk_polyps']:,} polyps")
    print(f"\n📁 Outputs:")
    print(f"   Reports: {Config.REPORTS_DIR}")
    print(f"   Visualizations: {Config.VISUALIZATION_DIR}")
    print(f"   High Risk Cases: {Config.HIGH_RISK_CASES_DIR}")
    print(f"   Uncertain Cases: {Config.UNCERTAIN_CASES_DIR}")

if __name__ == '__main__':
    main()
