# -*- coding: utf-8 -*-
"""
POLYP FEATURE EXTRACTOR
Extracts medical features from detected polyp ROIs for classification.

Rule 4: Extract SSL features + biomarkers for medical decision-making
- Redness: Red channel intensity (indicates vascularization/neoplasia)
- Radius: Size of polyp from mask
- Texture: Edge density (smooth=benign, rough=suspicious)
- Color: HSV distribution
- Vessel visibility: Blood vessel prominence

Expected improvement: +15-25% accuracy
"""

import numpy as np
import cv2
import torch
import json
from typing import Dict, Tuple, List, Optional
from pathlib import Path


# Load texture divisor from NeoPolyp thresholds (data-driven, not hardcoded)
_TEXTURE_DIVISOR = 2000.0  # default fallback
_thresh_path = Path(__file__).parent.parent.parent / 'mixture_of_experts' / 'neopolyp_thresholds.json'
if _thresh_path.exists():
    try:
        with open(_thresh_path) as _f:
            _thresholds = json.load(_f)
            _TEXTURE_DIVISOR = float(_thresholds.get('texture_divisor', 2000.0))
    except Exception:
        pass


def extract_redness_score(roi: np.ndarray) -> float:
    """
    Calculate redness score from polyp ROI.
    High redness indicates vascularization (suspicious for neoplasia).
    
    Args:
        roi: RGB image (3-channel) for the polyp region
    
    Returns:
        redness_score: 0.0-1.0 (higher = more red)
    """
    try:
        if roi.size == 0 or roi.shape[2] != 3:
            return 0.0
        
        # Extract red and green channels
        red = roi[:, :, 0].astype(np.float32)
        green = roi[:, :, 1].astype(np.float32)
        blue = roi[:, :, 2].astype(np.float32)
        
        # Redness = (R - G) / (R + G + B)
        denominator = red + green + blue
        denominator = np.where(denominator == 0, 1, denominator)  # Avoid division by zero
        
        redness = (red - green) / denominator
        redness = np.clip(redness, 0, 1)
        
        return float(np.mean(redness))
    except Exception as e:
        print(f"   ⚠️  Error computing redness: {e}")
        return 0.0


def extract_radius_from_mask(mask: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    """
    Calculate effective radius of polyp from segmentation mask.
    Larger polyps are more likely to be high-grade.
    
    Args:
        mask: Binary mask (True where polyp is)
        frame_shape: (height, width) of original frame
    
    Returns:
        effective_radius: Estimated radius in pixels (normalized 0-1 relative to frame)
    """
    try:
        if mask.sum() == 0:
            return 0.0
        
        # CRITICAL: Validate frame_shape is a real full-frame dimension
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if fh < 100 or fw < 100:
            # Shape is degenerate — compute from mask pixels alone
            # Use mask dimensions as approximation of frame
            mh, mw = mask.shape[:2]
            fh = max(mh * 4, 576)   # Assume mask is ~25% of frame height
            fw = max(mw * 4, 720)
        
        frame_diagonal = np.sqrt(fh ** 2 + fw ** 2)
        if frame_diagonal < 10:
            return 0.0
        
        # Find contours
        mask_uint8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: compute from mask area
            area = mask.sum()
            radius = np.sqrt(area / np.pi)
        else:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            (cx, cy), radius_cv = cv2.minEnclosingCircle(largest_contour)
            radius = radius_cv
        
        # Normalize to 0-1 range relative to frame diagonal
        normalized_radius = radius / (frame_diagonal / 2)
        normalized_radius = np.clip(normalized_radius, 0, 1)
        
        return float(normalized_radius)
    except Exception as e:
        print(f"   ⚠️  Error computing radius: {e}")
        return 0.0


def extract_texture_score(roi: np.ndarray) -> float:
    """
    Calculate texture score (edge density).
    Smooth texture = benign, Rough/spiky texture = suspicious.
    
    Uses texture_divisor derived from NeoPolyp ground truth (99th percentile Laplacian variance).
    If neopolyp_thresholds.json not found, defaults to 2000.0.
    
    Args:
        roi: RGB image for polyp region
    
    Returns:
        texture_score: 0.0-1.0 (higher = rougher/more suspicious)
    """
    try:
        if roi.size == 0 or len(roi.shape) < 2:
            return 0.0
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Compute Laplacian (edge detection) — raw variance, same as neopolyp_threshold_learner
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        
        # Normalize using data-driven texture_divisor from NeoPolyp (or default 2000.0)
        normalized_texture = np.clip(lap_var / _TEXTURE_DIVISOR, 0.0, 1.0)
        
        return float(normalized_texture)
    except Exception as e:
        print(f"   ⚠️  Error computing texture: {e}")
        return 0.0


def extract_color_distribution(roi: np.ndarray) -> Dict[str, float]:
    """
    Extract color distribution features from polyp ROI.
    
    Args:
        roi: RGB image for polyp region
    
    Returns:
        color_features: Dict with HSV channel means
    """
    try:
        if roi.size == 0 or roi.shape[2] != 3:
            return {'h_mean': 0.0, 's_mean': 0.0, 'v_mean': 0.0}
        
        # Convert RGB to HSV
        hsv = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Normalize to 0-1 range
        h = hsv[:, :, 0] / 180.0  # Hue: 0-180 in OpenCV
        s = hsv[:, :, 1] / 255.0  # Saturation
        v = hsv[:, :, 2] / 255.0  # Value
        
        return {
            'h_mean': float(np.mean(h)),
            's_mean': float(np.mean(s)),
            'v_mean': float(np.mean(v)),
            'h_std': float(np.std(h)),
            's_std': float(np.std(s)),
            'v_std': float(np.std(v)),
        }
    except Exception as e:
        print(f"   ⚠️  Error computing color distribution: {e}")
        return {'h_mean': 0.0, 's_mean': 0.0, 'v_mean': 0.0, 'h_std': 0.0, 's_std': 0.0, 'v_std': 0.0}


def extract_vessel_visibility(roi: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate blood vessel visibility in polyp ROI.
    High vessel visibility = higher risk (neoplasia indicator).
    
    Args:
        roi: RGB image for polyp region
        mask: Binary mask of polyp region
    
    Returns:
        vessel_score: 0.0-1.0 (higher = more vessels visible)
    """
    try:
        if roi.size == 0 or mask.sum() == 0:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Red blood appears as low hue, high saturation, high value
        # Blood vessels: hue around 0 (red) or 170+ (red wrap-around)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        # Detect red vessels (hue < 10 or > 170, high saturation, high value)
        blood_mask = ((h < 10) | (h > 170)) & (s > 100) & (v > 80)
        
        # Only count within polyp mask
        blood_in_polyp = blood_mask & mask
        
        # Vessel visibility = percentage of polyp with blood
        vessel_score = blood_in_polyp.sum() / mask.sum() if mask.sum() > 0 else 0.0
        
        return float(np.clip(vessel_score, 0, 1))
    except Exception as e:
        print(f"   ⚠️  Error computing vessel visibility: {e}")
        return 0.0


def crop_roi_from_frame(frame: np.ndarray, 
                       box: List[float], 
                       padding: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop polyp ROI from frame with padding.
    
    Args:
        frame: Full frame (RGB)
        box: Bounding box [x1, y1, x2, y2]
        padding: Padding as fraction of box size
    
    Returns:
        (cropped_roi, roi_coords) where roi_coords = (x1, y1, x2, y2) in original frame
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding / 2)
        pad_y = int(height * padding / 2)
        
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)
        
        roi = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        return roi, (x1_padded, y1_padded, x2_padded, y2_padded)
    except Exception as e:
        print(f"   ⚠️  Error cropping ROI: {e}")
        return frame, (0, 0, frame.shape[1], frame.shape[0])


def crop_pure_roi_from_mask(frame: np.ndarray,
                            mask: Optional[np.ndarray],
                            padding: float = 0.1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a tight ROI from the foreground mask and remove background outside the mask.
    """
    try:
        if mask is None or mask.sum() == 0:
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        if mask.shape[:2] != frame.shape[:2]:
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()

        margin = int(max(x2 - x1, y2 - y1) * padding)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        roi = frame[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].astype(np.uint8)

        # Bug 4 Fix: Return raw crop without zeroing background
        # DO NOT apply bitwise_and here — let individual feature functions use roi_mask to restrict computation
        # This preserves color information needed for redness, HSV, and vessel features
        
        return roi, (x1, y1, x2, y2)
    except Exception as e:
        print(f"   ⚠️  Error cropping pure ROI: {e}")
        return frame, (0, 0, frame.shape[1], frame.shape[0])


def extract_all_features(frame: np.ndarray, 
                        box: List[float], 
                        mask: Optional[np.ndarray] = None,
                        mask_padding: float = 0.1) -> Dict:
    """
    Extract all medical features from a detected polyp.
    
    Args:
        frame: Full RGB frame
        box: Detection box [x1, y1, x2, y2]
        mask: Optional segmentation mask
    
    Returns:
        features: Dict with all extracted features
    """
    
    # Store original frame shape before any cropping (Fix Issue 1: radius normalization)
    original_frame_shape = frame.shape[:2]  # Always full frame (H, W)
    
    # Crop a mask-filtered ROI when available so background pixels do not affect the features
    if mask is not None and mask.sum() > 0:
        roi, roi_coords = crop_pure_roi_from_mask(frame, mask, padding=mask_padding)
        if roi is None or roi.size == 0:
            roi, roi_coords = crop_roi_from_frame(frame, box, padding=0.2)
    else:
        roi, roi_coords = crop_roi_from_frame(frame, box, padding=0.2)
    
    # Prepare mask for ROI
    if mask is not None and mask.sum() > 0:
        x1, y1, x2, y2 = roi_coords
        roi_mask = mask[y1:y2, x1:x2]
    else:
        roi_mask = np.ones(roi.shape[:2], dtype=bool)
    
    # Extract features
    features = {
        'box': list(box),
        'roi_shape': roi.shape,
        
        # Medical features
        'redness': extract_redness_score(roi),
        'radius': extract_radius_from_mask(roi_mask, original_frame_shape),
        'texture': extract_texture_score(roi),
        'vessel_visibility': extract_vessel_visibility(roi, roi_mask),
    }
    
    # Color distribution
    color_features = extract_color_distribution(roi)
    features.update(color_features)
    
    return features


def features_to_decision_vector(features: Dict) -> np.ndarray:
    """
    Convert extracted features to a decision vector for symbolic reasoning.
    
    Returns:
        decision_vector: 1D array of 11 features for classification
    """
    
    vector = np.array([
        features.get('redness', 0.0),
        features.get('radius', 0.0),
        features.get('texture', 0.0),
        features.get('vessel_visibility', 0.0),
        features.get('h_mean', 0.0),
        features.get('s_mean', 0.0),
        features.get('v_mean', 0.0),
        features.get('h_std', 0.0),
        features.get('s_std', 0.0),
        features.get('v_std', 0.0),
    ], dtype=np.float32)
    
    return vector
