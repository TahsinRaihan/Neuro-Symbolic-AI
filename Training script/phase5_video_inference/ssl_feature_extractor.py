# -*- coding: utf-8 -*-
"""
SSL FEATURE EXTRACTOR FOR VIDEO INFERENCE
Extracts 444-dimensional Fact Vectors (384 SSL + 60 biomarkers) from video frames for Rule 3 Symbolic Reasoning.
Replicates the Phase 3 feature extraction pipeline at inference time.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import timm
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')


class ViTEncoder(nn.Module):
    """ViT-Small SSL encoder - same as Phase 3"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224',
                                         pretrained=False,
                                         num_classes=0,
                                         img_size=256)
        self.backbone_dim = 384

    def forward(self, x):
        """Extract 384-dimensional SSL features"""
        return self.backbone(x)


def load_ssl_encoder(device):
    """Load the trained SSL encoder from Phase 2"""
    print("   Loading SSL encoder...")
    
    # Try different checkpoint locations
    possible_paths = [
        Path(__file__).parent.parent.parent / 'thesis_outputs' / 'ssl_outputs' / 'ssl_encoder_final.pth',
        Path(__file__).parent.parent.parent / 'thesis_outputs' / 'ssl_outputs' / 'ssl_model_final.pth',
    ]

    model = ViTEncoder().to(device)
    model._ssl_weights_loaded = False

    for path in possible_paths:
        if path.exists():
            try:
                state_dict = torch.load(str(path), map_location=device)
                if isinstance(state_dict, dict) and 'backbone' in str(state_dict.keys()):
                    if 'backbone.pos_embed' in state_dict:
                        model.backbone.load_state_dict({k.replace('backbone.', ''): v
                                                       for k, v in state_dict.items()
                                                       if k.startswith('backbone.')})
                    else:
                        model.load_state_dict(state_dict)
                else:
                    model.backbone.load_state_dict(state_dict)
                
                model._ssl_weights_loaded = True
                print(f"      ✅ SSL encoder loaded from: {path}")
                model.eval()
                return model
            except Exception as e:
                print(f"      ⚠️  Failed loading {path}: {e}")
                continue

    print(f"      ⚠️  SSL encoder using RANDOM WEIGHTS — predictions will be unreliable")
    return model


def pad_to_square(image_np, target_size=256):
    """Pad image to square while preserving aspect ratio"""
    h, w = image_np.shape[:2]
    max_side = max(h, w)
    scale = target_size / max_side
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize
    image_resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized
    
    return canvas


def extract_ssl_features(frame_rgb, ssl_model, device, transform=None):
    """
    Extract 384-dimensional SSL features from frame using ViT encoder.
    
    Args:
        frame_rgb: RGB frame (H, W, 3)
        ssl_model: ViT encoder model
        device: torch device
        transform: Optional preprocessing transform
    
    Returns:
        ssl_features: (384,) numpy array
    """
    try:
        # Pad frame to square while preserving aspect ratio
        frame_padded = pad_to_square(frame_rgb, 256)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(frame_padded)
        
        # Default transform: ImageNet normalization
        if transform is None:
            from torchvision import transforms as T
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        
        # Extract features
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            ssl_features = ssl_model(image_tensor).cpu().numpy().flatten()
        
        assert len(ssl_features) == 384, f"Expected 384 SSL features, got {len(ssl_features)}"
        return ssl_features.astype(np.float32)
    
    except Exception as e:
        print(f"      ⚠️  SSL extraction failed: {e}")
        return np.zeros(384, dtype=np.float32)


def compute_cielab_histograms(image_np):
    """Extract 9 CIELAB histogram features (3 L* + 3 a* + 3 b*)"""
    try:
        # Convert RGB to LAB
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
        # Extract histograms for each channel
        hist_l = cv2.calcHist([image_lab], [0], None, [256], [0, 256])
        hist_a = cv2.calcHist([image_lab], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_lab], [2], None, [256], [0, 256])
        
        # Get statistics (mean, std, max) for each channel
        features = np.concatenate([
            [hist_l.mean(), hist_l.std(), hist_l.max()],
            [hist_a.mean(), hist_a.std(), hist_a.max()],
            [hist_b.mean(), hist_b.std(), hist_b.max()]
        ])
        
        return features.astype(np.float32)
    except Exception as e:
        return np.zeros(9, dtype=np.float32)


def compute_saturation_histogram(image_np):
    """Extract 16 saturation histogram features"""
    try:
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        saturation = image_hsv[:, :, 1]
        
        # Compute histogram with 16 bins
        hist_sat = cv2.calcHist([saturation], [0], None, [16], [0, 256])
        
        # Normalize and flatten
        hist_sat = hist_sat.flatten() / (hist_sat.sum() + 1e-6)
        
        assert len(hist_sat) == 16
        return hist_sat.astype(np.float32)
    except Exception as e:
        return np.zeros(16, dtype=np.float32)


def compute_haralick_features(image_np):
    """Extract 13 Haralick texture features"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Compute co-occurrence matrix (Haralick)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract Haralick features
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            try:
                feat = graycoprops(glcm, prop)
                features.extend(feat.flatten()[:2])  # Take first 2 values per property
            except:
                features.extend([0.0, 0.0])
        
        # Pad to 13 features if needed
        while len(features) < 13:
            features.append(0.0)
        features = features[:13]
        
        return np.array(features, dtype=np.float32)
    except Exception as e:
        return np.zeros(13, dtype=np.float32)


def compute_lbp_features(image_np):
    """Extract 19 Local Binary Pattern histogram features"""
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Compute LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Compute histogram (19 bins for uniform LBP)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=19, range=(0, 19))
        
        # Normalize
        hist_lbp = hist_lbp.astype(np.float32) / (hist_lbp.sum() + 1e-6)
        
        assert len(hist_lbp) == 19
        return hist_lbp
    except Exception as e:
        return np.zeros(19, dtype=np.float32)


def compute_shape_features(image_np):
    """Extract 3 shape features: edge density, area, compactness"""
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Edge detection (Sobel)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Area (non-black pixels)
        area = np.sum(gray > 10) / gray.size
        
        # Compactness (perimeter/area ratio approximation using Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        compactness = np.sum(np.abs(laplacian) > 0) / (np.sum(gray > 10) + 1e-6)
        
        return np.array([edge_density, area, compactness], dtype=np.float32)
    except Exception as e:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def extract_biomarkers(frame_rgb):
    """
    Extract 60-dimensional biomarker vector:
    - 9 CIELAB histogram
    - 16 Saturation histogram
    - 13 Haralick texture
    - 19 LBP histogram
    - 3 Shape features
    """
    biomarkers = np.concatenate([
        compute_cielab_histograms(frame_rgb),      # 9
        compute_saturation_histogram(frame_rgb),   # 16
        compute_haralick_features(frame_rgb),      # 13
        compute_lbp_features(frame_rgb),           # 19
        compute_shape_features(frame_rgb)          # 3
    ])
    
    assert len(biomarkers) == 60, f"Expected 60 biomarkers, got {len(biomarkers)}"
    return biomarkers.astype(np.float32)


def extract_444_features(frame_rgb, ssl_model, device, transform=None):
    """
    Extract complete 444-dimensional Fact Vector for symbolic reasoning:
    - 384 SSL features (Neural stream)
    - 60 Biomarkers (Symbolic stream)
    
    Args:
        frame_rgb: RGB frame (H, W, 3)
        ssl_model: ViT encoder model
        device: torch device
        transform: Optional preprocessing
    
    Returns:
        features_444: (444,) numpy array with full feature vector
    """
    # Stream A: SSL features (384 dimensions)
    ssl_features = extract_ssl_features(frame_rgb, ssl_model, device, transform)
    
    # Stream B: Biomarkers (60 dimensions)
    biomarkers = extract_biomarkers(frame_rgb)
    
    # Combine: 384 + 60 = 444
    features_444 = np.concatenate([ssl_features, biomarkers])
    
    assert len(features_444) == 444, f"Expected 444 features, got {len(features_444)}"
    return features_444.astype(np.float32)
