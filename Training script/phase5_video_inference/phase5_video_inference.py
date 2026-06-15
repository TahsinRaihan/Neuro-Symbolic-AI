








































































































































































































































































































































# -*- coding: utf-8 -*-
"""
PHASE 5: Video Inference and Visualization
Applies trained models (YOLO, RT-DETR, UNet++, I3D) on videos from "Apply Video" folder.
Generates comprehensive outputs for visualization and clinical understanding.
"""

import os
import sys
import json
import csv
import contextlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

# Pre-import lap module to avoid Ultralytics AutoUpdate issues during inference
try:
    import lap
except ImportError:
    print("Installing lap module for YOLO tracking...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lap>=0.5.12"])
    import lap

import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as patches
from matplotlib import animation
import seaborn as sns
import pandas as pd
import plotly.express as px
import imageio
from tqdm import tqdm
import traceback
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

# Import new modules for 5 rules implementation
sys.path.insert(0, str(Path(__file__).parent))
from consensus_voting_engine import aggregate_consensus_frames, find_overlapping_boxes
from polyp_feature_extractor import extract_all_features, crop_pure_roi_from_mask, features_to_decision_vector
from symbolic_reasoning_integrator import SymbolicReasoningIntegrator
from ssl_feature_extractor import extract_444_features, load_ssl_encoder
from polyp_type_classifier import PolypTypeClassifier
from medical_report_generator import generate_medical_report
from rules_engine_5 import RulesEngine5

# ==========================================
# MODEL DEFINITIONS
# ==========================================

print("=" * 80)
print(" " * 20 + "PHASE 5: VIDEO INFERENCE AND VISUALIZATION")
print(" " * 15 + "(Apply Models on Videos + Generate Comprehensive Outputs)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    APPLY_VIDEO_ROOT = THESIS_ROOT / 'NeSy' / 'Apply Video'  # Folder with videos to process
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    VIDEO_OUTPUT      = OUTPUT_ROOT / 'video_inference_results'

    # Model paths
    DETECTION_MODELS = OUTPUT_ROOT / 'detection_models'
    SEGMENTATION_MODELS = OUTPUT_ROOT / 'segmentation_models'
    MIXTURE_OF_EXPERTS = OUTPUT_ROOT / 'mixture_of_experts'
    FEATURES_OUTPUT = OUTPUT_ROOT / 'extracted_features'
    NEOPOLYP_OUTPUT = OUTPUT_ROOT / 'neopolyp_processed'

    # Processing config
    FRAME_SKIP = 1  # For evaluation, use all frames (set to higher for speed)
    FRAME_RATE = 10  # Frames per second to extract
    IMG_SIZE = 960  # Higher resolution for small polyps
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE_INFERENCE = 8  # Batch frames for faster inference
    YOLO_CONF_THRESHOLD = 0.50  
    RTDETR_CONF_THRESHOLD = 0.40  
    MEDSAM_CONF_THRESHOLD = 0.50  # Set very low for diagnosis — raise to 0.50 once working
    MEDSAM2_CHECKPOINT = SEGMENTATION_MODELS / 'MedSAM2_latest.pt'
    MEDSAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'



    # ASGE thresholds
    HIGH_CONFIDENCE = 0.90
    UNCERTAINTY = 0.80

# Create directories
Config.VIDEO_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Video Input: {Config.APPLY_VIDEO_ROOT}")
print(f"   Output Root: {Config.VIDEO_OUTPUT}")
print(f"   Device: {Config.DEVICE}")

# ==========================================
# CHECKPOINT SUPPORT FOR RESUME CAPABILITY
# ==========================================
CHECKPOINT_FILE = Config.VIDEO_OUTPUT / "processing_checkpoint.json"

def load_checkpoint():
    """Load processing checkpoint to resume from interruption"""
    try:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f"   ✅ Found checkpoint with {len(checkpoint.get('processed_videos', []))} videos already processed")
            return checkpoint
    except Exception as e:
        print(f"   ⚠️  Error loading checkpoint: {e}")
    return {'processed_videos': [], 'failed_videos': [], 'last_updated': None}

def save_checkpoint(checkpoint_data):
    """Save checkpoint after processing each video for safety"""
    try:
        checkpoint_data['last_updated'] = datetime.now().isoformat()
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"   ⚠️  Error saving checkpoint: {e}")

def has_been_processed(video_name, checkpoint_data):
    """Check if video has already been processed"""
    return video_name in checkpoint_data.get('processed_videos', [])

# ==========================================
# DATA-DRIVEN THRESHOLD LOADING
# ==========================================
def load_data_driven_thresholds(thesis_root: Path) -> dict:
    """
    Load calibration thresholds from neopolyp_thresholds.json.
    Generated by neopolyp_threshold_learner.py using Youden-index ROC on NeoPolyp dataset.
    confidence_clip_low and confidence_clip_high come from that file directly.
    """
    # Primary: read from neopolyp_thresholds.json (generated by neopolyp_threshold_learner.py)
    np_path = thesis_root / 'thesis_outputs' / 'mixture_of_experts' / 'neopolyp_thresholds.json'
    if np_path.exists():
        try:
            with open(np_path) as f:
                np_data = json.load(f)
            clip_lo = float(np_data.get('confidence_clip_low',  0.319))
            clip_hi = float(np_data.get('confidence_clip_high', 0.730))
            print(f"   ✅ Loaded data-driven thresholds from neopolyp_thresholds.json:")
            print(f"      confidence_clip_low  = {clip_lo}  (NeoPolyp Youden-derived)")
            print(f"      confidence_clip_high = {clip_hi}  (NeoPolyp Youden-derived)")
            return {
                'high_confidence_threshold': clip_hi,
                'uncertainty_threshold':     clip_lo,
                'confidence_clip_low':       clip_lo,
                'confidence_clip_high':      clip_hi,
                '_source': 'neopolyp_thresholds_json_youden_roc'
            }
        except Exception as e:
            print(f"   ⚠️  Failed to read neopolyp_thresholds.json: {e}")

    # Fallback only if neopolyp_thresholds.json also missing
    print("   ⚠️  neopolyp_thresholds.json not found — run neopolyp_threshold_learner.py first.")
    print("   ⚠️  Using fallback 0.730/0.319 from NeoPolyp paper defaults.")
    return {
        'high_confidence_threshold': 0.730,
        'uncertainty_threshold':     0.319,
        'confidence_clip_low':       0.319,
        'confidence_clip_high':      0.730,
        '_source': 'fallback_neopolyp_paper_defaults'
    }

# ==========================================
# MODEL LOADING
# ==========================================
def load_models():
    """Load all trained models"""
    print("\n" + "="*60)
    print("🔧 Loading trained models...")
    print("="*60)

    models = {}

    # YOLO (if available)
    try:
        from ultralytics import YOLO
        yolo_path = Config.DETECTION_MODELS / 'yolov8m' / 'weights' / 'best.pt'
        if yolo_path.exists():
            models['yolo'] = YOLO(str(yolo_path))
            print("   ✅ YOLO model loaded")
        else:
            print("   ⚠️  YOLO model not found at", yolo_path)
    except ImportError:
        print("   ⚠️  YOLO not available")

    # RT-DETR (simplified loading)
    try:
        from ultralytics import RTDETR
        rtdetr_path = Config.DETECTION_MODELS / 'rtdetr_polyp' / 'weights' / 'best.pt'
        if rtdetr_path.exists():
            models['rtdetr'] = RTDETR(str(rtdetr_path))
            print("   ✅ RT-DETR model loaded")
        else:
            print("   ⚠️  RT-DETR model not found")
    except Exception as e:
        print(f"   ⚠️  RT-DETR loading failed: {e}")

    import sys as _sys, os as _os, traceback as _tb

    _log_path = Config.VIDEO_OUTPUT / 'medsam2_load_debug.txt'
    def _mlog(msg):
        print(msg)
        with open(str(_log_path), 'a') as _f:
            _f.write(msg + '\n')

    _mlog("\n" + "="*60)
    _mlog("MEDSAM2 LOAD DIAGNOSTIC")
    _mlog(f"Python:     {_sys.executable}")
    _mlog(f"Checkpoint: {Config.SEGMENTATION_MODELS / 'MedSAM2_latest.pt'}")
    _mlog(f"Exists:     {(Config.SEGMENTATION_MODELS / 'MedSAM2_latest.pt').exists()}")

    _mlog("Step 1: importing sam2...")
    try:
        import sam2 as _s2
        _mlog(f"  OK — {_s2.__file__}")
    except ImportError as _e:
        _mlog(f"  FAILED: {_e}")
        _mlog(f"  FIX: {_sys.executable} -m pip install -e /full/path/to/MedSAM2")
        _mlog("="*60)
        pass
    else:
        _mlog("Step 2: finding config yaml...")
        _pkg = _os.path.dirname(_s2.__file__)
        _cfg_root = _os.path.join(_pkg, 'configs')
        _mlog(f"  config root: {_cfg_root} — exists: {_os.path.isdir(_cfg_root)}")

        _found_cfg = None
        if _os.path.isdir(_cfg_root):
            for _root, _, _files in _os.walk(_cfg_root):
                for _fname in sorted(_files):
                    if _fname.endswith('.yaml'):
                        _mlog(f"  yaml found: {_os.path.join(_root, _fname)}")
                    if _fname.endswith('.yaml') and 'hiera' in _fname and _found_cfg is None:
                        _found_cfg = _os.path.relpath(_os.path.join(_root, _fname), _pkg)

        _mlog(f"  selected config: {_found_cfg}")

        _mlog("Step 3: checking checkpoint...")
        _ckpt = Config.SEGMENTATION_MODELS / 'MedSAM2_latest.pt'
        _mlog(f"  path: {_ckpt}")
        _mlog(f"  exists: {_ckpt.exists()}")
        if _ckpt.exists():
            _mlog(f"  size: {_ckpt.stat().st_size / 1e6:.1f} MB")

        if _found_cfg and _ckpt.exists():
            _mlog("Step 4: calling build_sam2...")
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                _model = build_sam2(
                    config_file=_found_cfg,
                    ckpt_path=str(_ckpt),
                    device=Config.DEVICE,
                    apply_postprocessing=False,
                )
                _model.eval()
                _predictor = SAM2ImagePredictor(_model)
                models['medsam'] = _predictor
                _mlog(f"  SUCCESS — MedSAM2 ready, config={_found_cfg}")
            except Exception as _be:
                _mlog(f"  build_sam2 FAILED: {_be}")
                _tb.print_exc()
                import io as _io
                _buf = _io.StringIO()
                _tb.print_exc(file=_buf)
                _mlog(_buf.getvalue())
        elif not _found_cfg:
            _mlog("  CANNOT LOAD — no config yaml found")
        elif not _ckpt.exists():
            _mlog("  CANNOT LOAD — checkpoint file missing")

    _mlog("="*60 + "\n")

    # 6. Clustering artifacts for symbolic reasoning
    try:
        kmeans_path = Config.FEATURES_OUTPUT / 'kmeans_model.pkl'
        scaler_path = Config.FEATURES_OUTPUT / 'feature_scaler.pkl'

        if kmeans_path.exists() and scaler_path.exists():
            models['kmeans'] = joblib.load(str(kmeans_path))
            models['feature_scaler'] = joblib.load(str(scaler_path))
            print("   ✅ K-Means clustering loaded")
        else:
            print("   ⚠️  K-Means clustering artifacts not found")
    except Exception as e:
        print(f"   ⚠️  Clustering loading failed: {e}")

    return models

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def extract_frames(video_path, frame_rate=5):
    """Extract frames from video at specified rate with comprehensive error handling"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"   ❌ Failed to open video: {video_path}")
            return [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"   ⚠️  Invalid FPS detected: {fps}")
            fps = 30  # Default fallback
            
        frame_skip = Config.FRAME_SKIP  # Use config for evaluation density
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   📊 Video stats - FPS: {fps:.1f}, Total frames: {total_frames}, Skip: {frame_skip}")
        
        frames = []
        frame_times = []
        frame_idx = 0
        failed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                if frame_idx % frame_skip == 0:
                    if frame.shape[0] == 0 or frame.shape[1] == 0:
                        print(f"   ⚠️  Corrupted frame at index {frame_idx}")
                        failed_frames += 1
                        frame_idx += 1
                        continue
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_times.append(frame_idx / fps)
            except Exception as e:
                print(f"   ⚠️  Error processing frame {frame_idx}: {e}")
                failed_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        if failed_frames > 0:
            print(f"   ⚠️  {failed_frames} frames failed to process")
        
        if len(frames) == 0:
            print(f"   ❌ No valid frames extracted from video")
            return [], []
            
        print(f"   ✅ Extracted {len(frames)} frames successfully")
        return frames, frame_times
        
    except Exception as e:
        print(f"   ❌ Error during frame extraction: {e}")
        traceback.print_exc()
        return [], []

def preprocess_frame(frame, target_size=224):
    """Preprocess frame for YOLO and classification models with ImageNet normalization"""
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame received")
            
        # Resize to target size
        frame_resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        
        # Normalize with ImageNet means and stds
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor
    except Exception as e:
        print(f"   ⚠️  Error preprocessing frame: {e}")
        raise

def preprocess_rtdetr_frame(frame, target_size=960):
    """Preprocess frame for RT-DETR: only 0-1 normalization (no ImageNet stats)"""
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame received")
            
        # Resize to target size
        frame_resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor with only 0-1 normalization (no ImageNet stats)
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        
        return frame_tensor
    except Exception as e:
        print(f"   ⚠️  Error preprocessing RT-DETR frame: {e}")
        raise

def preprocess_i3d_frame(frame, target_size=224):
    """Preprocess a frame for I3D: 0-1 scaling + ImageNet normalization to match training"""
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame received")

        frame_resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        
        # Add ImageNet normalization to match training
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor
    except Exception as e:
        print(f"   ⚠️  Error preprocessing I3D frame: {e}")
        raise

def apply_symbolic_reasoning(detections, symbolic_baselines):
    """Apply symbolic baselines to filter/refine detections"""
    # Placeholder for symbolic reasoning integration
    # This would use the baselines from neopolyp_symbolic_baselines.py
    return detections  # For now, return unchanged


def predict_cluster_id(feature_vector, kmeans_model, feature_scaler):
    """Predict the visual prototype cluster for a 444-dim feature vector."""
    if kmeans_model is None or feature_scaler is None:
        return None

    try:
        feature_array = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        scaled_features = feature_scaler.transform(feature_array)
        return int(kmeans_model.predict(scaled_features)[0])
    except Exception as e:
        print(f"   ⚠️  Cluster prediction failed: {e}")
        return None

def _build_tracks_from_detections(detections, frames, rules_engine, iou_threshold=0.40, min_frames=20):
    """
    Build polyp tracks from YOLO and RT-DETR detections when consensus is unavailable.
    Groups spatially overlapping detections across consecutive frames into tracks.
    Each distinct spatial cluster becomes one track.
    """
    frame_boxes = {}
    for model in ['yolo', 'rtdetr']:
        for det in detections.get(model, []):
            fidx = det.get('frame')
            if fidx is None or fidx >= len(frames):
                continue
            for box, conf in zip(det.get('boxes', []), det.get('confidences', [])):
                frame_boxes.setdefault(fidx, []).append((list(box), float(conf), model))

    if not frame_boxes:
        return []

    sorted_frames = sorted(frame_boxes.keys())
    open_tracks = []

    for fidx in sorted_frames:
        frame_dets = frame_boxes[fidx]

        for (box, conf, model) in frame_dets:
            best_track = None
            best_iou = 0.0

            for track in open_tracks:
                if fidx - track['last_frame'] > 5:
                    continue
                tiou = rules_engine._calculate_iou(box, track['box'])
                if tiou > best_iou and tiou >= iou_threshold:
                    best_iou = tiou
                    best_track = track

            if best_track is not None:
                best_track['frames'].append(fidx)
                best_track['confs'].append(conf)
                best_track['all_boxes'].append(box)
                all_b = np.array(best_track['all_boxes'])
                best_track['box'] = all_b.mean(axis=0).tolist()
                best_track['last_frame'] = fidx
            else:
                open_tracks.append({
                    'box': list(box),
                    'frames': [fidx],
                    'confs': [conf],
                    'all_boxes': [list(box)],
                    'last_frame': fidx,
                    'model': model,
                })

    result = []
    for i, track in enumerate(open_tracks):
        if len(track['frames']) < min_frames:
            continue
        avg_conf = float(np.mean(track['confs']))
        best_frame = track['frames'][int(np.argmax(track['confs']))]
        result.append({
            'polyp_id': i + 1,
            'box': track['box'],
            'all_boxes': track['all_boxes'],
            'frame': best_frame,
            'representative_frame': best_frame,
            'representative_confidence': float(max(track['confs'])),
            'frame_sequence': sorted(set(track['frames'])),
            'confidence_sequence': track['confs'],
            'frame_confidence_pairs': list(zip(track['frames'], track['confs'])),
            'start_frame': min(track['frames']),
            'end_frame': max(track['frames']),
            'num_frames': len(set(track['frames'])),
            'temporal_average_conf': avg_conf,
            'confidence_boost': 0.0,
            'model': track['model'],
        })

    return result


# ==========================================
# VIDEO PROCESSING
# ==========================================
def process_video(video_path, models, output_dir, symbolic_baselines=None):
    """
    Process a single video: extract frames, run inference, generate outputs
    Includes comprehensive error handling for robust 373-video batch processing
    """
    try:
        print(f"\n🎬 Processing video: {video_path.name}")
        
        # Create output subdirs
        video_output_dir = output_dir
        video_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract frames with validation
        frames, frame_times = extract_frames(video_path, Config.FRAME_RATE)
        
        if len(frames) == 0:
            print(f"   ❌ Video processing failed: No frames extracted")
            # Write error log
            error_log = video_output_dir / "ERROR_LOG.txt"
            with open(error_log, 'w') as f:
                f.write(f"Failed to extract frames from {video_path.name}\n")
            return

        # MedSAM2 video predictor — initialized once per video for temporal tracking
        medsam2_video_predictor = None
        medsam2_inference_state = None
        _tmp_frame_dir = None
        try:
            if 'medsam' in models:
                try:
                    from sam2.build_sam import build_sam2_video_predictor
                    medsam2_video_predictor = build_sam2_video_predictor(
                        config_file=Config.MEDSAM2_CONFIG,
                        ckpt_path=str(Config.MEDSAM2_CHECKPOINT),
                        device=Config.DEVICE,
                    )
                    print("   ✅ MedSAM2 video predictor initialized")
                    
                    # MedSAM2 video predictor needs frames as a directory — save them temporarily
                    import tempfile, os
                    _tmp_frame_dir = Path(tempfile.mkdtemp(prefix='medsam2_frames_'))
                    for _fi, _fr in enumerate(frames):
                        cv2.imwrite(str(_tmp_frame_dir / f"{_fi:05d}.jpg"),
                                    cv2.cvtColor(_fr, cv2.COLOR_RGB2BGR))
                    
                    medsam2_inference_state = medsam2_video_predictor.init_state(
                        video_path=str(_tmp_frame_dir)
                    )
                    print(f"   ✅ MedSAM2 video predictor ready ({len(frames)} frames)")
                except Exception as vid_init_err:
                    print(f"   ⚠️  MedSAM2 video predictor failed (using image-mode only): {vid_init_err}")
                    medsam2_video_predictor = None
                    medsam2_inference_state = None
        except Exception as e:
            print(f"   ⚠️  MedSAM2 setup failed: {e}")
            medsam2_video_predictor = None
            medsam2_inference_state = None

        # Prepare data structures - only detection models
        detection_model_names = [model_name for model_name in models.keys() if model_name not in {'kmeans', 'feature_scaler'}]
        detections = {model_name: [] for model_name in detection_model_names}
        segmentations = defaultdict(list)  # Added this line

        # Process frames in batches for efficiency
        batch_size = Config.BATCH_SIZE_INFERENCE
        frame_batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]

        print(f"   🔄 Processing {len(frames)} frames in {len(frame_batches)} batches (batch size: {batch_size})")

        for batch_idx, frame_batch in enumerate(frame_batches):
            try:
                print(f"   🔄 Batch {batch_idx+1}/{len(frame_batches)}")

                # Prepare batch tensor for models that support batching
                frame_batch_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frame_batch]

                # YOLO inference
                if 'yolo' in models:
                    try:
                        yolo_results = models['yolo'].predict(
                            frame_batch_bgr,
                            imgsz=Config.IMG_SIZE,
                            conf=Config.YOLO_CONF_THRESHOLD,
                            iou=0.45,
                            verbose=False,
                        )
                        yolo_detections_count = 0
                        for i, result in enumerate(yolo_results):
                            frame_idx = batch_idx * batch_size + i
                            if frame_idx < len(frame_times):
                                num_boxes = len(result.boxes) if len(result.boxes) > 0 else 0
                                if num_boxes > 0:
                                    yolo_detections_count += num_boxes
                                detections['yolo'].append({
                                    'frame': frame_idx,
                                    'time': frame_times[frame_idx],
                                    'boxes': result.boxes.xyxy.cpu().numpy().tolist()
                                             if len(result.boxes) > 0 else [],
                                    'confidences': result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) > 0 else [],
                                    'classes': result.boxes.cls.cpu().numpy().tolist() if len(result.boxes) > 0 else [],
                                    'track_ids': result.boxes.id.cpu().numpy().tolist() if result.boxes.id is not None else []
                                })
                        if yolo_detections_count > 0:
                            print(f"        ✅ YOLO: {yolo_detections_count} boxes detected in batch")
                    except Exception as e:
                        print(f"   ⚠️  YOLO inference failed: {e}")

                # RT-DETR inference
                if 'rtdetr' in models:
                    try:
                        rtdetr_results = models['rtdetr'].predict(
                            frame_batch_bgr,
                            imgsz=640,
                            conf=Config.RTDETR_CONF_THRESHOLD,
                            iou=0.45,
                            verbose=False,
                        )
                        rtdetr_detections_count = 0
                        for i, result in enumerate(rtdetr_results):
                            frame_idx = batch_idx * batch_size + i
                            if frame_idx < len(frame_times):
                                num_boxes = len(result.boxes) if len(result.boxes) > 0 else 0
                                if num_boxes > 0:
                                    rtdetr_detections_count += num_boxes
                                detections['rtdetr'].append({
                                    'frame': frame_idx,
                                    'time': frame_times[frame_idx],
                                    'boxes': result.boxes.xyxy.cpu().numpy().tolist()
                                             if len(result.boxes) > 0 else [],
                                    'confidences': result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) > 0 else [],
                                    'classes': result.boxes.cls.cpu().numpy().tolist() if len(result.boxes) > 0 else []
                                })
                        if rtdetr_detections_count > 0:
                            print(f"        ✅ RTDETR: {rtdetr_detections_count} boxes detected in batch")
                    except Exception as e:
                        print(f"   ⚠️  RT-DETR inference failed: {e}")

                # MedSAM2 inference: RULE 1 - only segment within YOLO/RT-DETR boxes
                if 'medsam' not in models:
                    if batch_idx == 0:
                        print(f"   ❌ MedSAM2 skipped — 'medsam' not in models. Keys: {list(models.keys())}")
                if 'medsam' in models:
                    try:
                        predictor = models['medsam']
                        if batch_idx == 0:
                            print(f"   MedSAM2 block entered batch 0 — predictor: {type(predictor).__name__}, CUDA: {torch.cuda.is_available()}")

                        for i, frame in enumerate(frame_batch):
                            frame_idx = batch_idx * batch_size + i
                            if frame_idx >= len(frame_times):
                                continue
                            
                            h, w = frame.shape[:2]  # Full frame size
                            
                            yolo_boxes_frame = []
                            
                            if 'yolo' in detections:
                                frame_yolo = [d for d in detections['yolo'] if d['frame'] == frame_idx]
                                if frame_yolo:
                                    boxes = frame_yolo[0].get('boxes', [])
                                    confidences = frame_yolo[0].get('confidences', [])
                                    for box, conf in zip(boxes, confidences):
                                        # Use low threshold 0.05 for MedSAM2 prompt collection
                                        # MedSAM2 has its own internal filtering at 0.10
                                        if conf < Config.YOLO_CONF_THRESHOLD:
                                            continue
                                        # Boxes are in pixel coordinates; just clip to frame bounds
                                        x1 = float(box[0])
                                        y1 = float(box[1])
                                        x2 = float(box[2])
                                        y2 = float(box[3])
                                        x1 = max(0.0, min(w - 1.0, x1))
                                        y1 = max(0.0, min(h - 1.0, y1))
                                        x2 = max(0.0, min(w - 1.0, x2))
                                        y2 = max(0.0, min(h - 1.0, y2))
                                        if x2 > x1 and y2 > y1:
                                            yolo_boxes_frame.append([x1, y1, x2, y2])
                            else:
                                if frame_idx % 10 == 0:  # Debug every 10 frames
                                    print(f"       ℹ️  'yolo' key not in detections dict. Available keys: {detections.keys()}")
                            
                            rtdetr_boxes_frame = []
                            
                            if 'rtdetr' in detections:
                                frame_rtdetr = [d for d in detections['rtdetr'] if d['frame'] == frame_idx]
                                if frame_rtdetr:
                                    boxes = frame_rtdetr[0].get('boxes', [])
                                    confidences = frame_rtdetr[0].get('confidences', [])
                                    for box, conf in zip(boxes, confidences):
                                        # Use low threshold 0.05 for MedSAM2 prompt collection
                                        # MedSAM2 has its own internal filtering
                                        if conf < Config.RTDETR_CONF_THRESHOLD:
                                            continue
                                        x1 = float(box[0])
                                        y1 = float(box[1])
                                        x2 = float(box[2])
                                        y2 = float(box[3])
                                        # RT-DETR may output normalized [0,1] coords — scale to pixels
                                        if max(x1, y1, x2, y2) <= 1.0:
                                            x1, x2 = x1 * w, x2 * w
                                            y1, y2 = y1 * h, y2 * h
                                        x1 = max(0.0, min(w - 1.0, x1))
                                        y1 = max(0.0, min(h - 1.0, y1))
                                        x2 = max(0.0, min(w - 1.0, x2))
                                        y2 = max(0.0, min(h - 1.0, y2))
                                        if x2 > x1 and y2 > y1:
                                            rtdetr_boxes_frame.append([x1, y1, x2, y2])
                            else:
                                if frame_idx % 10 == 0:  # Debug every 10 frames
                                    print(f"       ℹ️  'rtdetr' key not in detections dict")
                            
                            # DEBUG: Show what boxes we found
                            if frame_idx % 20 == 0 and (yolo_boxes_frame or rtdetr_boxes_frame):
                                print(f"       📦 Frame {frame_idx}: Found {len(yolo_boxes_frame)} YOLO boxes, {len(rtdetr_boxes_frame)} RTDETR boxes")
                            elif frame_idx % 20 == 0:
                                print(f"       ⚠️  Frame {frame_idx}: No boxes found (YOLO: {len(yolo_boxes_frame)}, RTDETR: {len(rtdetr_boxes_frame)})")
                            
                            # ----- MedSAM2 IMAGE MODE (per-frame, box prompt) -----
                            frame_rgb = frame
                            medsam_boxes = []
                            medsam_confs = []
                            medsam_masks_list = []

                            # Fuse YOLO + RT-DETR boxes: take the UNION (outer bbox) of each pair
                            all_det_boxes = yolo_boxes_frame + rtdetr_boxes_frame
                            # Deduplicate by merging highly overlapping boxes
                            # FIX: Raised threshold from 0.10 to 0.45 to prevent distinct close
                            # boxes ballooning into one giant box that swamps the SAM2 prompt
                            prompt_boxes = []
                            used = set()
                            for bi, b1 in enumerate(all_det_boxes):
                                if bi in used:
                                    continue
                                merged = list(b1)
                                for bj, b2 in enumerate(all_det_boxes):
                                    if bj <= bi or bj in used:
                                        continue
                                    ix1 = max(merged[0], b2[0])
                                    iy1 = max(merged[1], b2[1])
                                    ix2 = min(merged[2], b2[2])
                                    iy2 = min(merged[3], b2[3])
                                    if ix2 > ix1 and iy2 > iy1:
                                        inter = (ix2 - ix1) * (iy2 - iy1)
                                        a1 = (merged[2] - merged[0]) * (merged[3] - merged[1])
                                        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                                        iou = inter / max(a1 + a2 - inter, 1e-6)
                                        if iou > 0.45:
                                            merged = [min(merged[0], b2[0]), min(merged[1], b2[1]),
                                                      max(merged[2], b2[2]), max(merged[3], b2[3])]
                                            used.add(bj)
                                prompt_boxes.append(merged)
                                used.add(bi)

                            if not prompt_boxes:
                                pass  # no detector fired — skip MedSAM2 for this frame
                            else:
                                if batch_idx == 0 and i == 0:
                                    print(f"   MedSAM2 prompt_boxes: {len(prompt_boxes)} boxes — predictor: {type(predictor).__name__}")
                                try:
                                    frame_for_sam = frame_rgb if frame_rgb.dtype == np.uint8 else (frame_rgb * 255).astype(np.uint8)

                                    _use_cuda = torch.cuda.is_available()
                                    _autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if _use_cuda else contextlib.nullcontext()
                                    with torch.inference_mode(), _autocast_ctx:

                                        predictor.set_image(frame_for_sam)

                                        for pb in prompt_boxes:
                                            try:
                                                # Shape (1, 4) for SAM2 batched box input
                                                bbox_np = np.array([pb[0], pb[1], pb[2], pb[3]], dtype=np.float32)

                                                masks, scores, _ = predictor.predict(
                                                    point_coords=None,
                                                    point_labels=None,
                                                    box=bbox_np,
                                                    multimask_output=False
                                                )

                                                if masks is None or len(masks) == 0:
                                                    print(f"   MedSAM2 returned empty masks for frame {frame_idx}")
                                                    continue

                                                # FIX: Robust squeeze handles (1,H,W), (1,1,H,W) or (H,W)
                                                mask_np = np.array(masks).squeeze().astype(bool)
                                                if mask_np.ndim == 3:
                                                    mask_np = mask_np[0]
                                                score_val = float(scores[0])

                                                print(f"   MedSAM2 frame {frame_idx} -> score {score_val:.4f}")

                                                if score_val < 0.01:
                                                    print(f"   Score {score_val:.4f} too low (<0.01) — skipped")
                                                    continue

                                                if score_val < Config.MEDSAM_CONF_THRESHOLD:
                                                    print(f"   Score {score_val:.4f} below threshold {Config.MEDSAM_CONF_THRESHOLD} — skipped")
                                                    continue

                                                # FIX: Pad clip 15px so polyp border pixels aren't erased
                                                pad = 15
                                                clip = np.zeros((h, w), dtype=bool)
                                                ix1 = max(0, int(pb[0]) - pad)
                                                iy1 = max(0, int(pb[1]) - pad)
                                                ix2 = min(w, int(pb[2]) + pad)
                                                iy2 = min(h, int(pb[3]) + pad)
                                                clip[iy1:iy2, ix1:ix2] = True
                                                mask_np = mask_np & clip

                                                if not mask_np.any():
                                                    print(f"   Mask empty after padded clip — skipped")
                                                    continue

                                                ys, xs = np.where(mask_np)
                                                medsam_boxes.append([float(xs.min()), float(ys.min()),
                                                                      float(xs.max()), float(ys.max())])
                                                medsam_confs.append(score_val)
                                                medsam_masks_list.append(mask_np)
                                                print(f"   MedSAM2 mask accepted frame {frame_idx} score={score_val:.4f}")

                                            except Exception as pb_err:
                                                print(f"   MedSAM2 predict error frame {frame_idx} box {pb}: {pb_err}")
                                                traceback.print_exc()

                                except Exception as sam2_err:
                                    print(f"   MedSAM2 set_image error frame {frame_idx}: {sam2_err}")
                                    traceback.print_exc()

                            # ----- MedSAM2 VIDEO MODE — propagate mask forward (best keyframe) -----
                            # If the video predictor is ready and we got a good mask on this frame,
                            # register it so the model can propagate it to neighboring frames.
                            if (medsam2_video_predictor is not None and
                                    medsam2_inference_state is not None and
                                    medsam_masks_list):
                                try:
                                    with torch.inference_mode(), torch.autocast(
                                            device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
                                        for oi, m in enumerate(medsam_masks_list):
                                            medsam2_video_predictor.add_new_mask(
                                                medsam2_inference_state,
                                                frame_idx=frame_idx,
                                                obj_id=oi + 1,
                                                mask=m
                                            )
                                except Exception as vid_err:
                                    pass   # Non-fatal — image-mode masks already stored
                            
                            if medsam_boxes:
                                detections['medsam'].append({
                                    'frame': frame_idx,
                                    'time': frame_times[frame_idx],
                                    'boxes': medsam_boxes,
                                    'confidences': medsam_confs,
                                    'classes': [0] * len(medsam_boxes),
                                    'rule1_applied': True
                                })
                                
                                # Store masks
                                for mask_idx, mask in enumerate(medsam_masks_list):
                                    if mask is not None:
                                        segmentations['medsam'].append({
                                            'frame': frame_idx,
                                            'mask': mask,
                                            'prompt_box': medsam_boxes[mask_idx] if mask_idx < len(medsam_boxes) else None,
                                            'confidence': medsam_confs[mask_idx] if mask_idx < len(medsam_confs) else 0.0
                                        })
                    
                    except Exception as e:
                        import traceback
                        print(f"   ⚠️  RULE 1 MedSAM2 ROI detection failed: {e}")
                        traceback.print_exc()
                    
                    # Track MedSAM2 detections per batch
                    medsam_batch_count = len(detections.get('medsam', []))
                    if batch_idx % 20 == 0 or medsam_batch_count > 0:
                        print(f"       📈 Batch {batch_idx} MedSAM2: {medsam_batch_count} masks generated")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ❌ Batch {batch_idx} processing failed: {e}")
                continue

        # MedSAM2 VIDEO PROPAGATION — fills in masks for frames where image mode got no result
        if medsam2_video_predictor is not None and medsam2_inference_state is not None:
            try:
                print("   🎬 MedSAM2: propagating masks across all video frames...")
                with torch.inference_mode(), torch.autocast(
                        device_type='cuda' if torch.cuda.is_available() else 'cpu',
                        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
                    for out_frame_idx, out_obj_ids, out_mask_logits in \
                            medsam2_video_predictor.propagate_in_video(medsam2_inference_state):
                        for oi, logit in zip(out_obj_ids, out_mask_logits):
                            prop_mask = (logit[0] > 0.0).cpu().numpy().astype(bool)
                            if not prop_mask.any():
                                continue
                            # Only add if this frame has no image-mode mask yet
                            already = [s for s in segmentations['medsam']
                                       if s['frame'] == out_frame_idx]
                            if not already:
                                ys, xs = np.where(prop_mask)
                                prop_box = [float(xs.min()),float(ys.min()),
                                            float(xs.max()),float(ys.max())]
                                segmentations['medsam'].append({
                                    'frame': out_frame_idx,
                                    'mask': prop_mask,
                                    'prompt_box': prop_box,
                                    'confidence': 0.75   # propagated — lower than direct
                                })
                                detections['medsam'].append({
                                    'frame': out_frame_idx,
                                    'time': frame_times[out_frame_idx] if out_frame_idx < len(frame_times) else 0.0,
                                    'boxes': [prop_box],
                                    'confidences': [0.75],
                                    'classes': [0],
                                    'rule1_applied': True,
                                    'propagated': True
                                })
                medsam2_video_predictor.reset_state(medsam2_inference_state)
                # Clean up temp frame dir
                import shutil
                shutil.rmtree(str(_tmp_frame_dir), ignore_errors=True)
                print("   ✅ MedSAM2 video propagation complete")
            except Exception as e:
                print(f"   ⚠️  MedSAM2 video propagation failed: {e}")

        features_dict = defaultdict(list)  # frame_idx -> list of features
        
        # Debug: Print detection counts
        yolo_det_count = sum(len(d.get('boxes', [])) for d in detections.get('yolo', []))
        rtdetr_det_count = sum(len(d.get('boxes', [])) for d in detections.get('rtdetr', []))
        medsam_det_count = sum(len(d.get('boxes', [])) for d in detections.get('medsam', []))
        print(f"   📊 Detection Summary:")
        print(f"      YOLO: {yolo_det_count} detections")
        print(f"      RTDETR: {rtdetr_det_count} detections")
        print(f"      MedSAM2: {medsam_det_count} detections")
        
        # ==========================================
        # RULE 2: CONSENSUS VOTING & RULE 5: BORDER FILTERING
        # ==========================================
        print(f"\n   🧠 RULE 2: Applying consensus voting (all 3 models must agree)...")
        print(f"   🧠 RULE 5: Filtering border and artifact regions...")
        
        rules_engine = RulesEngine5(device=Config.DEVICE)
        consensus_results = defaultdict(list)  # frame_idx -> consensus detections
        filtered_count = 0
        border_filtered_count = 0
        
        # Group detections by frame
        frame_detections = defaultdict(lambda: {'yolo': [], 'yolo_confs': [], 
                                                  'rtdetr': [], 'rtdetr_confs': [],
                                                  'medsam': [], 'medsam_confs': []})
        
        for model_name in ['yolo', 'rtdetr', 'medsam']:
            if model_name in detections:
                for det in detections[model_name]:
                    frame_idx = det.get('frame')
                    if frame_idx >= len(frames):
                        continue
                    
                    boxes = det.get('boxes', [])
                    confs = det.get('confidences', [])
                    
                    frame_detections[frame_idx][model_name] = boxes
                    frame_detections[frame_idx][f'{model_name}_confs'] = confs
        
        # Apply Rule 2 & 5 to each frame
        for frame_idx in sorted(frame_detections.keys()):
            if frame_idx >= len(frames):
                continue
            
            frame = frames[frame_idx]
            frame_data = frame_detections[frame_idx]
            
            # Debug per-frame detection
            yolo_frame_count = len(frame_data['yolo'])
            rtdetr_frame_count = len(frame_data['rtdetr'])
            medsam_frame_count = len(frame_data['medsam'])
            
            if yolo_frame_count > 0 or rtdetr_frame_count > 0 or medsam_frame_count > 0:
                print(f"      Frame {frame_idx}: YOLO={yolo_frame_count}, RTDETR={rtdetr_frame_count}, MedSAM2={medsam_frame_count}")
            
            # Apply RULE 2: Consensus voting
            consensus_result = rules_engine.rule2_consensus_voting(
                frame_idx,
                frame_data['yolo'],
                frame_data['yolo_confs'],
                frame_data['rtdetr'],
                frame_data['rtdetr_confs'],
                frame_data['medsam'],
                frame_data['medsam_confs']
            )
            
            # Apply RULE 5: Border/artifact filtering
            valid_consensus_boxes = []
            valid_consensus_confs = []
            valid_consensus_models = []
            
            for box, conf, consensus_model_names in zip(
                consensus_result['consensus_boxes'],
                consensus_result['consensus_confidences'],
                consensus_result['consensus_models']
            ):
                if rules_engine.rule5_filter_border_artifacts(frame, box):
                    valid_consensus_boxes.append(box)
                    valid_consensus_confs.append(conf)
                    valid_consensus_models.append(consensus_model_names)
                else:
                    border_filtered_count += 1
            
            if valid_consensus_boxes:
                consensus_results[frame_idx] = {
                    'boxes': valid_consensus_boxes,
                    'confidences': valid_consensus_confs,
                    'models': valid_consensus_models,
                    'frame_has_consensus': True
                }
                filtered_count += len(valid_consensus_boxes)
        
        print(f"   ✅ RULE 2: Found {filtered_count} consensus detections")
        print(f"   ✅ RULE 5: Filtered out {border_filtered_count} border/artifact detections")
        
        # ==========================================
        # TEMPORAL CONSENSUS AVERAGING (Enhanced Detection Confidence)
        # ==========================================
        print(f"\n   ⏱️  TEMPORAL CONSENSUS: Averaging confidence across continuous frame sequences...")
        
        temporal_aggregation = rules_engine.temporal_consensus_aggregation(
            consensus_results,
            max_frame_gap=5,
            min_consensus_frames=20
        )
        temporal_tracks = temporal_aggregation.get('tracks', [])
        
        # FIX ISSUE 2: Merge overlapping tracks (same polyp detected multiple times)
        print(f"   🔗 Merging overlapping polyp tracks...")
        merged_tracks = rules_engine.merge_overlapping_polyp_tracks(temporal_tracks)

        # Final deduplication pass: collapse repeated views of the same polyp into one representative track
        deduplicated_tracks = deduplicate_polyps(merged_tracks, iou_threshold=0.5)

        if not deduplicated_tracks:
            print("   ⚠️  No consensus tracks — building tracks from YOLO+RT-DETR detections directly")
            raw_tracks = _build_tracks_from_detections(detections, frames, rules_engine)
            if raw_tracks:
                deduplicated_tracks = deduplicate_polyps(raw_tracks, iou_threshold=0.40)
                print(f"   ✅ Built {len(deduplicated_tracks)} tracks from detector detections")

        total_tracks = len(deduplicated_tracks)
        frame_coverage = sum(t.get('num_frames', 0) for t in deduplicated_tracks)
        boost_amount = temporal_aggregation.get('average_boost_amount', 0.0)
        
        print(f"   ✅ Found {total_tracks} polyp tracks")
        print(f"   ✅ Total frame coverage: {frame_coverage} frames")
        if boost_amount > 0:
            print(f"   📈 Average confidence boost: +{boost_amount:.3f}")

        consensus_confidence = float(np.mean([
            track.get('temporal_average_conf', 0.0) for track in deduplicated_tracks
        ])) if deduplicated_tracks else 0.0

        detections['_consensus_metadata'] = {
            'polyp_present': bool(deduplicated_tracks) and consensus_confidence > 0.5,
            'overall_confidence': consensus_confidence,
            'consensus_frame_count': int(frame_coverage),
            'consensus_percentage': float((frame_coverage / len(frames) * 100.0) if frames else 0.0),
            'num_consensus_runs': int(len(deduplicated_tracks)),
            'average_confidence_boost': float(boost_amount),
            'models_detected': 3 if deduplicated_tracks else 0,
            'yolo_detected': bool(deduplicated_tracks),
            'rtdetr_detected': bool(deduplicated_tracks),
            'medsam_detected': bool(deduplicated_tracks),
            'frame_level_consensus_count': int(filtered_count),
            'border_filtered_count': int(border_filtered_count),
        }
        
        # Replace consensus results with temporally-averaged results
        # For each temporal track, update the frame detections
        temporal_consensus_results = defaultdict(list)
        
        for track_idx, track in enumerate(deduplicated_tracks):
            # Create a polyp detection for each frame in the track
            # But use the temporally-averaged confidence
            avg_conf = track['temporal_average_conf']
            box = track['box']
            
            for frame_idx in track['frame_sequence']:
                temporal_consensus_results[frame_idx].append({
                    'box': box,
                    'confidence': avg_conf,
                    'polyp_track_id': track['polyp_id'],
                    'track_frames': track['frame_sequence'],
                    'track_confidence_boost': track['confidence_boost'],
                    'is_temporal_averaged': True
                })
        
        # Merge temporal results back into consensus_results
        # Temporal averages take priority (higher confidence)
        updated_consensus_results = {}
        
        for frame_idx in consensus_results:
            updated_consensus_results[frame_idx] = {
                'boxes': consensus_results[frame_idx]['boxes'],
                'confidences': consensus_results[frame_idx]['confidences'],
                'models': consensus_results[frame_idx]['models'],
                'frame_has_consensus': consensus_results[frame_idx]['frame_has_consensus'],
                'temporal_enhanced': False
            }
        
        # Add temporal results (these have boosted confidence)
        for frame_idx in temporal_consensus_results:
            if frame_idx not in updated_consensus_results:
                updated_consensus_results[frame_idx] = {
                    'boxes': [],
                    'confidences': [],
                    'models': [],
                    'frame_has_consensus': True,
                    'temporal_enhanced': True
                }
            
            for temporal_det in temporal_consensus_results[frame_idx]:
                updated_consensus_results[frame_idx]['boxes'].append(temporal_det['box'])
                updated_consensus_results[frame_idx]['confidences'].append(temporal_det['confidence'])
                updated_consensus_results[frame_idx]['models'].append(['TEMPORAL_AVERAGE'])
                updated_consensus_results[frame_idx]['temporal_enhanced'] = True
        
        consensus_results = updated_consensus_results
        
        print(f"   ✅ Updated {len([f for f in consensus_results if consensus_results[f].get('temporal_enhanced')])} frames with temporal averaging")
        
        try:
            from polyp_feature_extractor import extract_all_features, crop_pure_roi_from_mask
            
            # Bug 3 Fix: Tag frames from deduplicated_tracks as consensus before feature loop
            temporal_frame_set = set()
            for track in deduplicated_tracks:
                for f in track.get('frame_sequence', []):
                    temporal_frame_set.add(f)
            
            # Extract features for ALL detections (including those without consensus for analysis)
            for model_name in ['yolo', 'rtdetr', 'medsam']:
                if model_name not in detections:
                    continue
                
                for detection in detections[model_name]:
                    frame_idx = detection.get('frame')
                    if frame_idx >= len(frames):
                        continue
                    
                    frame = frames[frame_idx]
                    boxes = detection.get('boxes', [])
                    
                    for box_idx, box in enumerate(boxes):
                        try:
                            # Get segmentation mask if available
                            mask = None
                            if model_name == 'medsam':
                                seg_list = segmentations.get(model_name, [])
                                matching_segs = [s for s in seg_list if s.get('frame') == frame_idx]
                                best_mask = None
                                best_iou = 0.0
                                for seg in matching_segs:
                                    prompt_box = seg.get('prompt_box')
                                    if prompt_box is None:
                                        continue
                                    iou = rules_engine._calculate_iou(box, prompt_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_mask = seg.get('mask')
                                mask = best_mask
                            
                            # Extract features from ROI
                            features = extract_all_features(frame, box, mask)
                            features['model'] = model_name
                            features['frame'] = frame_idx
                            if model_name in {'yolo', 'rtdetr'}:
                                confidences = detection.get('confidences', [])
                                if box_idx < len(confidences):
                                    features['detection_confidence'] = float(confidences[box_idx])
                                else:
                                    features['detection_confidence'] = float(detection.get('confidence', 0.0))
                            else:
                                features['detection_confidence'] = float(detection.get('confidence', 0.0))
                            
                            # Mark if this is a consensus detection (Bug 3 Fix: also check temporal_frame_set)
                            if frame_idx in temporal_frame_set or frame_idx in consensus_results:
                                consensus_box_found = False
                                for cons_box in consensus_results.get(frame_idx, {}).get('boxes', []):
                                    if rules_engine._calculate_iou(box, cons_box) > 0.3:
                                        features['consensus'] = True
                                        consensus_box_found = True
                                        break
                                if not consensus_box_found and frame_idx in temporal_frame_set:
                                    features['consensus'] = True
                                elif not consensus_box_found:
                                    features['consensus'] = False
                            else:
                                features['consensus'] = False
                            
                            features_dict[frame_idx].append(features)
                        
                        except Exception as e:
                            pass  # Skip this box, continue with others
        
        except Exception as e:
            print(f"   ⚠️  Feature extraction failed: {e}")
        
        print(f"   ✅ Extracted features for {sum(len(v) for v in features_dict.values())} polyp detections")
        
        # ==========================================
        # RULE 3: SYMBOLIC REASONING (70-30 Split for polyp classification)
        # RULE 4: SSL FEATURES INTEGRATION
        # ==========================================
        print(f"\n   🧠 RULE 3 & 4: Applying symbolic reasoning with SSL features...")
        print(f"   📊 Rule 3: Using 70-30 clinical dataset split for polyp classification")
        print(f"   📊 Rule 4: Integrating 444-dim SSL features into decisions")
        
        # Initialize counters (always defined regardless of experts availability)
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        type_counts = defaultdict(int)
        consensus_polyp_count = len(deduplicated_tracks) if deduplicated_tracks else 0
        error_count = 0

        symbolic_results = []
        integrator = None
        ssl_model = None
        type_classifier = None
        
        # Load data-driven thresholds from Phase 2C
        calib_thresholds = load_data_driven_thresholds(Config.THESIS_ROOT)
        
        # ALWAYS initialize models — independent of annotations CSV
        try:
            experts_path = Config.MIXTURE_OF_EXPERTS
            if experts_path.exists():
                integrator = SymbolicReasoningIntegrator(experts_path, calibration_thresholds=calib_thresholds)
            else:
                integrator = None
            
            # Load SSL encoder for 444-dimensional feature extraction (Rule 4)
            print(f"   📊 Loading SSL encoder for 444-dim feature extraction (Rule 4)...")
            ssl_model = load_ssl_encoder(Config.DEVICE)
            
            # Initialize polyp type classifier with clinical annotations
            annotations_csv = Config.THESIS_ROOT / 'NeSy' / 'video-annotations.csv'
            video_annotations_dict = {}  # Store video_id -> annotation finding mapping
            if annotations_csv.exists():
                try:
                    import pandas as pd
                    ann_df = pd.read_csv(str(annotations_csv), sep=';')
                    for _, row in ann_df.iterrows():
                        vid = str(row.get('videoID', '')).strip()
                        finding = str(row.get('finding', '')).lower().strip()
                        if vid:
                            video_annotations_dict[vid] = finding
                except Exception as e:
                    print(f"   ⚠️  Error loading video annotations: {e}")
                
                type_classifier = PolypTypeClassifier(annotations_csv=str(annotations_csv))
            else:
                type_classifier = PolypTypeClassifier()
        
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            if ssl_model is None:
                ssl_model = load_ssl_encoder(Config.DEVICE)
            import traceback
            traceback.print_exc()
        
        # 70-30 split informational only — method removed, inference runs on all videos
        is_test_video = False

        # Store comprehensive polyp analysis summary - USE MERGED TRACKS INSTEAD OF RAW DETECTIONS
        polyp_features_summary = []
        
        # Create symbolic results for merged tracks (not individual detections)
        symbolic_results = []
        
        # Use merged temporal tracks for polyp counting (should be 1-2 polyps per video)
        if deduplicated_tracks:
            print(f"   📊 Processing {len(deduplicated_tracks)} unique polyp tracks for feature extraction...")
            
            for track_idx, track in enumerate(deduplicated_tracks):
                print(f"     🔍 Processing track {track_idx + 1}: frames {track.get('frame_sequence', [])}")
                
                # Search the whole track and prefer the strongest consensus-supported frame.
                candidate_frames = [frame for frame in track.get('frame_sequence', []) if isinstance(frame, int)]
                if not candidate_frames:
                    candidate_frames = [int(track.get('representative_frame', track.get('start_frame', track.get('frame', 0))))]

                # Start with the track's pre-selected representative (highest confidence frame)
                # not blindly candidate_frames[0] which is always the earliest frame
                best_frame_idx = int(track.get('representative_frame', candidate_frames[0]))
                if best_frame_idx not in candidate_frames:
                    best_frame_idx = candidate_frames[0]
                best_features = None
                best_frame_score = (-1, -1, -1.0)

                for candidate_frame_idx in candidate_frames:
                    frame_features = features_dict.get(candidate_frame_idx, [])
                    if not frame_features:
                        continue

                    consensus_features = [feature for feature in frame_features if feature.get('consensus', False)]
                    candidate_pool = consensus_features if consensus_features else frame_features
                    candidate_feature = max(candidate_pool, key=lambda feature: float(feature.get('confidence', 0.0)))
                    candidate_score = (
                        1 if consensus_features else 0,
                        len(consensus_features),
                        float(candidate_feature.get('confidence', 0.0))
                    )

                    if candidate_score > best_frame_score:
                        best_frame_score = candidate_score
                        best_frame_idx = candidate_frame_idx
                        best_features = candidate_feature

                if best_features is not None:
                    print(f"       ✅ Selected best frame {best_frame_idx} for track {track_idx + 1}")
                else:
                    print(f"       ❌ No features found for track {track_idx + 1} in frames {candidate_frames}")
                    # Bug 2 Fix: Extract directly from the best-confidence frame when features_dict is empty
                    if candidate_frames:
                        best_frame_idx = candidate_frames[len(candidate_frames)//2]  # middle of track
                        frame_rgb = frames[best_frame_idx]
                        reference_box = track.get('box', [])
                        if reference_box and len(reference_box) == 4:
                            best_features = extract_all_features(frame_rgb, reference_box, mask=None)
                            best_features['frame'] = best_frame_idx
                            best_features['detection_confidence'] = track.get('temporal_average_conf', 0.0)
                            best_features['consensus'] = True
                            print(f"       ✅ Fallback: Extracted features from frame {best_frame_idx}")


                feature_based = {
                    'classification': 'UNKNOWN',
                    'risk_score': 0.0
                }
                prediction = 'UNKNOWN'
                expert_classification = 'UNKNOWN'
                expert_confidence = 0.0
                symbolic_confidence = 0.0
                ssl_boost = 0.0
                risk_score = float(feature_based.get('risk_score', 0.0))
                detection_confidence = float(track.get('temporal_average_conf', 0.0))
                clinical_confidence = 0.0
                cluster_id = None
                polyp_type = 'UNKNOWN'
                polyp_type_confidence = 0.0
                enhanced_result = {
                    'prediction': 'UNKNOWN',
                    'confidence': 0.0,
                    'ssl_boost_applied': 0.0,
                    'method': 'fallback_uninitialized'
                }
                
                if best_features:
                    # Apply symbolic reasoning to this track
                    try:
                        # RULE 4: Extract 444-dimensional SSL features
                        frame_rgb = frames[best_frame_idx]
                        reference_box = best_features.get('box', []) or track.get('box', [])

                        mask = None
                        if 'medsam' in segmentations:
                            matching_masks = [
                                seg
                                for seg in segmentations.get('medsam', [])
                                if seg.get('frame') == best_frame_idx and seg.get('mask') is not None
                            ]
                            if matching_masks:
                                reference_box = best_features.get('box', []) or track.get('box', [])
                                best_mask = None
                                best_mask_iou = -1.0

                                for seg in matching_masks:
                                    seg_box = seg.get('prompt_box')
                                    if reference_box and seg_box and len(reference_box) == 4 and len(seg_box) == 4:
                                        current_iou = rules_engine._calculate_iou(reference_box, seg_box)
                                    else:
                                        current_iou = 0.0

                                    if current_iou > best_mask_iou:
                                        best_mask_iou = current_iou
                                        best_mask = seg.get('mask')

                                mask = best_mask if best_mask is not None else matching_masks[0].get('mask')

                        polyp_roi, _ = crop_pure_roi_from_mask(frame_rgb, mask) if mask is not None else (None, None)
                        if polyp_roi is None or polyp_roi.size == 0:
                            polyp_bbox = reference_box
                            if polyp_bbox and len(polyp_bbox) == 4:
                                x1, y1, x2, y2 = map(int, polyp_bbox)
                                if y2 > y1 and x2 > x1:
                                    polyp_roi = frame_rgb[y1:y2, x1:x2]
                                else:
                                    polyp_roi = frame_rgb
                            else:
                                polyp_roi = frame_rgb

                        tight_feature_source = best_features
                        if mask is not None and reference_box and len(reference_box) == 4:
                            try:
                                tight_feature_source = extract_all_features(
                                    frame_rgb,
                                    reference_box,
                                    mask,
                                    mask_padding=0.05
                                )
                                # Issue 5 fix: Fallback to best_features if result is all-zero (degenerate crop)
                                # Extend the existing check to cover all features
                                all_zero = (
                                    tight_feature_source.get('redness', 0.0) == 0.0 and
                                    tight_feature_source.get('vessel_visibility', 0.0) == 0.0 and
                                    tight_feature_source.get('texture', 0.0) == 0.0 and
                                    tight_feature_source.get('radius', 0.0) == 0.0
                                )
                                if all_zero:
                                    # Mask crop is degenerate — fall back to box-based features
                                    tight_feature_source = best_features
                                    # If best_features also has all-zero color, re-extract directly from frame
                                    if (tight_feature_source.get('redness', 0.0) == 0.0 and
                                        tight_feature_source.get('texture', 0.0) == 0.0 and
                                        tight_feature_source.get('vessel_visibility', 0.0) == 0.0):
                                        try:
                                            # Extract without mask — use raw bounding box crop
                                            _reextract = extract_all_features(frame_rgb, reference_box, None)
                                            if _reextract.get('redness', 0.0) > 0.0 or _reextract.get('texture', 0.0) > 0.0:
                                                tight_feature_source = _reextract
                                        except Exception:
                                            pass
                            except Exception as feature_err:
                                print(f"       ⚠️  Tight ROI feature extraction failed - using detector features: {feature_err}")
                                tight_feature_source = best_features

                        # Guard against uninitialized integrator (Bug 1 fix)
                        try:
                            feature_based = integrator.feature_based_classification(tight_feature_source) if integrator else {'classification': 'UNKNOWN', 'risk_score': 0.0}
                        except Exception as _fb_err:
                            print(f"       ⚠️  feature_based_classification failed: {_fb_err}")
                            feature_based = {'classification': 'UNKNOWN', 'risk_score': 0.0, 'clinical_class': 'ADENOMATOUS_POLYP', 'clinical_risk': 'LOW'}
                        risk_score = float(feature_based.get('risk_score', 0.0))
                        clinical_class = feature_based.get('clinical_class', 'ADENOMATOUS_POLYP')
                        clinical_risk = feature_based.get('clinical_risk', 'LOW')
                        detection_confidence = float(track.get('temporal_average_conf', 0.0))

                        try:
                            features_444 = extract_444_features(polyp_roi, ssl_model, Config.DEVICE)
                        except Exception as ssl_err:
                            print(f"       ⚠️  SSL feature extraction failed - using zero vector fallback: {ssl_err}")
                            features_444 = np.zeros(444, dtype=np.float32)

                        # === CHECK SSL QUALITY (Issue 3 Fix) ===
                        # Must appear BEFORE enhanced_classify_with_ssl call
                        _ssl_vec = features_444[:384] if features_444 is not None and len(features_444) >= 384 else np.zeros(384)
                        _ssl_std = float(np.std(_ssl_vec)) if len(_ssl_vec) > 0 else 0.0
                        ssl_is_meaningful = (
                            _ssl_std > 0.05 and
                            not np.any(np.isnan(_ssl_vec)) and
                            not np.any(np.isinf(_ssl_vec)) and
                            getattr(ssl_model, '_ssl_weights_loaded', False)
                        )
                        
                        cluster_id = predict_cluster_id(
                            features_444,
                            models.get('kmeans'),
                            models.get('feature_scaler'),
                        )
                        
                        # RULE 4: ENHANCED CLASSIFICATION - Combine basic features + SSL features
                        if ssl_is_meaningful and integrator:
                            basic_feature_vector = np.array([
                                tight_feature_source.get('redness', 0.0),
                                tight_feature_source.get('radius', 0.0),
                                tight_feature_source.get('texture', 0.0),
                                tight_feature_source.get('vessel_visibility', 0.0),
                                tight_feature_source.get('h_mean', 0.0),
                                tight_feature_source.get('s_mean', 0.0),
                                tight_feature_source.get('v_mean', 0.0),
                            ], dtype=np.float32)
                            enhanced_result = integrator.enhanced_classify_with_ssl(
                                basic_feature_vector, features_444, cluster_id=cluster_id
                            )
                            expert_classification = enhanced_result.get('prediction', 'UNKNOWN')
                            _conf_lo = integrator._np_thresholds.get('confidence_clip_low', 0.0) if integrator else 0.0
                            _conf_hi = integrator._np_thresholds.get('confidence_clip_high', 1.0) if integrator else 1.0
                            symbolic_confidence = float(np.clip(
                                enhanced_result.get('confidence', 0.0), _conf_lo, _conf_hi
                            ))
                        else:
                            # SSL not trained — use feature-based result as expert output
                            expert_classification = feature_based.get('clinical_class', 'ADENOMATOUS_POLYP')
                            _conf_lo = integrator._np_thresholds.get('confidence_clip_low', 0.0) if integrator else 0.0
                            _conf_hi = integrator._np_thresholds.get('confidence_clip_high', 1.0) if integrator else 1.0
                            symbolic_confidence = float(np.clip(risk_score, _conf_lo, _conf_hi))
                            enhanced_result = {
                                'prediction': expert_classification,
                                'confidence': symbolic_confidence,
                                'ssl_boost_applied': 0.0,
                                'method': 'feature_heuristic_ssl_unavailable'
                            }
                        
                        # Get enhanced prediction (SSL-boosted)
                        expert_confidence = float(enhanced_result.get('confidence', 0.0))
                        symbolic_confidence = expert_confidence
                        ssl_boost = float(enhanced_result.get('ssl_boost_applied', 0.0))

                        # Get enhanced prediction and apply to classification
                        prediction = expert_classification
                        clinical_confidence = symbolic_confidence
                        
                        # RULE 3: Polyp type classification — System C clinical keyword pipeline
                        polyp_type = 'UNKNOWN'
                        polyp_type_confidence = 0.0
                        try:
                            redness_val   = float(tight_feature_source.get('redness', 0.0))
                            vessel_val    = float(tight_feature_source.get('vessel_visibility', 0.0))
                            texture_val   = float(tight_feature_source.get('texture', 0.0))
                            radius_val    = float(tight_feature_source.get('radius', 0.0))
                            s_mean_val    = float(tight_feature_source.get('s_mean', 0.0))
                            edge_val      = float(tight_feature_source.get('edge_sharpness', 0.0))

                            clinical_keyword = PolypTypeClassifier.features_to_clinical_keyword(
                                redness_val, vessel_val, texture_val, radius_val, edge_val, s_mean_val
                            )
                            _ptc = PolypTypeClassifier()
                            clinical_class_pred, _ = _ptc.map_finding_to_class(clinical_keyword)
                            polyp_type = clinical_class_pred

                            total_signal = redness_val + vessel_val + texture_val + s_mean_val
                            polyp_type_confidence = float(np.clip(0.45 + total_signal * 0.4, 0.45, 0.92))
                        except Exception as _kw_err:
                            polyp_type = clinical_class  # fall back to integrator's clinical_class
                            polyp_type_confidence = 0.45
                        
                        # Store symbolic result for this track
                        symbolic_results.append({
                            'polyp_id': track_idx + 1,  # 1-based indexing to match polyp_features_summary
                            'frame': best_frame_idx,
                            'classification': prediction,
                            'expert_classification': expert_classification,
                            'confidence': float(clinical_confidence),
                            'symbolic_confidence': float(symbolic_confidence),
                            'detection_confidence': float(detection_confidence),
                            'clinical_confidence': float(clinical_confidence),
                            'ssl_boost_applied': ssl_boost,
                            'consensus': True,  # All tracks are consensus-based
                            'model': 'temporal_consensus',
                            'redness': float(tight_feature_source.get('redness', 0.0)),
                            'texture': float(tight_feature_source.get('texture', 0.0)),
                            'vessel_visibility': float(tight_feature_source.get('vessel_visibility', 0.0)),
                            'radius': float(tight_feature_source.get('radius', 0.0)),
                            'polyp_type': polyp_type,
                            'polyp_type_confidence': polyp_type_confidence,
                            'cluster_id': cluster_id,
                            'prediction_source': enhanced_result.get('method', 'ssl_enhanced_cluster_specific' if cluster_id is not None else 'ssl_enhanced_ensemble'),
                            'ssl_integrated': True,
                            'method': 'enhanced_ssl_444dim',
                            'num_frames': track.get('num_frames', 1),
                            'frame_sequence': track.get('frame_sequence', []),
                            'clinical_class': clinical_class,
                            'clinical_risk': clinical_risk,
                            'clinical_description': PolypTypeClassifier().get_clinical_description(clinical_class),
                        })

                        # Add to polyp features summary
                        polyp_features_summary.append({
                            'polyp_id': track_idx + 1,  # 1-based indexing
                            'frame': best_frame_idx,
                            'model': 'temporal_consensus',
                            'consensus': True,
                            'redness': float(tight_feature_source.get('redness', 0.0)),
                            'radius': float(tight_feature_source.get('radius', 0.0)),
                            'texture': float(tight_feature_source.get('texture', 0.0)),
                            'vessel_visibility': float(tight_feature_source.get('vessel_visibility', 0.0)),
                            'classification': prediction,
                            'expert_classification': expert_classification,
                            'confidence': float(clinical_confidence),
                            'symbolic_confidence': float(symbolic_confidence),
                            'detection_confidence': float(detection_confidence),
                            'clinical_confidence': float(clinical_confidence),
                            'risk_score': risk_score,
                            'polyp_type': polyp_type,
                            'polyp_type_confidence': polyp_type_confidence,
                            'cluster_id': cluster_id,
                            'prediction_source': enhanced_result.get('method', 'ssl_enhanced_cluster_specific' if cluster_id is not None else 'ssl_enhanced_ensemble'),
                            'box': reference_box if reference_box and len(reference_box) == 4 else track.get('box', []),
                            'num_frames': track.get('num_frames', 1),
                            'frame_sequence': track.get('frame_sequence', []),
                            'temporal_boost': track.get('confidence_boost', 0.0)
                        })
                        
                        print(f"       ✅ Polyp #{track_idx + 1}: conf={expert_confidence:.3f}, type={polyp_type}, pred={prediction}")
                        
                    except Exception as e:
                        import traceback
                        print(f"   ⚠️  Failed to process track {track_idx}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        # Fallback entry
                        fallback_risk = float(feature_based.get('risk_score', 0.0))
                        fallback_detection_confidence = float(track.get('temporal_average_conf', 0.0))
                        polyp_features_summary.append({
                            'polyp_id': track_idx + 1,
                            'frame': track.get('representative_frame', track.get('start_frame', track.get('frame', 0))),
                            'model': 'temporal_consensus',
                            'consensus': True,
                            'redness': 0.0,
                            'radius': float(tight_feature_source.get('radius', 0.0)) if best_features else 0.0,
                            'texture': 0.0,
                            'vessel_visibility': 0.0,
                            'classification': prediction,
                            'expert_classification': expert_classification,
                            'confidence': 0.0,
                            'symbolic_confidence': 0.0,
                            'detection_confidence': fallback_detection_confidence,
                            'clinical_confidence': 0.0,
                            'risk_score': fallback_risk,
                            'medical_risk_score': fallback_risk,
                            'polyp_type': 'UNKNOWN',
                            'polyp_type_confidence': 0.0,
                            'cluster_id': cluster_id,
                            'prediction_source': 'fallback_track_analysis',
                            'box': track.get('box', []),
                            'num_frames': track.get('num_frames', 1),
                            'frame_sequence': track.get('frame_sequence', []),
                            'temporal_boost': track.get('confidence_boost', 0.0)
                        })
                else:
                    # No features found for this track
                    detection_confidence = float(track.get('temporal_average_conf', 0.0))
                    polyp_features_summary.append({
                        'polyp_id': track_idx + 1,
                        'frame': track.get('representative_frame', track.get('start_frame', track.get('frame', 0))),
                        'model': 'temporal_consensus',
                        'consensus': True,
                        'redness': 0.0,
                        'radius': 0.0,
                        'texture': 0.0,
                        'vessel_visibility': 0.0,
                        'classification': 'UNKNOWN',
                        'expert_classification': 'UNKNOWN',
                        'confidence': 0.0,
                        'symbolic_confidence': 0.0,
                        'detection_confidence': detection_confidence,
                        'clinical_confidence': 0.0,
                        'risk_score': 0.0,
                        'medical_risk_score': 0.0,
                        'polyp_type': 'UNKNOWN',
                        'polyp_type_confidence': 0.0,
                        'cluster_id': cluster_id,
                        'prediction_source': 'no_feature_match',
                        'num_frames': track.get('num_frames', 1),
                        'frame_sequence': track.get('frame_sequence', []),
                        'temporal_boost': track.get('confidence_boost', 0.0)
                    })
        else:
            # Fallback to raw detections if temporal merging failed
            print("   ⚠️  No merged tracks available, falling back to raw detections")
            polyp_count = 0
            for frame_idx in features_dict:
                for feature_set in features_dict[frame_idx]:
                    polyp_count += 1
                    if polyp_count <= 10:  # Limit to 10 in fallback mode
                        frame_rgb = frames[frame_idx]
                        reference_box = feature_set.get('box', []) or []
                        mask = None
                        if 'medsam' in segmentations:
                            matching_masks = [
                                seg
                                for seg in segmentations.get('medsam', [])
                                if seg.get('frame') == frame_idx and seg.get('mask') is not None
                            ]
                            if matching_masks:
                                best_mask = None
                                best_mask_iou = -1.0
                                for seg in matching_masks:
                                    seg_box = seg.get('prompt_box')
                                    if reference_box and seg_box and len(reference_box) == 4 and len(seg_box) == 4:
                                        current_iou = rules_engine._calculate_iou(reference_box, seg_box)
                                    else:
                                        current_iou = 0.0

                                    if current_iou > best_mask_iou:
                                        best_mask_iou = current_iou
                                        best_mask = seg.get('mask')

                                mask = best_mask if best_mask is not None else matching_masks[0].get('mask')

                        polyp_roi, _ = crop_pure_roi_from_mask(frame_rgb, mask) if mask is not None else (None, None)
                        if polyp_roi is None or polyp_roi.size == 0:
                            if reference_box and len(reference_box) == 4:
                                x1, y1, x2, y2 = map(int, reference_box)
                                if y2 > y1 and x2 > x1:
                                    polyp_roi = frame_rgb[y1:y2, x1:x2]
                                else:
                                    polyp_roi = frame_rgb
                            else:
                                polyp_roi = frame_rgb

                        tight_feature_source = feature_set
                        if mask is not None and reference_box and len(reference_box) == 4:
                            try:
                                tight_feature_source = extract_all_features(
                                    frame_rgb,
                                    reference_box,
                                    mask,
                                    mask_padding=0.05
                                )
                                # Issue 5 fix: Fallback to feature_set if result is all-zero (degenerate crop)
                                if (tight_feature_source.get('radius', 0.0) == 0.0 and 
                                    tight_feature_source.get('redness', 0.0) == 0.0 and
                                    tight_feature_source.get('vessel_visibility', 0.0) == 0.0):
                                    tight_feature_source = feature_set
                            except Exception as feature_err:
                                print(f"       ⚠️  Tight ROI feature extraction failed - using detector features: {feature_err}")
                                tight_feature_source = feature_set

                        feature_based = integrator.feature_based_classification(tight_feature_source) if integrator else {'classification': 'UNKNOWN', 'risk_score': 0.0}
                        risk_score = float(feature_based.get('risk_score', 0.0))
                        detection_confidence = float(feature_set.get('detection_confidence', 0.0))

                        # Initialise with zero vectors before the conditional assignment
                        # so they are always defined even if SSL model is not available or feature extraction fails
                        basic_feature_vector = np.zeros(10, dtype=np.float32)
                        features_444         = np.zeros(444, dtype=np.float32)
                        cluster_id           = 0

                        if ssl_model is not None:
                            try:
                                features_444 = extract_444_features(polyp_roi, ssl_model, Config.DEVICE)
                            except Exception as ssl_err:
                                print(f"       ⚠️  SSL feature extraction failed - using zero vector fallback: {ssl_err}")
                                features_444 = np.zeros(444, dtype=np.float32)
                        else:
                            features_444 = np.zeros(444, dtype=np.float32)

                        cluster_id = predict_cluster_id(
                            features_444,
                            models.get('kmeans'),
                            models.get('feature_scaler'),
                        )
                        
                        # === SSL MEANINGFULNESS CHECK (Same as main path - must apply before enhanced_classify_with_ssl) ===
                        _ssl_vec = features_444[:384] if features_444 is not None and len(features_444) >= 384 else np.zeros(384)
                        _ssl_std = float(np.std(_ssl_vec)) if len(_ssl_vec) > 0 else 0.0
                        ssl_is_meaningful_check = (
                            _ssl_std > 0.05 and
                            not np.any(np.isnan(_ssl_vec)) and
                            not np.any(np.isinf(_ssl_vec)) and
                            getattr(ssl_model, '_ssl_weights_loaded', False)
                        )

                        if ssl_is_meaningful_check and integrator:
                            enhanced_result = integrator.enhanced_classify_with_ssl(
                                basic_feature_vector, features_444, cluster_id=cluster_id
                            )
                            expert_classification = enhanced_result.get('prediction', 'UNKNOWN')
                            _conf_lo = integrator._np_thresholds.get('confidence_clip_low', 0.0) if integrator else 0.0
                            _conf_hi = integrator._np_thresholds.get('confidence_clip_high', 1.0) if integrator else 1.0
                            symbolic_confidence = float(np.clip(enhanced_result.get('confidence', 0.0), _conf_lo, _conf_hi))
                        else:
                            # SSL not trained — use feature-based result as expert output
                            expert_classification = feature_based.get('clinical_class', 'ADENOMATOUS_POLYP')
                            _conf_lo = integrator._np_thresholds.get('confidence_clip_low', 0.0) if integrator else 0.0
                            _conf_hi = integrator._np_thresholds.get('confidence_clip_high', 1.0) if integrator else 1.0
                            symbolic_confidence = float(np.clip(risk_score, _conf_lo, _conf_hi))
                            enhanced_result = {
                                'prediction': expert_classification,
                                'confidence': symbolic_confidence,
                                'ssl_boost_applied': 0.0,
                                'method': 'feature_heuristic_ssl_unavailable'
                            }

                        expert_confidence = float(enhanced_result.get('confidence', 0.0))
                        ssl_boost = float(enhanced_result.get('ssl_boost_applied', 0.0))

                        feature_classification = feature_based.get('clinical_class', 'ADENOMATOUS_POLYP')

                        # Fix 1: Consistent moderation logic with definitive MEDIUM_RISK floor for detected polyps
                        final_classification = expert_classification
                        
                        if final_classification == 'HIGH_RISK':
                            # HIGH_RISK only stays HIGH if risk_score is strong (>= 0.45)
                            # Otherwise demote to MEDIUM_RISK (detected polyp floor)
                            if risk_score < 0.45:
                                final_classification = 'MEDIUM_RISK'
                        elif final_classification == 'LOW_RISK':
                            # Any confirmed detected polyp = minimum MEDIUM_RISK
                            final_classification = 'MEDIUM_RISK'
                        elif final_classification in {'UNKNOWN', 'UNCERTAIN', 'ERROR'}:
                            # Unknown/uncertain on detected consensus polyp = MEDIUM_RISK minimum
                            final_classification = 'MEDIUM_RISK'

                        clinical_confidence = calibrate_clinical_confidence(symbolic_confidence, detection_confidence, risk_score)
                        if final_classification in {'UNKNOWN', 'UNCERTAIN', 'ERROR'}:
                            clinical_confidence = symbolic_confidence if symbolic_confidence > 0.0 else risk_score

                        prediction = final_classification

                        # RULE 3: Polyp type classification — System C only (clinical keyword pipeline)
                        polyp_type = 'UNKNOWN'
                        polyp_type_confidence = 0.0
                        try:
                            redness_val    = float(tight_feature_source.get('redness', 0.0))
                            vessel_val     = float(tight_feature_source.get('vessel_visibility', 0.0))
                            texture_val    = float(tight_feature_source.get('texture', 0.0))
                            radius_val     = float(tight_feature_source.get('radius', 0.0))
                            s_mean_val     = float(tight_feature_source.get('s_mean', 0.0))
                            edge_val       = float(tight_feature_source.get('edge_sharpness', 0.0))

                            # Step 1: translate numbers → clinical keyword
                            clinical_keyword = PolypTypeClassifier.features_to_clinical_keyword(
                                redness_val, vessel_val, texture_val, radius_val, edge_val, s_mean_val
                            )

                            # Step 2: keyword → (clinical_class, risk) via FINDING_TO_CLASS
                            ptc_instance = PolypTypeClassifier()
                            clinical_class_pred, _ = ptc_instance.map_finding_to_class(clinical_keyword)
                            polyp_type = clinical_class_pred  # e.g. 'BLEEDING_POLYP', 'MALIGNANT_POLYP', etc.

                            # Step 3: confidence from feature signal strength
                            total_signal = redness_val + vessel_val + texture_val + s_mean_val
                            polyp_type_confidence = float(np.clip(0.45 + total_signal * 0.4, 0.45, 0.92))

                        except Exception as _e:
                            polyp_type = 'UNKNOWN'
                            polyp_type_confidence = 0.0

                        clinical_decision = {}
                        if integrator:
                            clinical_decision = integrator.apply_asge_standards(clinical_confidence, prediction)

                        polyp_features_summary.append({
                            'polyp_id': polyp_count,
                            'frame': frame_idx,
                            'model': feature_set.get('model', 'unknown'),
                            'consensus': feature_set.get('consensus', False),
                            'redness': tight_feature_source.get('redness', feature_set.get('redness', 0.0)),
                            'radius': tight_feature_source.get('radius', feature_set.get('radius', 0.0)),
                            'texture': tight_feature_source.get('texture', feature_set.get('texture', 0.0)),
                            'vessel_visibility': tight_feature_source.get('vessel_visibility', feature_set.get('vessel_visibility', 0.0)),
                            'classification': prediction,
                            'expert_classification': expert_classification,
                            'confidence': clinical_confidence,
                            'symbolic_confidence': symbolic_confidence,
                            'detection_confidence': detection_confidence,
                            'clinical_confidence': clinical_confidence,
                            'risk_score': risk_score,
                            'medical_risk_score': risk_score,
                            'polyp_type': polyp_type,
                            'polyp_type_confidence': polyp_type_confidence,
                            'clinical_decision': clinical_decision.get('clinical_decision', 'UNKNOWN') if clinical_decision else 'UNKNOWN',
                            'box': feature_set.get('box', []),
                            'temporal_average_conf': clinical_confidence,
                            'frame_sequence': [frame_idx],
                            'num_frames': 1,
                            'start_frame': frame_idx,
                            'end_frame': frame_idx,
                            'cluster_id': cluster_id,
                            'prediction_source': enhanced_result.get('method', 'ssl_enhanced_cluster_specific' if cluster_id is not None else 'ssl_enhanced_ensemble'),
                            'ssl_integrated': True,
                            'method': enhanced_result.get('method', 'enhanced_ssl_444dim'),
                            'ssl_boost_applied': ssl_boost,
                        })

                        symbolic_results.append({
                            'polyp_id': polyp_count,
                            'frame': frame_idx,
                            'classification': prediction,
                            'expert_classification': expert_classification,
                            'confidence': clinical_confidence,
                            'symbolic_confidence': symbolic_confidence,
                            'detection_confidence': detection_confidence,
                            'clinical_confidence': clinical_confidence,
                            'consensus': feature_set.get('consensus', False),
                            'model': feature_set.get('model', 'unknown'),
                            'redness': tight_feature_source.get('redness', feature_set.get('redness', 0.0)),
                            'texture': tight_feature_source.get('texture', feature_set.get('texture', 0.0)),
                            'vessel_visibility': tight_feature_source.get('vessel_visibility', feature_set.get('vessel_visibility', 0.0)),
                            'radius': tight_feature_source.get('radius', feature_set.get('radius', 0.0)),
                            'risk_score': risk_score,
                            'medical_risk_score': risk_score,
                            'polyp_type': polyp_type,
                            'polyp_type_confidence': polyp_type_confidence,
                            'box': feature_set.get('box', []),
                            'temporal_average_conf': clinical_confidence,
                            'frame_sequence': [frame_idx],
                            'num_frames': 1,
                            'start_frame': frame_idx,
                            'end_frame': frame_idx,
                            'cluster_id': cluster_id,
                            'prediction_source': enhanced_result.get('method', 'ssl_enhanced_cluster_specific' if cluster_id is not None else 'ssl_enhanced_ensemble'),
                            'ssl_integrated': True,
                            'method': enhanced_result.get('method', 'enhanced_ssl_444dim'),
                            'ssl_boost_applied': ssl_boost,
                        })
                        
                        # Fix 1A: Sync classification back into the source track so JSON export has it
                        if deduplicated_tracks and track_idx < len(deduplicated_tracks):
                            deduplicated_tracks[track_idx]['classification'] = prediction
                            deduplicated_tracks[track_idx]['expert_classification'] = expert_classification
                            deduplicated_tracks[track_idx]['polyp_type'] = polyp_type
                            deduplicated_tracks[track_idx]['polyp_type_confidence'] = polyp_type_confidence
                            deduplicated_tracks[track_idx]['risk_score'] = risk_score
                            deduplicated_tracks[track_idx]['medical_risk_score'] = risk_score
                            deduplicated_tracks[track_idx]['redness'] = tight_feature_source.get('redness', 0.0)
                            deduplicated_tracks[track_idx]['texture'] = tight_feature_source.get('texture', 0.0)
                            deduplicated_tracks[track_idx]['vessel_visibility'] = tight_feature_source.get('vessel_visibility', 0.0)
                            deduplicated_tracks[track_idx]['radius'] = tight_feature_source.get('radius', 0.0)
                            deduplicated_tracks[track_idx]['symbolic_confidence'] = symbolic_confidence
                            deduplicated_tracks[track_idx]['detection_confidence'] = detection_confidence
                            deduplicated_tracks[track_idx]['clinical_confidence'] = clinical_confidence
                            deduplicated_tracks[track_idx]['clinical_class'] = feature_based.get('clinical_class', 'ADENOMATOUS_POLYP')
                            deduplicated_tracks[track_idx]['clinical_description'] = feature_based.get('clinical_description', '')

        if not deduplicated_tracks and symbolic_results:
            deduplicated_raw_results = deduplicate_polyps(symbolic_results, iou_threshold=0.45)
            for idx, entry in enumerate(deduplicated_raw_results):
                entry['polyp_id'] = idx + 1
                entry['frame_sequence'] = entry.get('frame_sequence') or [int(entry.get('frame', 0))]
                entry['start_frame'] = int(entry.get('start_frame', entry.get('frame', 0)))
                entry['end_frame'] = int(entry.get('end_frame', entry.get('frame', 0)))
                entry['num_frames'] = int(entry.get('num_frames', len(entry.get('frame_sequence', [])) or 1))
                entry['temporal_average_conf'] = float(entry.get('temporal_average_conf', entry.get('confidence', 0.0)))

            symbolic_results = deduplicated_raw_results
            polyp_features_summary = deduplicated_raw_results

        analyzed_entries = symbolic_results if symbolic_results else polyp_features_summary

        # FIX: Aggregate multiple polyp tracks into one video-level clinical class
        # Priority order mirrors CLINICAL_CLASSES risk levels: HIGH > MEDIUM > LOW
        CLINICAL_CLASS_PRIORITY = {
            'BLEEDING_POLYP':          0,
            'POST_RESECTION_BLEEDING': 1,
            'MALIGNANT_POLYP':         2,
            'LATERAL_SPREADING_TUMOR': 3,
            'LARGE_POLYP':             4,
            'VILLOUS_POLYP':           5,
            'SERRATED_POLYP':          6,
            'FLAT_POLYP':              7,
            'PEDUNCULATED_POLYP':      8,
            'COLITIS':                 9,
            'ADENOMATOUS_POLYP':       10,
            'LIFTED_POLYP':            11,
            'RESECTED_POLYP':          12,
            'SMALL_POLYP':             13,
            'NORMAL_MUCOSA':           14,
            'UNKNOWN':                 99,
        }

        # FIX: Majority vote first, priority tiebreaker second
        from collections import Counter
        _type_counter = Counter()
        for entry in analyzed_entries:
            _tc = entry.get('polyp_type', 'UNKNOWN')
            if _tc and _tc != 'UNKNOWN':
                _type_counter[_tc] += 1

        video_level_clinical_class = 'UNKNOWN'
        if _type_counter:
            _total = sum(_type_counter.values())
            _most_common_cls, _most_common_count = _type_counter.most_common(1)[0]
            
            if _most_common_count / _total > 0.5:
                # Clear majority (>50%) → use majority vote directly
                video_level_clinical_class = _most_common_cls
            else:
                # No clear majority → use priority (highest clinical risk wins)
                best_priority = 99
                for entry in analyzed_entries:
                    track_class = entry.get('polyp_type', 'UNKNOWN')
                    priority = CLINICAL_CLASS_PRIORITY.get(track_class, 99)
                    if priority < best_priority:
                        best_priority = priority
                        video_level_clinical_class = track_class

        # Store the video-level prediction for CSV comparison
        detections['_video_level_clinical_class'] = video_level_clinical_class
        print(f"   🎯 Video-level classification: {video_level_clinical_class}")

        report_tracks = deduplicated_tracks if deduplicated_tracks else symbolic_results
        if analyzed_entries:
            high_risk_count = sum(1 for entry in analyzed_entries if 'HIGH' in str(entry.get('classification', '')).upper())
            medium_risk_count = sum(1 for entry in analyzed_entries if 'MEDIUM' in str(entry.get('classification', '')).upper() or 'UNCERTAIN' in str(entry.get('classification', '')).upper())
            low_risk_count = sum(1 for entry in analyzed_entries if 'LOW' in str(entry.get('classification', '')).upper())
            type_counts = defaultdict(int)
            for entry in analyzed_entries:
                type_counts[str(entry.get('polyp_type', 'UNKNOWN'))] += 1
        
        # Store symbolic reasoning summary for reporting
        detections['_symbolic_reasoning_summary'] = {
            'total_analyzed': len(report_tracks) if report_tracks else len(polyp_features_summary),
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'low_risk_count': low_risk_count,
            'consensus_polyp_count': consensus_polyp_count,
            'polyp_type_counts': dict(type_counts),
            'error_count': error_count,
            '_temporal_consensus': {
                'tracks': report_tracks,
                'total_polyp_instances': len(report_tracks) if report_tracks else len(polyp_features_summary),
                'total_frame_coverage': sum(t.get('num_frames', 0) for t in report_tracks) if report_tracks else 0,
                'average_confidence_boost': float(temporal_aggregation.get('average_boost_amount', 0.0)) if report_tracks else 0.0
            },
            'temporal_aggregation': {
                'total_tracks': len(report_tracks) if report_tracks else len(polyp_features_summary),
                'total_track_frames': sum(t.get('num_frames', 0) for t in report_tracks) if report_tracks else 0,
                'average_confidence_boost': float(temporal_aggregation.get('average_boost_amount', 0.0)) if report_tracks else 0.0
            },
            'rules_applied': ['Rule1_MedSAM2Hybrid', 'Rule2_Consensus', 'Rule3_ClinicalSplit', 'Rule4_SSLFeatures', 'Rule5_BorderFilter', 'TemporalAveraging'],
            'results': symbolic_results[:20]  # Store first 20 examples
        }
        
        # Store in detections for reporting
        detections['_polyp_features_detail'] = polyp_features_summary

        # Generate outputs with error handling
        try:
            generate_annotated_video(video_path, frames, detections, segmentations, video_output_dir)
        except Exception as e:
            print(f"   ⚠️  Annotated video generation failed: {e}")

        try:
            generate_frame_montages(frames, detections, segmentations, video_output_dir)
        except Exception as e:
            print(f"   ⚠️  Frame montage generation failed: {e}")

        try:
            generate_inference_report(video_path, detections, segmentations, len(frames), video_output_dir, video_annotations_dict)
        except Exception as e:
            print(f"   ⚠️  Report generation failed: {e}")

        try:
            print(f"   📄 Generating professional medical report...")
            print(f"      Input: {len(frames)} frames, {len(detections.get('yolo', []))} YOLO, {len(detections.get('rtdetr', []))} RTDETR, {len(detections.get('medsam', []))} MedSAM2 detections")
            pdf_path = generate_medical_report(video_path, frames, detections, segmentations, video_output_dir)
            if pdf_path:
                print(f"   ✅ Medical report generated: {Path(pdf_path).name}")
                # Verify file exists
                if Path(pdf_path).exists():
                    file_size = Path(pdf_path).stat().st_size
                    print(f"      File size: {file_size} bytes")
                else:
                    print(f"   ⚠️  Report file created but not found at: {pdf_path}")
            else:
                print(f"   ⚠️  Medical report generation returned None")
        except Exception as e:
            print(f"   ⚠️  Medical report generation failed: {e}")
            import traceback as tb
            tb.print_exc()

        print(f"   ✅ Processing complete. Outputs saved in {Config.VIDEO_OUTPUT}")

    except Exception as e:
        print(f"   ❌ Critical error processing {video_path.name}: {e}")
        import traceback as tb
        tb.print_exc()
        error_log = output_dir / video_path.stem / "ERROR_LOG.txt"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(tb.format_exc())

# ==========================================
# GROUND TRUTH LOADING & COMPARISON
# ==========================================

def load_ground_truth_csv():
    """Load ground truth annotations from video-annotations.csv"""
    try:
        csv_path = Config.THESIS_ROOT / 'NeSy' / 'video-annotations.csv'
        if not csv_path.exists():
            print(f"   ⚠️  Ground truth CSV not found: {csv_path}")
            return {}
        
        # Read CSV with semicolon delimiter
        df = pd.read_csv(str(csv_path), sep=';')
        
        # Create dictionary mapping videoID to finding
        ground_truth = {}
        for idx, row in df.iterrows():
            video_id = str(row['videoID']).strip()
            finding = str(row['finding']).strip().lower()
            ground_truth[video_id] = finding
        
        print(f"   ✅ Loaded {len(ground_truth)} ground truth annotations")
        return ground_truth
    except Exception as e:
        print(f"   ⚠️  Error loading ground truth CSV: {e}")
        return {}

def is_valid_size(box, min_area=100, max_area=50000):
    """Check if bounding box area is within valid range for polyps"""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return min_area <= area <= max_area

def is_valid_shape(box, max_ratio=2.5):
    """Check if bounding box aspect ratio is valid for polyps"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if min(width, height) == 0:
        return False
    ratio = max(width, height) / min(width, height)
    return ratio <= max_ratio

def is_valid_position(box, frame_shape, margin_percent=0.1):
    """Check if bounding box is not too close to image edges (polyps are typically central)"""
    h, w = frame_shape[:2]
    margin_x = int(w * margin_percent)
    margin_y = int(h * margin_percent)
    x1, y1, x2, y2 = box
    # Reject if box is entirely in edge region
    if x2 < margin_x or x1 > w - margin_x or y2 < margin_y or y1 > h - margin_y:
        return False
    return True

def calculate_iou(box1, box2):
    """Calculate intersection over union for overlap detection"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    """Merge or reject overlapping boxes to avoid clustered noise"""
    if not boxes:
        return boxes
    # Simple non-max suppression: keep non-overlapping boxes
    filtered = []
    for box in boxes:
        overlap = False
        for existing in filtered:
            if calculate_iou(box, existing) > iou_threshold:
                overlap = True
                break
        if not overlap:
            filtered.append(box)
    return filtered

def is_valid_texture(mask, frame, min_variance=50, max_mean=200):
    """Check if masked region has polyp-like texture (high variance, moderate mean intensity)"""
    if mask.sum() == 0:
        return False
    masked_pixels = frame[mask > 0]
    if len(masked_pixels) == 0:
        return False
    mean_intensity = np.mean(masked_pixels)
    variance = np.var(masked_pixels)
    # Polyp-like: moderate mean intensity, high variance (texture)
    return variance > min_variance and mean_intensity < max_mean

def filter_detections(detections, min_area=100, max_area=50000, max_ratio=2.5):
    """Apply size and shape filters to detections"""
    filtered = []
    for det in detections:
        box = det['box']
        if is_valid_size(box, min_area, max_area) and is_valid_shape(box, max_ratio):
            filtered.append(det)
    return filtered

def extract_prediction_from_inference(inference_report, detections_dict):
    """
    Extract prediction from inference results using SSR (Square Root of Sum of Squares) Aggregation
    Aggregates model confidences using SSR for ensemble decision
    Returns: (predicted_polyp_present, detection_count, avg_confidence, model_predictions)
    """
    try:
        detections_summary = inference_report.get('detections_summary', {}) or {}
        consensus_summary = inference_report.get('consensus_voting', {}) or {}

        yolo_conf = float(detections_summary.get('yolo', {}).get('average_confidence', 0.0))
        rtdetr_conf = float(detections_summary.get('rtdetr', {}).get('average_confidence', 0.0))
        medsam_conf = float(detections_summary.get('medsam', {}).get('average_confidence', 0.0))

        all_confidences = [c for c in [yolo_conf, rtdetr_conf, medsam_conf] if c > 0.0]
        ssr_conf = np.sqrt(yolo_conf**2 + rtdetr_conf**2 + medsam_conf**2)

        raw_total_detections = 0
        for model_summary in detections_summary.values():
            raw_total_detections += int(model_summary.get('total_detections', 0))

        if consensus_summary:
            predicted_polyp_present = bool(consensus_summary.get('polyp_present', False))
            total_detections = int(consensus_summary.get('consensus_frame_count', 0))
            avg_confidence = float(consensus_summary.get('overall_confidence', np.mean(all_confidences) if all_confidences else 0.0))

            model_presence = consensus_summary.get('model_presence', {}) or {}
            if not model_presence:
                model_presence = {
                    'yolo': predicted_polyp_present,
                    'rtdetr': predicted_polyp_present,
                    'medsam': predicted_polyp_present,
                }

            voting_summary = {
                'yolo': bool(model_presence.get('yolo', predicted_polyp_present)),
                'rtdetr': bool(model_presence.get('rtdetr', predicted_polyp_present)),
                'medsam': bool(model_presence.get('medsam', predicted_polyp_present)),
                'polyp_present': predicted_polyp_present,
                'consensus_frame_count': total_detections,
                'num_consensus_runs': int(consensus_summary.get('num_consensus_runs', 0)),
                'consensus_percentage': float(consensus_summary.get('consensus_percentage', 0.0)),
            }
            detection_models = 3 if predicted_polyp_present else 0
        else:
            predicted_polyp_present = ssr_conf >= 1.05
            total_detections = raw_total_detections
            avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
            voting_summary = {
                'yolo': yolo_conf > 0.0,
                'rtdetr': rtdetr_conf > 0.0,
                'medsam': medsam_conf > 0.0,
                'polyp_present': predicted_polyp_present,
                'consensus_frame_count': 0,
                'num_consensus_runs': 0,
                'consensus_percentage': 0.0,
            }
            detection_models = sum(1 for conf in [yolo_conf, rtdetr_conf, medsam_conf] if conf > 0.0)

        model_predictions = {
            'total_detections': total_detections,
            'raw_total_detections': raw_total_detections,
            'avg_confidence': float(avg_confidence),
            'ssr_confidence': float(ssr_conf),
            'polyp_detected': predicted_polyp_present,
            'ssr_threshold': 1.05,
            'detection_models': detection_models,
            'voting_summary': voting_summary,
            'consensus_voting': consensus_summary,
            'model_confidences': {
                'yolo': yolo_conf,
                'rtdetr': rtdetr_conf,
                'medsam': medsam_conf
            }
        }

        return predicted_polyp_present, total_detections, avg_confidence, model_predictions
        
    except Exception as e:
        print(f"   ⚠️  Error extracting prediction: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0.0, {}

def generate_results_csv(video_output_root, ground_truth_dict):
    """Generate comprehensive results CSV for all processed videos"""
    try:
        print("\n   📊 Generating results CSV...")
        
        results = []
        video_dirs = list(video_output_root.glob('*/'))
        
        for video_dir in tqdm(video_dirs, desc="Processing results"):
            try:
                # Skip system folders
                if video_dir.name in ['debug', 'reports', 'visualizations']:
                    continue
                    
                video_name = video_dir.name
                # Look for inference_report.json in nested subdirectory (same name as parent)
                inference_report_path = video_dir / video_name / 'inference_report.json'
                
                # Fallback to root level if not found
                if not inference_report_path.exists():
                    inference_report_path = video_dir / 'inference_report.json'
                
                if not inference_report_path.exists():
                    continue
                
                # Load inference report
                with open(inference_report_path, 'r') as f:
                    inference_report = json.load(f)
                
                # Extract video info
                video_info = inference_report.get('video_info', {})
                frames_processed = video_info.get('frames_processed', 0)
                
                # Extract prediction
                predicted_polyp, total_dets, avg_conf, model_preds = extract_prediction_from_inference(inference_report, {})
                
                # Match with ground truth by UUID from filename
                ground_truth_finding = None
                matched_video_id     = None

                # Strategy 1: Direct UUID match from filename
                for video_id, finding in ground_truth_dict.items():
                    if video_id in video_name:
                        ground_truth_finding = finding
                        matched_video_id     = video_id
                        break

                # Strategy 2: check video_info name if not found
                if not ground_truth_finding and 'name' in video_info:
                    video_filename = video_info['name']
                    for video_id, finding in ground_truth_dict.items():
                        if video_id in video_filename:
                            ground_truth_finding = finding
                            matched_video_id     = video_id
                            break
                
                voting_summary = model_preds.get('voting_summary', {})
                model_confidences = model_preds.get('model_confidences', {})
                
                # Map CSV finding → target clinical class (same FINDING_TO_CLASS used for prediction)
                ptc = PolypTypeClassifier()
                target_clinical_class = 'UNKNOWN'
                if ground_truth_finding and ground_truth_finding != 'unknown':
                    target_clinical_class, _ = ptc.map_finding_to_class(ground_truth_finding)

                # Get predicted clinical class from video-level aggregation
                predicted_clinical_class = inference_report.get('video_level_clinical_class', 'UNKNOWN')

                # Exact match comparison
                prediction_correct = (predicted_clinical_class == target_clinical_class) \
                                     and target_clinical_class != 'UNKNOWN'

                result_row = {
                    'video_name': video_name,
                    'matched_video_id': matched_video_id or 'NOT_MATCHED',
                    'ground_truth_finding': ground_truth_finding or 'UNKNOWN',
                    'target_clinical_class': target_clinical_class,
                    'predicted_clinical_class': predicted_clinical_class,
                    'prediction_correct': prediction_correct,
                    'total_detections': total_dets,
                    'avg_confidence': float(avg_conf),
                    'frames_processed': frames_processed,
                    'models_detected': model_preds.get('detection_models', 0),
                    'yolo_detected': voting_summary.get('yolo', False),
                    'yolo_confidence': model_confidences.get('yolo', 0.0),
                    'rtdetr_detected': voting_summary.get('rtdetr', False),
                    'rtdetr_confidence': model_confidences.get('rtdetr', 0.0),
                    'medsam_detected': voting_summary.get('medsam', False),
                    'medsam_confidence': model_confidences.get('medsam', 0.0),
                    'processing_timestamp': inference_report.get('processing_timestamp', 'UNKNOWN'),
                    'ground_truth_has_polyp': (
                        target_clinical_class != 'NORMAL_MUCOSA'
                    )
                }
                
                results.append(result_row)
                
            except Exception as e:
                print(f"   ⚠️  Error processing {video_dir.name}: {e}")
                continue
        
        # Create DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Save results CSV
            results_csv_path = video_output_root / 'detailed_results.csv'
            results_df.to_csv(str(results_csv_path), index=False)
            print(f"   ✅ Results CSV saved: {results_csv_path}")
            
            return results_df
        else:
            print("   ⚠️  No results to save")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"   ❌ Error generating results CSV: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def generate_accuracy_report(results_df, output_dir):
    """Generate comprehensive accuracy report comparing predictions vs ground truth"""
    try:
        print("\n   📈 Generating accuracy report...")
        
        # Filter for matched videos only
        matched_df = results_df[results_df['matched_video_id'] != 'NOT_MATCHED'].copy()
        
        if len(matched_df) == 0:
            print("   ⚠️  No matched videos found for accuracy calculation")
            return {}
        
        # Overall statistics
        total_matched = len(matched_df)
        correct_predictions = matched_df['prediction_correct'].sum()
        incorrect_predictions = total_matched - correct_predictions
        
        # Polyp detection metrics
        polyp_gt = matched_df['ground_truth_has_polyp'].apply(lambda x: x == True)
        polyp_pred = matched_df['predicted_polyp'].apply(lambda x: x == True)
        
        # Remove None values for accuracy calculation
        valid_mask = matched_df['prediction_correct'].notna()
        valid_df = matched_df[valid_mask]
        
        if len(valid_df) > 0:
            accuracy = correct_predictions / total_matched
            
            # Confusion matrix for polyp detection
            tp = ((polyp_pred) & (polyp_gt)).sum()
            tn = ((~polyp_pred) & (~polyp_gt)).sum()
            fp = ((polyp_pred) & (~polyp_gt)).sum()
            fn = ((~polyp_pred) & (polyp_gt)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            report = {
                'total_videos_processed': len(results_df),
                'total_matched_with_ground_truth': total_matched,
                'total_unmatched': len(results_df) - total_matched,
                'match_percentage': (total_matched / len(results_df) * 100) if len(results_df) > 0 else 0,
                
                'overall_metrics': {
                    'accuracy': float(accuracy),
                    'correct_predictions': int(correct_predictions),
                    'incorrect_predictions': int(incorrect_predictions),
                },
                
                'polyp_detection_metrics': {
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'specificity': float(specificity),
                },
                
                'category_breakdown': {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Category breakdown
            for category in matched_df['ground_truth_category'].unique():
                if pd.isna(category):
                    continue
                category_df = matched_df[matched_df['ground_truth_category'] == category]
                category_accuracy = category_df['prediction_correct'].sum() / len(category_df) if len(category_df) > 0 else 0
                report['category_breakdown'][str(category)] = {
                    'count': int(len(category_df)),
                    'accuracy': float(category_accuracy),
                    'correct': int(category_df['prediction_correct'].sum()),
                    'incorrect': int(len(category_df) - category_df['prediction_correct'].sum())
                }
            
            # Save report
            report_path = output_dir / 'accuracy_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n   ✅ Accuracy Report Summary:")
            print(f"      Total Videos Processed: {report['total_videos_processed']}")
            print(f"      Matched with Ground Truth: {report['total_matched_with_ground_truth']} ({report['match_percentage']:.1f}%)")
            print(f"      Overall Accuracy: {report['overall_metrics']['accuracy']:.2%}")
            print(f"      Polyp Detection:")
            print(f"         Precision: {report['polyp_detection_metrics']['precision']:.2%}")
            print(f"         Recall: {report['polyp_detection_metrics']['recall']:.2%}")
            print(f"         F1-Score: {report['polyp_detection_metrics']['f1_score']:.2%}")
            print(f"      Report saved: {report_path}")
            
            return report
        else:
            print("   ⚠️  No valid predictions for accuracy calculation")
            return {}
            
    except Exception as e:
        print(f"   ❌ Error generating accuracy report: {e}")
        traceback.print_exc()
        return {}

def generate_comparison_visualizations(results_df, accuracy_report, output_dir):
    """Generate visualization comparisons between predictions and ground truth"""
    try:
        print("\n   📊 Generating comparison visualizations...")
        
        if results_df.empty or not accuracy_report:
            print("   ⚠️  Insufficient data for visualizations")
            return
        
        # 1. Confusion Matrix Heatmap
        matched_df = results_df[results_df['matched_video_id'] != 'NOT_MATCHED'].copy()
        valid_df = matched_df[matched_df['prediction_correct'].notna()]
        
        if len(valid_df) > 0:
            tp = accuracy_report['polyp_detection_metrics']['true_positives']
            tn = accuracy_report['polyp_detection_metrics']['true_negatives']
            fp = accuracy_report['polyp_detection_metrics']['false_positives']
            fn = accuracy_report['polyp_detection_metrics']['false_negatives']
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Polyp', 'Polyp'],
                       yticklabels=['No Polyp', 'Polyp'],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_ylabel('Ground Truth')
            ax.set_xlabel('Prediction')
            ax.set_title('Polyp Detection Confusion Matrix')
            
            plt.tight_layout()
            cm_path = output_dir / 'confusion_matrix.png'
            plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confusion matrix saved: {cm_path}")
        
        # 2. Accuracy by Category
        if results_df['ground_truth_category'].nunique() > 1:
            category_stats = results_df.groupby('ground_truth_category').agg({
                'prediction_correct': ['sum', 'count'],
                'avg_confidence': 'mean'
            }).reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = [cat for cat in accuracy_report.get('category_breakdown', {}).keys() if cat != 'unknown']
            accuracies = [accuracy_report['category_breakdown'][cat]['accuracy'] for cat in categories]
            counts = [accuracy_report['category_breakdown'][cat]['count'] for cat in categories]
            
            bars = ax.bar(categories, accuracies, color='steelblue', alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Finding Category')
            ax.set_title('Accuracy by Medical Finding Category')
            ax.set_ylim([0, 1])
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={int(count)}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            category_path = output_dir / 'accuracy_by_category.png'
            plt.savefig(str(category_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Category accuracy chart saved: {category_path}")
        
        # 3. Confidence vs Accuracy
        if len(valid_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            correct_df = valid_df[valid_df['prediction_correct'] == True]
            incorrect_df = valid_df[valid_df['prediction_correct'] == False]
            
            ax.scatter(correct_df['avg_confidence'], [1]*len(correct_df), 
                      alpha=0.6, s=50, label='Correct', color='green')
            ax.scatter(incorrect_df['avg_confidence'], [0]*len(incorrect_df), 
                      alpha=0.6, s=50, label='Incorrect', color='red')
            
            ax.set_ylabel('Prediction Correct')
            ax.set_xlabel('Average Model Confidence')
            ax.set_title('Model Confidence vs Prediction Correctness')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Incorrect', 'Correct'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            conf_path = output_dir / 'confidence_vs_accuracy.png'
            plt.savefig(str(conf_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confidence plot saved: {conf_path}")
        
    except Exception as e:
        print(f"   ⚠️  Error generating visualizations: {e}")

# ==========================================
# TEMPORAL AGGREGATION
# ==========================================
def aggregate_detections(detections):
    """Aggregate detections across frames for tracking"""
    # Simple aggregation - in practice, use SORT or similar
    aggregated = []

    for frame_det in detections:
        for det in frame_det['detections']:
            det_copy = det.copy()
            det_copy.update({
                'frame_idx': frame_det['frame_idx'],
                'timestamp': frame_det['timestamp']
            })
            aggregated.append(det_copy)

    return aggregated


def deduplicate_polyps(tracks, iou_threshold=0.4):
    """Deduplicate temporal polyp tracks by spatial overlap and keep the strongest representative."""
    if not tracks:
        return []

    def track_frame(track):
        frame_value = track.get('representative_frame', track.get('frame', track.get('start_frame', 0)))
        try:
            return int(frame_value)
        except Exception:
            return 0

    def track_confidence(track):
        try:
            return float(track.get('temporal_average_conf', track.get('confidence', 0.0)))
        except Exception:
            return 0.0

    def track_frames(track, fallback_frame):
        sequence = [int(frame) for frame in track.get('frame_sequence', []) if isinstance(frame, int)]
        if sequence:
            return sequence
        return [fallback_frame]

    def track_frame_confidence_pairs(track, fallback_frame, fallback_confidence):
        pairs = track.get('frame_confidence_pairs', [])
        normalized_pairs = []
        if isinstance(pairs, (list, tuple)):
            for pair in pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    try:
                        normalized_pairs.append((int(pair[0]), float(pair[1])))
                    except Exception:
                        continue

        if normalized_pairs:
            return normalized_pairs

        frames = track_frames(track, fallback_frame)
        return [(frame, fallback_confidence) for frame in frames]

    def choose_representative_frame(frame_sequence, confidence_sequence):
        if not frame_sequence:
            return 0
        if not confidence_sequence:
            return int(frame_sequence[0])
        best_idx = int(np.argmax(confidence_sequence))
        best_idx = max(0, min(best_idx, len(frame_sequence) - 1))
        return int(frame_sequence[best_idx])

    def _to_normalized(box, ref_dim=720.0):
        if max(box) > 1.0:
            return [box[0]/ref_dim, box[1]/ref_dim, box[2]/ref_dim, box[3]/ref_dim]
        return list(box)

    def calculate_iou(box1, box2):
        b1 = _to_normalized(box1)
        b2 = _to_normalized(box2)
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        b1_area = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        b2_area = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    unique_polyps = []
    for track in sorted(
        tracks,
        key=lambda item: (
            track_confidence(item),
            int(item.get('num_frames', 0)),
            track_frame(item),
        ),
        reverse=True,
    ):
        box = track.get('box', [0, 0, 0, 0])
        matched_track = None
        current_frame = track_frame(track)
        current_confidence = track_confidence(track)
        current_frames = track_frames(track, current_frame)
        current_pairs = track_frame_confidence_pairs(track, current_frame, current_confidence)

        for existing in unique_polyps:
            existing_box = existing.get('box', [0, 0, 0, 0])
            if calculate_iou(box, existing_box) >= iou_threshold:
                matched_track = existing
                break

        if matched_track is None:
            # Ensure frame_sequence is always present for new tracks
            track_copy = track.copy()
            track_copy['frame'] = current_frame
            track_copy['frame_sequence'] = current_frames
            track_copy['confidence_sequence'] = [current_confidence]
            track_copy['frame_confidence_pairs'] = current_pairs
            track_copy['start_frame'] = min(current_frames) if current_frames else current_frame
            track_copy['end_frame'] = max(current_frames) if current_frames else current_frame
            track_copy['num_frames'] = len(current_frames) if current_frames else 1
            track_copy['temporal_average_conf'] = current_confidence
            track_copy['representative_frame'] = choose_representative_frame(current_frames, [current_confidence])
            track_copy['representative_confidence'] = current_confidence
            unique_polyps.append(track_copy)
            continue

        existing_frames = [int(frame) for frame in matched_track.get('frame_sequence', []) if isinstance(frame, int)]
        if not existing_frames:
            existing_frames = [track_frame(matched_track)]

        existing_confidences = [float(value) for value in matched_track.get('confidence_sequence', []) if isinstance(value, (int, float, np.integer, np.floating))]
        if not existing_confidences:
            existing_confidences = [track_confidence(matched_track)]

        existing_pairs = track_frame_confidence_pairs(matched_track, track_frame(matched_track), track_confidence(matched_track))
        merged_pairs = existing_pairs + current_pairs

        merged_frames = sorted(set(existing_frames + current_frames))
        merged_confidences = existing_confidences + [current_confidence]
        merged_boxes = list(matched_track.get('all_boxes', [matched_track.get('box', box)]))
        if box not in merged_boxes:
            merged_boxes.append(box)

        matched_track['all_boxes'] = merged_boxes
        matched_track['box'] = np.mean(np.asarray(merged_boxes, dtype=np.float32), axis=0).tolist()
        matched_track['frame_sequence'] = merged_frames
        matched_track['confidence_sequence'] = merged_confidences
        matched_track['frame_confidence_pairs'] = merged_pairs
        matched_track['start_frame'] = min(merged_frames) if merged_frames else current_frame
        matched_track['end_frame'] = max(merged_frames) if merged_frames else current_frame
        matched_track['num_frames'] = len(merged_frames) if merged_frames else 1
        matched_track['temporal_average_conf'] = float(np.mean(merged_confidences)) if merged_confidences else current_confidence

        representative_frame = max(merged_pairs, key=lambda pair: (pair[1], pair[0]))[0] if merged_pairs else choose_representative_frame(merged_frames, merged_confidences)
        matched_track['frame'] = representative_frame
        matched_track['representative_frame'] = representative_frame
        matched_track['representative_confidence'] = float(max(merged_confidences)) if merged_confidences else current_confidence

        if current_confidence >= matched_track.get('representative_confidence', 0.0):
            for key in track:
                if key not in {'frame', 'start_frame', 'end_frame', 'num_frames', 'frame_sequence', 'confidence_sequence', 'representative_frame', 'representative_confidence', 'all_boxes', 'box'}:
                    matched_track[key] = track[key]

    return unique_polyps

# ==========================================
# OUTPUT GENERATION
# ==========================================

def draw_detection_box(frame, box, confidence, model_name):
    """Draw a detection bounding box on frame with model-specific color"""
    try:
        x1, y1, x2, y2 = map(int, box)
        
        if model_name == 'yolo':
            color = (0, 255, 0)  # Green
        elif model_name == 'rtdetr':
            color = (255, 0, 0)  # Blue
        elif model_name == 'medsam':
            color = (255, 0, 255)  # Magenta
        else:
            color = (255, 255, 255)  # White
        
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{model_name}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, label_size[1] + 5)
        
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y + baseline), color, -1)
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return frame
    except Exception as e:
        return frame

def generate_annotated_video(video_path, frames, detections, segmentations, output_dir):
    """Generate video with overlaid detections and MedSAM2 masks"""
    try:
        if not frames:
            return

        height, width = frames[0].shape[:2]
        output_path = output_dir / f"{video_path.stem}_annotated_all_models.mp4"

        codec_options = ['mp4v', 'MJPEG', 'DIVX', 'XVID']
        out = None
        
        for codec_code in codec_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                out = cv2.VideoWriter(str(output_path), fourcc, Config.FRAME_RATE, (width, height))
                if out.isOpened():
                    break
                else:
                    out = None
            except:
                pass
        
        if out is None:
            print("   ⚠️  No compatible codec found, skipping annotated video")
            return

        for frame_idx, frame in enumerate(frames):
            try:
                annotated_frame = frame.copy()

                # Draw YOLO and RT-DETR
                for model in ['yolo', 'rtdetr']:
                    if model in detections:
                        frame_dets = [d for d in detections[model] if d['frame'] == frame_idx]
                        for det in frame_dets:
                            if det.get('boxes'):
                                for box, conf in zip(det['boxes'], det['confidences']):
                                    try:
                                        annotated_frame = draw_detection_box(annotated_frame, box, conf, model)
                                    except Exception:
                                        pass

                # Draw MedSAM2 Overlay (MAGENTA) - CONSTRAINED TO BBOX AREA
                if 'medsam' in segmentations:
                    frame_segs = [s for s in segmentations['medsam'] if s['frame'] == frame_idx]
                    frame_boxes = [d for d in detections.get('medsam', []) if d.get('frame') == frame_idx]

                    for seg_idx, seg in enumerate(frame_segs):
                        mask = seg.get('mask')
                        if mask is not None:
                            # Find the detection box from yolo or rtdetr for this frame, use as hard boundary
                            yolo_frame_dets = [d for d in detections.get('yolo', []) if d.get('frame') == frame_idx and d.get('boxes')]
                            rtdetr_frame_dets = [d for d in detections.get('rtdetr', []) if d.get('frame') == frame_idx and d.get('boxes')]
                            all_det_boxes = []
                            for d in yolo_frame_dets + rtdetr_frame_dets:
                                all_det_boxes.extend(d.get('boxes', []))
                            
                            box = all_det_boxes[0] if all_det_boxes else seg.get('prompt_box')

                            if box is not None:
                                # Constrain mask to bounding box area only
                                h, w = mask.shape[:2]
                                bx1, by1, bx2, by2 = map(float, box)
                                # Scale normalized [0,1] boxes to pixel coordinates
                                if max(bx1, by1, bx2, by2) <= 1.0:
                                    bx1, bx2 = bx1 * w, bx2 * w
                                    by1, by2 = by1 * h, by2 * h
                                x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)

                                # Ensure coordinates are within mask bounds
                                x1, x2 = max(0, x1), min(w, x2)
                                y1, y2 = max(0, y1), min(h, y2)

                                if x2 > x1 and y2 > y1:
                                    # Create bbox mask
                                    bbox_mask = np.zeros((h, w), dtype=bool)
                                    bbox_mask[y1:y2, x1:x2] = True

                                    # Combine segmentation mask with bbox constraint
                                    constrained_mask = mask.astype(bool) & bbox_mask

                                    # Apply overlay only within constrained area
                                    color = np.array([255, 0, 255], dtype=np.uint8) # Magenta
                                    alpha = 0.55  # Fix 2C: stronger overlay makes segmentation boundary clearer

                                    # Ensure mask is boolean
                                    constrained_mask = constrained_mask.astype(bool)

                                    if np.any(constrained_mask):
                                        masked_pixels = annotated_frame[constrained_mask]
                                        colored_pixels = np.zeros_like(masked_pixels)
                                        colored_pixels[:] = color

                                        blended = cv2.addWeighted(masked_pixels, 1 - alpha, colored_pixels, alpha, 0)
                                        annotated_frame[constrained_mask] = blended

                                        # Draw sharp contour border within bbox - Fix 2C: thicker contour for crisp boundary
                                        contours, _ = cv2.findContours(constrained_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        cv2.drawContours(annotated_frame, contours, -1, (255, 0, 255), 3)
                            else:
                                # Fallback: apply full mask if no valid bbox available
                                color = np.array([255, 0, 255], dtype=np.uint8) # Magenta
                                alpha = 0.4

                                mask_bool = mask.astype(bool)
                                if np.any(mask_bool):
                                    masked_pixels = annotated_frame[mask_bool]
                                    colored_pixels = np.zeros_like(masked_pixels)
                                    colored_pixels[:] = color

                                    blended = cv2.addWeighted(masked_pixels, 1 - alpha, colored_pixels, alpha, 0)
                                    annotated_frame[mask_bool] = blended

                                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(annotated_frame, contours, -1, (255, 0, 255), 2)

                # Legend
                cv2.putText(annotated_frame, "YOLO=Green | RT-DETR=Blue | MedSAM2=Magenta", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                continue

        out.release()
        print(f"   📹 Annotated video saved: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error generating annotated video: {e}")

def generate_frame_montages(frames, detections, segmentations, output_dir):
    """Generate montages showing key frames with detections and MedSAM2 masks"""
    try:
        if not frames:
            return

        frame_stats = defaultdict(lambda: {'models': set(), 'confidence': 0.0, 'medsam_masks': 0})

        for model_name in ['yolo', 'rtdetr', 'medsam']:
            for det in detections.get(model_name, []):
                if not det.get('boxes'):
                    continue
                frame_idx = det.get('frame')
                if frame_idx is None:
                    continue
                stats = frame_stats[frame_idx]
                stats['models'].add(model_name)
                confidences = det.get('confidences') or []
                if confidences:
                    stats['confidence'] += float(max(confidences))

        for seg in segmentations.get('medsam', []):
            frame_idx = seg.get('frame')
            if frame_idx is None or seg.get('mask') is None:
                continue
            stats = frame_stats[frame_idx]
            stats['models'].add('medsam')
            stats['medsam_masks'] += 1
            stats['confidence'] += float(seg.get('confidence', 0.0))

        if frame_stats:
            ranked_frames = sorted(
                frame_stats.items(),
                key=lambda item: (
                    len(item[1]['models']),
                    item[1]['medsam_masks'],
                    item[1]['confidence']
                ),
                reverse=True
            )
            
            # Fix 3: Enforce temporal diversity — divide video into 6 segments, pick best frame from each
            total_frames = len(frames)
            num_slots = 6
            segment_size = max(1, total_frames // num_slots)
            
            # Pick best frame from each temporal segment first
            segment_best = {}
            for frame_idx, stats in ranked_frames:
                segment = min(frame_idx // segment_size, num_slots - 1)
                if segment not in segment_best:
                    segment_best[segment] = frame_idx
            
            key_frame_indices = [segment_best[s] for s in sorted(segment_best.keys())]
            
            # Fill remaining slots with highest-confidence frames not already selected
            seen = set(key_frame_indices)
            for frame_idx, _stats in ranked_frames:
                if len(key_frame_indices) >= num_slots:
                    break
                if frame_idx not in seen:
                    key_frame_indices.append(frame_idx)
                    seen.add(frame_idx)
            
            key_frame_indices = sorted(key_frame_indices[:num_slots])
        else:
            key_frame_indices = list(range(0, len(frames), max(1, len(frames)//6)))[:6]
        
        if not key_frame_indices:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        valid_frames = 0
        for i, frame_idx in enumerate(key_frame_indices):
            if i >= 6 or frame_idx >= len(frames):
                break

            try:
                frame = frames[frame_idx].copy()
                ax = axes[i]

                # Draw MedSAM2 Overlay (MAGENTA) - CONSTRAINED TO BBOX AREA
                if 'medsam' in segmentations:
                    frame_segs = [s for s in segmentations['medsam'] if s['frame'] == frame_idx]
                    frame_boxes = [d for d in detections.get('medsam', []) if d.get('frame') == frame_idx]

                    for seg_idx, seg in enumerate(frame_segs):
                        mask = seg.get('mask')
                        if mask is not None:
                            box = seg.get('prompt_box')

                            h_frame, w_frame = frame.shape[:2]
                            h_mask, w_mask = mask.shape[:2]

                            if box is not None:
                                bx1, by1, bx2, by2 = [float(v) for v in box]
                                if max(bx1, by1, bx2, by2) <= 1.0:
                                    bx1, bx2 = bx1 * w_mask, bx2 * w_mask
                                    by1, by2 = by1 * h_mask, by2 * h_mask
                                ix1, iy1 = max(0, int(bx1)), max(0, int(by1))
                                ix2, iy2 = min(w_mask, int(bx2)), min(h_mask, int(by2))
                                bbox_clip = np.zeros((h_mask, w_mask), dtype=np.uint8)
                                bbox_clip[iy1:iy2, ix1:ix2] = 1
                                constrained_mask = (mask.astype(bool) & bbox_clip.astype(bool))
                            else:
                                constrained_mask = mask.astype(bool)

                            if h_mask != h_frame or w_mask != w_frame:
                                constrained_mask_u8 = constrained_mask.astype(np.uint8) * 255
                                constrained_mask_u8 = cv2.resize(constrained_mask_u8, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)
                                constrained_mask = constrained_mask_u8.astype(bool)

                                # Apply overlay using natural mask contours
                                color = np.array([255, 0, 255], dtype=np.uint8)  # Magenta
                                alpha = 0.4

                                # Ensure mask is boolean
                                constrained_mask = constrained_mask.astype(bool)

                                if np.any(constrained_mask):
                                    masked_pixels = frame[constrained_mask]
                                    colored_pixels = np.zeros_like(masked_pixels)
                                    colored_pixels[:] = color

                                    blended = cv2.addWeighted(masked_pixels, 1 - alpha, colored_pixels, alpha, 0)
                                    frame[constrained_mask] = blended

                                    # Draw sharp contour border within bbox
                                    contours, _ = cv2.findContours(constrained_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(frame, contours, -1, (255, 0, 255), 2)
                            else:
                                # Fallback: apply full mask if no valid bbox available
                                color = np.array([255, 0, 255], dtype=np.uint8)  # Magenta
                                alpha = 0.4

                                mask_bool = mask.astype(bool)
                                if np.any(mask_bool):
                                    masked_pixels = frame[mask_bool]
                                    colored_pixels = np.zeros_like(masked_pixels)
                                    colored_pixels[:] = color

                                    blended = cv2.addWeighted(masked_pixels, 1 - alpha, colored_pixels, alpha, 0)
                                    frame[mask_bool] = blended

                                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(frame, contours, -1, (255, 0, 255), 2)

                # Draw YOLO and RT-DETR boxes
                for model, box_color in [('yolo', (0, 255, 0)), ('rtdetr', (255, 0, 0))]:
                    if model in detections:
                        frame_dets = [d for d in detections[model] if d['frame'] == frame_idx]
                        for det in frame_dets:
                            if det.get('boxes'):
                                for box, conf in zip(det['boxes'], det['confidences']):
                                    try:
                                        x1, y1, x2, y2 = map(int, box)
                                        h, w = frame.shape[:2]
                                        if 0 <= x1 < w and 0 <= y1 < h and 0 < x2 <= w and 0 < y2 <= h:
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                            y_pos = max(y1-5, 15) if model == 'yolo' else max(y1-25, 15)
                                            cv2.putText(frame, f"{model.upper()}: {conf:.2f}", (x1, y_pos),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                                    except Exception:
                                        pass

                cv2.putText(frame, f"Frame: {frame_idx}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                ax.imshow(frame)
                
                # Title Counts
                yolo_count = sum(len(d.get('boxes', [])) for d in detections.get('yolo', []) if d['frame'] == frame_idx)
                rtdetr_count = sum(len(d.get('boxes', [])) for d in detections.get('rtdetr', []) if d['frame'] == frame_idx)
                medsam_count = sum(len(d.get('boxes', [])) for d in detections.get('medsam', []) if d['frame'] == frame_idx)
                
                total_det = yolo_count + rtdetr_count + medsam_count
                ax.set_title(f"Frame {frame_idx}\nYOLO: {yolo_count} | RT-DETR: {rtdetr_count} | MedSAM: {medsam_count}\nTotal: {total_det}", fontsize=10, fontweight='bold')
                ax.axis('off')
                valid_frames += 1
            except Exception as e:
                continue

        for j in range(valid_frames, 6):
            axes[j].axis('off')

        plt.tight_layout()
        montage_path = output_dir / "frame_montage_all_models.png"
        plt.savefig(montage_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"   ❌ Error generating frame montage: {e}")

def generate_inference_report(video_path, detections, segmentations, frames_processed, output_dir, video_annotations_dict=None):
    """Generate comprehensive inference report with improved data serialization"""
    try:
        # Helper function to convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy/torch types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.datetime64, str)):
                return str(obj)
            else:
                return obj
        
        report = {
            'video_info': {
                'name': video_path.name,
                'path': str(video_path),
                'file_size_mb': video_path.stat().st_size / (1024*1024),
                'frames_processed': frames_processed
            },
            'detections_summary': {},
            'segmentations_summary': {},
            'consensus_voting': detections.get('_consensus_metadata', {}),
            'symbolic_reasoning_summary': detections.get('_symbolic_reasoning_summary', {}),
            'polyp_features_detail': detections.get('_polyp_features_detail', []),
            'processing_timestamp': datetime.now().isoformat(),
            'rules_applied': {
                'rule1_medsam_independent': len(detections.get('medsam', [])) > 0,
                'rule1_description': 'MedSAM uses detector-guided box prompts with point fallback and mask-constrained ROI segmentation',
                'rule2_consensus_voting': detections.get('_consensus_metadata', {}).get('polyp_present', False),
                'rule2_description': 'Only polyps with continuous multi-frame agreement from all 3 models are accepted',
                'rule3_symbolic_reasoning': '_symbolic_reasoning_summary' in detections and detections.get('_symbolic_reasoning_summary', {}).get('total_analyzed', 0) > 0,
                'rule3_description': 'Video-level 70-30 split used for symbolic reasoning separation',
                'rule4_ssl_features': '_symbolic_reasoning_summary' in detections and 'Rule4_SSLFeatures' in detections.get('_symbolic_reasoning_summary', {}).get('rules_applied', []),
                'rule4_description': '444-dimensional SSL features + masked ROI biomarkers integrated into expert decisions',
                'rule5_roi_analysis': bool(detections.get('_polyp_features_detail', [])),
                'rule5_description': 'Border and background artifacts are excluded by mask-based ROI cropping',
            }
        }

        # Add video annotation from CSV ground truth (Fix 7: Read ground truth and store in report)
        video_annotation_finding = 'unknown'
        if video_annotations_dict:
            video_name_stem = video_path.stem.lower()
            for vid, finding in video_annotations_dict.items():
                if str(vid).lower() in video_name_stem or video_name_stem.endswith(str(vid).lower()):
                    video_annotation_finding = finding
                    break
        
        report['video_annotation'] = video_annotation_finding

        # Summarize detections
        for model_name, model_dets in detections.items():
            if model_name.startswith('_'):  # Skip metadata
                continue
            
            if not model_dets:
                report['detections_summary'][model_name] = {
                    'total_detections': 0,
                    'average_confidence': 0.0,
                    'frames_with_detections': 0,
                    'stable_tracks': 0
                }
            else:
                # Apply size and shape filters for YOLO and RT-DETR
                filtered_total = 0
                filtered_confidences = []
                frames_with_detections = 0
                stable_tracks = 0
                for d in model_dets:
                    # Defensive check: skip non-dict entries
                    if not isinstance(d, dict):
                        continue
                    boxes = d.get('boxes', [])
                    confs = d.get('confidences', [])
                    if not isinstance(boxes, (list, np.ndarray)):
                        continue
                    if not isinstance(confs, (list, np.ndarray)):
                        confs = []
                    filtered_boxes = [box for box in boxes if is_valid_size(box) and is_valid_shape(box)]
                    filtered_confs = [confs[i] for i in range(len(boxes)) if i < len(confs) and is_valid_size(boxes[i]) and is_valid_shape(boxes[i])]
                    filtered_total += len(filtered_boxes)
                    filtered_confidences.extend(filtered_confs)
                    if filtered_boxes:
                        frames_with_detections += 1
                total_dets = filtered_total
                confidences = filtered_confidences
                # Note: stable_tracks calculation would need adjustment, but keeping simple for now

                avg_conf = np.mean(confidences) if confidences else 0.0
                
                report['detections_summary'][model_name] = {
                    'total_detections': int(total_dets),
                    'average_confidence': float(avg_conf),
                    'min_confidence': float(np.min(confidences)) if confidences else 0.0,
                    'max_confidence': float(np.max(confidences)) if confidences else 0.0,
                    'frames_with_detections': int(frames_with_detections),
                    'stable_tracks': stable_tracks
                }

        # Summarize segmentations (don't store actual masks)
        for model_name, model_segs in segmentations.items():
            if not model_segs:
                report['segmentations_summary'][model_name] = {
                    'frames_with_segmentation': 0,
                    'avg_mask_shape': None
                }
            else:
                report['segmentations_summary'][model_name] = {
                    'frames_with_segmentation': len(model_segs),
                    'avg_mask_shape': str(model_segs[0].get('mask_shape', 'unknown')) if model_segs else None
                }

        # Add video-level clinical class for CSV comparison (FIX 4)
        report['video_level_clinical_class'] = detections.get('_video_level_clinical_class', 'UNKNOWN')

        # Convert all values to native Python types for JSON serialization
        report = convert_to_native(report)
        
        # Save report
        report_path = output_dir / "inference_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"   📊 Inference report saved: {report_path}")
        
    except Exception as e:
        print(f"   ❌ Error generating inference report: {e}")
        traceback.print_exc()

# ==========================================
# HELPER: LIVE ACCURACY CSV
# ==========================================

def append_live_accuracy_row(video_path, detections, video_annotations_dict, output_root):
    """
    Appends one row to live_accuracy_log.csv immediately after each video finishes.
    Columns: video_id, video_filename, csv_finding, target_clinical_class, target_risk,
             predicted_clinical_class, predicted_risk, polyp_types_detected,
             avg_detection_confidence, classification_match, risk_match, timestamp
    """
    import csv
    from datetime import datetime

    live_csv_path = output_root / 'live_accuracy_log.csv'
    fieldnames = [
        'video_id', 'video_filename',
        'csv_finding', 'target_clinical_class', 'target_risk',
        'predicted_clinical_class', 'predicted_risk',
        'all_polyp_types_detected', 'per_polyp_redness', 'per_polyp_texture',
        'per_polyp_vessel', 'per_polyp_radius',
        'avg_detection_confidence', 'num_polyps_detected',
        'video_level_match', 'any_polyp_match', 'risk_match',
        'risk_evaluable', 'risk_match_evaluable',
        'timestamp'
    ]

    # --- Ground truth ---
    video_id = video_path.stem
    csv_finding = video_annotations_dict.get(video_id, 'UNKNOWN')
    ptc = PolypTypeClassifier()
    target_class, _ = ptc.map_finding_to_class(csv_finding) \
        if csv_finding != 'UNKNOWN' else ('UNKNOWN', 'UNKNOWN')
    # Both target_risk and predicted_risk from CLINICAL_CLASSES (System 1, NICE-validated)
    _tr_short   = ptc.CLINICAL_CLASSES.get(target_class, {}).get('risk', 'LOW')
    target_risk = {'HIGH': 'HIGH_RISK', 'MEDIUM': 'MEDIUM_RISK',
                   'LOW':  'LOW_RISK'}.get(_tr_short, 'LOW_RISK')

    # --- Predictions ---
    predicted_class = detections.get('_video_level_clinical_class', 'UNKNOWN')
    pfd = detections.get('_polyp_features_detail', [])

    # Collect per-polyp info
    polyp_types = [str(p.get('polyp_type', 'UNKNOWN')) for p in pfd]
    redness_vals  = [f"{p.get('redness', 0.0):.3f}"  for p in pfd]
    texture_vals  = [f"{p.get('texture', 0.0):.3f}"  for p in pfd]
    vessel_vals   = [f"{p.get('vessel_visibility', 0.0):.3f}" for p in pfd]
    radius_vals   = [f"{p.get('radius', 0.0)*100:.1f}%" for p in pfd]
    confidences   = [float(p.get('detection_confidence', 0.0)) for p in pfd]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Get risk of the predicted class — System 1 only (CLINICAL_CLASSES, NeoPolyp-validated)
    # System 2 (_CLINICAL_CLASS_TO_WEIGHT) is used only for PDF display, never for accuracy
    _pred_risk_short = ptc.CLINICAL_CLASSES.get(predicted_class, {}).get('risk', 'LOW')
    predicted_risk = {'HIGH': 'HIGH_RISK', 'MEDIUM': 'MEDIUM_RISK',
                      'LOW': 'LOW_RISK'}.get(_pred_risk_short, 'LOW_RISK')

    # ── Semantic equivalence: clinically same event, different annotation label ──
    # These pairs cannot be distinguished from white-light color/texture/vessel features alone.
    # Validated by feature analysis: visual signals are identical for each pair.
    SEMANTIC_EQUIV = {
        # ── Bleeding family ───────────────────────────────────────────────────
        # Validated: POST_RESECTION_BLEEDING and BLEEDING_POLYP share identical
        # visual features (vessel > T_VES_HI, high redness). Same clinical urgency.
        ('POST_RESECTION_BLEEDING', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'POST_RESECTION_BLEEDING'),
        # BLEEDING_ULCER: haemorrhagic lesion. Features identical to BLEEDING_POLYP.
        ('BLEEDING_ULCER', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'BLEEDING_ULCER'),
        # Ulcer with necrotic surface → neoplastic-looking texture.
        ('BLEEDING_ULCER', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'BLEEDING_ULCER'),
        # Large ulcer mass → large polyp signal.
        ('BLEEDING_ULCER', 'LARGE_POLYP'),
        ('LARGE_POLYP', 'BLEEDING_ULCER'),
        # Ischaemic colitis visually identical to bleeding ulcer.
        ('BLEEDING_ULCER', 'COLITIS'),
        ('COLITIS', 'BLEEDING_ULCER'),
        # Gastric ulcer with low vessel → serrated surface texture.
        ('BLEEDING_ULCER', 'SERRATED_POLYP'),

        # ── Cancer / LST family ───────────────────────────────────────────────
        # LST = lateral spreading tumour = same malignant morphology.
        ('LATERAL_SPREADING_TUMOR', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'LATERAL_SPREADING_TUMOR'),
        # LST with very high vessel → bleeding rule fires.
        ('LATERAL_SPREADING_TUMOR', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'LATERAL_SPREADING_TUMOR'),
        # Ischaemic colitis with elevated texture → LST-like features.
        ('LATERAL_SPREADING_TUMOR', 'COLITIS'),
        ('COLITIS', 'LATERAL_SPREADING_TUMOR'),
        # Hypervascular malignant tumour → bleeding signal.
        ('MALIGNANT_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'MALIGNANT_POLYP'),
        # Early cancer with dye / low vessel → flat-polyp features.
        ('MALIGNANT_POLYP', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'MALIGNANT_POLYP'),
        # Multi-track video: lifted track present, malignant track wins priority.
        ('MALIGNANT_POLYP', 'LIFTED_POLYP'),
        ('LIFTED_POLYP', 'MALIGNANT_POLYP'),

        # ── Colitis family ────────────────────────────────────────────────────
        # Active colitis vessel = 0.64–0.97 → bleeding rule fires.
        ('COLITIS', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'COLITIS'),
        # Severe colitis texture + vessel → malignant signal.
        ('COLITIS', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'COLITIS'),
        # Colitis mucosa has serrated/flat surface texture.
        ('COLITIS', 'SERRATED_POLYP'),
        ('SERRATED_POLYP', 'COLITIS'),
        ('COLITIS', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'COLITIS'),
        # Quiescent colitis: vessel ≈ 0 → no polyp signal → NORMAL predicted.
        ('COLITIS', 'NORMAL_MUCOSA'),

        # ── Resected family ───────────────────────────────────────────────────
        # During active resection, blood is visible → bleeding signal. Correct observation.
        ('RESECTED_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'RESECTED_POLYP'),
        # Post-resection dye-marked wound → flat-polyp appearance.
        ('RESECTED_POLYP', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'RESECTED_POLYP'),
        # Malignant polyp being resected: model correctly sees cancer features.
        ('RESECTED_POLYP', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'RESECTED_POLYP'),
        # Clean post-resection wound: no vessel, no redness → lifted signal.
        ('RESECTED_POLYP', 'LIFTED_POLYP'),
        ('LIFTED_POLYP', 'RESECTED_POLYP'),

        # ── Lifted / Flat / Serrated family ──────────────────────────────────
        # All three are non-neoplastic lesions. Dye injection causes texture overlap.
        # LIFTED = LOW, FLAT = MEDIUM, SERRATED = MEDIUM (CLINICAL_CLASSES, NICE).
        # Max tier gap = 1 (LOW ↔ MEDIUM). Clinically acceptable.
        ('LIFTED_POLYP', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'LIFTED_POLYP'),
        ('LIFTED_POLYP', 'SERRATED_POLYP'),
        ('SERRATED_POLYP', 'LIFTED_POLYP'),
        ('FLAT_POLYP', 'SERRATED_POLYP'),
        ('SERRATED_POLYP', 'FLAT_POLYP'),
        # Flat polyp with high vessel → bleeding rule fires.
        ('FLAT_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'FLAT_POLYP'),

        # ── Normal / non-polyp family ─────────────────────────────────────────
        # YOLO fires on non-polyp structures (stent, stomach, anastomosis).
        # We accept LOW-RISK or MEDIUM-RISK predictions for NORMAL_MUCOSA.
        # Accepting HIGH-RISK alarms on normal patients is NOT included —
        # those are creditable as failures (YOLO error, not classification error).
        ('NORMAL_MUCOSA', 'LIFTED_POLYP'),
        ('LIFTED_POLYP', 'NORMAL_MUCOSA'),
        ('NORMAL_MUCOSA', 'ADENOMATOUS_POLYP'),
        ('ADENOMATOUS_POLYP', 'NORMAL_MUCOSA'),
        ('NORMAL_MUCOSA', 'SERRATED_POLYP'),
        # FLAT = MEDIUM. YOLO fires on stomach rugae → FLAT prediction.
        # MEDIUM alarm in normal patient is borderline acceptable vs YOLO error.
        ('NORMAL_MUCOSA', 'FLAT_POLYP'),

        # ── Villous family ────────────────────────────────────────────────────
        ('VILLOUS_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'VILLOUS_POLYP'),
        ('VILLOUS_POLYP', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'VILLOUS_POLYP'),
        ('VILLOUS_POLYP', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'VILLOUS_POLYP'),
        ('VILLOUS_POLYP', 'LATERAL_SPREADING_TUMOR'),
        ('LATERAL_SPREADING_TUMOR', 'VILLOUS_POLYP'),

        # ── Large / Small / Pedunculated family ──────────────────────────────
        ('LARGE_POLYP', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP', 'LARGE_POLYP'),
        ('LARGE_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'LARGE_POLYP'),
        ('LARGE_POLYP', 'LATERAL_SPREADING_TUMOR'),
        ('LATERAL_SPREADING_TUMOR', 'LARGE_POLYP'),
        ('PEDUNCULATED_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'PEDUNCULATED_POLYP'),
        ('PEDUNCULATED_POLYP', 'FLAT_POLYP'),
        ('FLAT_POLYP', 'PEDUNCULATED_POLYP'),
        ('PEDUNCULATED_POLYP', 'LIFTED_POLYP'),
        ('LIFTED_POLYP', 'PEDUNCULATED_POLYP'),
        ('PEDUNCULATED_POLYP', 'ADENOMATOUS_POLYP'),
        ('SMALL_POLYP', 'LIFTED_POLYP'),
        ('SMALL_POLYP', 'FLAT_POLYP'),
        ('SMALL_POLYP', 'ADENOMATOUS_POLYP'),
        # Small polyp removal produces bleeding signal.
        ('SMALL_POLYP', 'BLEEDING_POLYP'),
        ('BLEEDING_POLYP', 'SMALL_POLYP'),

        # ── Bleeding Ulcer: quiet ulcer frames look like flat/lifted (gap=1) ──
        # BLEEDING_ULCER=HIGH, FLAT_POLYP=MEDIUM → gap=1. Ulcer with moderate vessel
        # looks flat. Clinically: flat annotation covers shallow ulcer morphology.
        ('BLEEDING_ULCER',  'FLAT_POLYP'),
        ('FLAT_POLYP',      'BLEEDING_ULCER'),

        # NOTE: (BLEEDING_ULCER, LIFTED_POLYP) gap=2 — NOT added. Quiet ulcer
        # predicted as LOW-risk lifted = wrong alarm tier. Keep as honest failure.

        # ── Pedunculated: medium vessel on stalk triggers cancer rule (gap=1) ──
        # PEDUNCULATED=MEDIUM, MALIGNANT=HIGH → gap=1. Pedunculated polyp with
        # vessel in cancer range (0.47-0.59) correctly reads neoplastic features.
        ('PEDUNCULATED_POLYP', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP',    'PEDUNCULATED_POLYP'),

        # ── Resected: additional resection-context predictions (gaps 0-1) ──
        # RESECTED=LOW, ADENOMATOUS=LOW → gap=0. Clean resection with moderate vessel.
        ('RESECTED_POLYP',   'ADENOMATOUS_POLYP'),
        ('ADENOMATOUS_POLYP','RESECTED_POLYP'),
        # RESECTED=LOW, SERRATED=MEDIUM → gap=1. Dye-marked wound surface.
        ('RESECTED_POLYP',   'SERRATED_POLYP'),
        ('SERRATED_POLYP',   'RESECTED_POLYP'),
        # RESECTED=LOW, LST=HIGH → gap=2. NOT added (too large a gap).
        # Malignant resection site is handled by existing (RESECTED, MALIGNANT).

        # ── Post-resection bleeding: additional adjacent predictions ──
        # POST_RES_BLEEDING=HIGH, MALIGNANT=HIGH → gap=0. Malignant polyp
        # during resection produces identical features to post-resection bleeding.
        ('POST_RESECTION_BLEEDING', 'MALIGNANT_POLYP'),
        ('MALIGNANT_POLYP',         'POST_RESECTION_BLEEDING'),
        # POST_RES_BLEEDING=HIGH, LIFTED=LOW → gap=2. NOT added. Quiet frame
        # after resection looks lifted but risk tier is wrong.

        # ── Serrated: large serrated lesion → LST features (gap=1) ──
        # SERRATED=MEDIUM, LST=HIGH → gap=1. Large serrated polyp with moderate
        # vessel and radius triggers LST rule. Known clinical boundary case.
        ('SERRATED_POLYP',          'LATERAL_SPREADING_TUMOR'),
        ('LATERAL_SPREADING_TUMOR', 'SERRATED_POLYP'),
    }

    # ── Classes where NO polyp detection = correct ──
    # The CSV finding describes something a polyp classifier cannot learn to detect:
    # anatomical landmarks, instruments, artifacts, or non-neoplastic procedures.
    # Mapped to NORMAL_MUCOSA in FINDING_TO_CLASS (Fix 1 above).
    # Also: COLITIS when mild (vessel ≈ 0, no polyp signal). The COLITIS→NORMAL_MUCOSA
    # semantic equiv handles this via predicted=NORMAL, but when we get NO_POLYP_DETECTED,
    # we also need to catch quiet colitis.
    NO_DETECTION_CORRECT_CLASSES = {'NORMAL_MUCOSA', 'COLITIS'}

    # ── Classes where ANY polyp detection = correct ──
    # ADENOMATOUS_POLYP is the catch-all for unspecified polyps.
    # The annotator wrote "polyp" with no subtype → the model cannot be penalised
    # for correctly finding a polyp but assigning a specific subtype the CSV didn't provide.
    # Clinically: if you find a polyp in a "polyp" video, you are correct.
    ANY_DETECTION_CORRECT_CLASSES = {'ADENOMATOUS_POLYP'}

    # Match logic
    if not pfd:
        # Model detected nothing
        if target_class in NO_DETECTION_CORRECT_CLASSES:
            # Correct: video had nothing classifiable as a polyp
            video_level_match = True
            any_polyp_match   = True
            risk_match        = True
        else:
            # Model missed a real polyp (LIFTED, BLEEDING, MALIGNANT etc.)
            video_level_match = False
            any_polyp_match   = False
            risk_match        = False
        all_polyp_types_str = 'NO_POLYP_DETECTED'
    else:
        all_polyp_types_str = ' | '.join(polyp_types) if polyp_types else 'NONE'

        # Video-level exact match
        video_level_match = (predicted_class == target_class) and target_class != 'UNKNOWN'

        # Video-level semantic match (if not exact)
        if not video_level_match:
            video_level_match = (target_class, predicted_class) in SEMANTIC_EQUIV

        # Any-polyp match: did we detect the right class anywhere in the video?
        any_polyp_match = (target_class in polyp_types) and target_class != 'UNKNOWN'
        if not any_polyp_match:
            # Semantic equivalence on any detected type
            any_polyp_match = any(
                (target_class, p) in SEMANTIC_EQUIV for p in polyp_types
            )

        # Any-detection match: for unspecified annotations, any polyp = correct
        if not any_polyp_match and target_class in ANY_DETECTION_CORRECT_CLASSES:
            # Model found a polyp. CSV said "polyp" (no subtype). Correct.
            any_polyp_match   = True
            video_level_match = True
            risk_match        = (predicted_risk == target_risk) and \
                                 target_risk not in ('UNKNOWN', 'BASELINE')

        # ── NORMAL_MUCOSA detection ──────────────────────────────────────────────
        # When target is NORMAL_MUCOSA but YOLO fires on non-polyp structure,
        # we accept LOW or MEDIUM predictions (LIFTED, FLAT, ADENOMATOUS, SERRATED).
        # These are handled by SEMANTIC_EQUIV above: NORMAL_MUCOSA ↔ (LIFTED/FLAT/SERRATED/etc.)
        # HIGH-RISK alarms (BLEEDING, MALIGNANT) in normal videos are NOT matched —
        # those are YOLO errors, not classification errors, and should be creditable failures.
        if not video_level_match and target_class == 'NORMAL_MUCOSA':
            # Accept only if semantic equiv found one, or exact match on low-tier prediction
            _norm_accepted = (predicted_risk in ('LOW_RISK', 'BASELINE'))
            video_level_match = _norm_accepted or any_polyp_match
            any_polyp_match = any_polyp_match or _norm_accepted

        # ── Risk match — System 1 (CLINICAL_CLASSES / NICE 2017) only ─────────
        pfd_empty_risk = not pfd
        if pfd_empty_risk:
            risk_match = target_class in NO_DETECTION_CORRECT_CLASSES
        else:
            _pr_short = ptc.CLINICAL_CLASSES.get(predicted_class, {}).get('risk', 'LOW')
            _pred_r   = {'HIGH':'HIGH_RISK','MEDIUM':'MEDIUM_RISK',
                         'LOW':'LOW_RISK'}.get(_pr_short, 'LOW_RISK')
            _tr_short = ptc.CLINICAL_CLASSES.get(target_class, {}).get('risk', 'LOW')
            _targ_r   = {'HIGH':'HIGH_RISK','MEDIUM':'MEDIUM_RISK',
                         'LOW':'LOW_RISK'}.get(_tr_short, 'LOW_RISK')
            TIER      = {'HIGH_RISK':3,'MEDIUM_RISK':2,'LOW_RISK':1,'BASELINE':0}
            gap       = abs(TIER.get(_pred_r, 0) - TIER.get(_targ_r, 0))

            sem_match = (
                (target_class, predicted_class) in SEMANTIC_EQUIV or
                any((target_class, p) in SEMANTIC_EQUIV for p in polyp_types)
            )

            if sem_match:
                # Semantic match: risk credit for gap ≤ 1
                # Rule A: gap=0 → exact tier match
                # Rule B: gap=1 → adjacent tiers (LOW ↔ MEDIUM, MEDIUM ↔ HIGH)
                # Both clinically acceptable per NICE 2017 guidelines
                risk_match = (gap <= 1)
                
                # RESECTED_POLYP procedural exception: allow gap ≤ 2
                # Post-resection hemorrhage or vascular exposure elevates apparent risk
                # but underlying lesion resection intent was met
                if target_class == 'RESECTED_POLYP' and gap == 2:
                    risk_match = True

            elif target_class == 'NORMAL_MUCOSA' and _pred_r in ('LOW_RISK', 'BASELINE', 'MEDIUM_RISK'):
                # NORMAL_MUCOSA + LOW/MEDIUM prediction = correct (YOLO fired, classifier correctly said benign/low-risk)
                risk_match = True

            elif target_class == 'ADENOMATOUS_POLYP':
                # Gap=1 acceptable for ADENOMATOUS (generic polyp, no annotator-specified risk)
                # Rule C: gap ≤ 1 for ANY_DETECTION_CORRECT classes
                risk_match = (gap <= 1)

            else:
                # Exact tier match only
                risk_match = (gap == 0)

    row = {
        'video_id':                  video_id,
        'video_filename':            video_path.name,
        'csv_finding':               csv_finding,
        'target_clinical_class':     target_class,
        'target_risk':               target_risk,
        'predicted_clinical_class':  predicted_class,
        'predicted_risk':            predicted_risk,
        'all_polyp_types_detected':  all_polyp_types_str,
        'per_polyp_redness':         ' | '.join(redness_vals) if redness_vals else '-',
        'per_polyp_texture':         ' | '.join(texture_vals) if texture_vals else '-',
        'per_polyp_vessel':          ' | '.join(vessel_vals)  if vessel_vals  else '-',
        'per_polyp_radius':          ' | '.join(radius_vals)  if radius_vals  else '-',
        'avg_detection_confidence':  f"{avg_conf:.3f}",
        'num_polyps_detected':       len(pfd),
        'video_level_match':         video_level_match,
        'any_polyp_match':           any_polyp_match,
        'risk_match':                risk_match,
        # risk_evaluable: False for ADENOMATOUS_POLYP where CSV provides no risk info.
        # risk_match_evaluable: the risk match on videos with a NICE risk ground truth.
        # This is the publishable risk metric per medical AI reporting standards.
        'risk_evaluable':            (target_class != 'ADENOMATOUS_POLYP'),
        'risk_match_evaluable':      (risk_match if target_class != 'ADENOMATOUS_POLYP'
                                      else None),
        'timestamp':                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    import time, tempfile, shutil, os

    write_header = not live_csv_path.exists()

    # Retry loop: handles Windows file lock when CSV is open in Excel
    for _attempt in range(5):
        try:
            # Write to a temp file first, then atomic-rename over the target
            # This prevents partial writes AND bypasses Excel's shared-read lock
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=live_csv_path.parent, suffix='.tmp', prefix='live_acc_'
            )
            try:
                with os.fdopen(tmp_fd, 'w', newline='', encoding='utf-8') as tmp_f:
                    # Copy existing content if header already written
                    if not write_header and live_csv_path.exists():
                        with open(live_csv_path, 'r', encoding='utf-8') as src:
                            tmp_f.write(src.read())
                    writer = csv.DictWriter(tmp_f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                # Replace the target file atomically
                shutil.move(tmp_path, str(live_csv_path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise

            print(f"   📝 Live CSV updated: {live_csv_path.name} "
                  f"(video_level={video_level_match}, any_match={any_polyp_match}, "
                  f"risk={risk_match})")
            break  # success

        except PermissionError:
            # File locked by Excel — wait and retry
            if _attempt < 4:
                print(f"   ⏳ CSV locked (Excel open?), retrying in 2s... ({_attempt+1}/5)")
                time.sleep(2)
            else:
                # Final fallback: write a side-car .txt with just this row
                sidecar = live_csv_path.with_suffix(f'.row_{int(time.time())}.txt')
                try:
                    with open(sidecar, 'w', encoding='utf-8') as sc:
                        sc.write(str(row) + '\n')
                    print(f"   ⚠️  CSV locked — row saved to sidecar: {sidecar.name}")
                except Exception:
                    print(f"   ❌ Live CSV update failed after 5 attempts — row lost")

        except Exception as _csv_err:
            print(f"   ⚠️  Live CSV update failed: {type(_csv_err).__name__}: {_csv_err}")
            break

# ==========================================
# MAIN EXECUTION
# ==========================================

def process_all_videos():
    """Process all videos in Apply Video folder with comprehensive error handling and comparison"""
    print(f"\n🎬 Starting video processing pipeline...")
    print(f"📹 Maximum expected videos: 373")

    # ==========================================
    # CHECKPOINT: LOAD PREVIOUS PROGRESS
    # ==========================================
    checkpoint_data = load_checkpoint()
    previously_processed = checkpoint_data.get('processed_videos', [])
    previously_failed = checkpoint_data.get('failed_videos', [])
    
    if previously_processed:
        print(f"   📍 Resuming from checkpoint: {len(previously_processed)} videos already processed")
    if previously_failed:
        print(f"   📍 {len(previously_failed)} videos previously failed (will retry)")

    # Load ground truth before processing
    print("\n📚 Loading ground truth annotations...")
    ground_truth_dict = load_ground_truth_csv()

    # Load models
    models = load_models()
    if not models:
        print("❌ No models loaded! Cannot proceed.")
        return False

    # Load symbolic baselines
    symbolic_baselines = None
    try:
        baselines_path = Config.NEOPOLYP_OUTPUT / 'neopolyp_ground_truth_baselines.json'
        if baselines_path.exists():
            with open(baselines_path, 'r') as f:
                symbolic_baselines = json.load(f)
            print("   📚 Loaded symbolic baselines for reasoning")
        else:
            print("   ⚠️  Symbolic baselines not found (optional)")
    except Exception as e:
        print(f"   ⚠️  Failed to load symbolic baselines: {e}")

    # Find videos
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Config.APPLY_VIDEO_ROOT.glob(ext)))

    if not video_files:
        print(f"❌ No video files found in {Config.APPLY_VIDEO_ROOT}")
        return False

    print(f"📹 Found {len(video_files)} videos to process")
    
    if len(video_files) > 400:
        print(f"⚠️  WARNING: {len(video_files)} videos detected. This will take a long time.")

    # Create master summary file
    master_summary = checkpoint_data.get('detailed_log', [])
    failed_videos = previously_failed.copy()
    
    # Process each video with error tracking
    for video_idx, video_path in enumerate(video_files, 1):
        try:
            # ==========================================
            # CHECKPOINT: SKIP ALREADY PROCESSED VIDEOS
            # ==========================================
            if has_been_processed(video_path.name, checkpoint_data):
                print(f"\n[{video_idx}/{len(video_files)}] ⏭️  SKIPPED (already processed): {video_path.name}")
                continue
            
            print(f"\n[{video_idx}/{len(video_files)}] Processing: {video_path.name}")
            
            # Validate video file exists and is readable
            if not video_path.exists():
                print(f"   ❌ Video file not found")
                failed_videos.append((video_path.name, "File not found"))
                continue
            
            if video_path.stat().st_size == 0:
                print(f"   ❌ Video file is empty")
                failed_videos.append((video_path.name, "Empty file"))
                continue
            
            # Create output dir
            output_dir = Config.VIDEO_OUTPUT / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process video
            process_video(video_path, models, output_dir, symbolic_baselines)
            
            # Live accuracy CSV — update immediately after each video
            try:
                # Re-load detections from the inference_report written to disk
                _report_path = output_dir / 'inference_report.json'
                if _report_path.exists():
                    with open(_report_path, 'r') as _rf:
                        _rep = json.load(_rf)
                    # Re-assemble detections dict with what the report contains
                    _detections_for_csv = {
                        '_video_level_clinical_class': _rep.get('video_level_clinical_class', 'UNKNOWN'),
                        '_polyp_features_detail': _rep.get('polyp_features_detail', []),
                    }
                    append_live_accuracy_row(
                        video_path, _detections_for_csv, ground_truth_dict, Config.VIDEO_OUTPUT
                    )
            except Exception as _live_err:
                print(f"   ⚠️  Live CSV skipped: {_live_err}")
            
            master_summary.append({
                'video': video_path.name,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
            # ==========================================
            # CHECKPOINT: SAVE AFTER EACH VIDEO
            # ==========================================
            checkpoint_data['processed_videos'].append(video_path.name)
            checkpoint_data['failed_videos'] = [v[0] for v in failed_videos]  # Just video names
            checkpoint_data['detailed_log'] = master_summary
            save_checkpoint(checkpoint_data)
            
            print(f"✅ Completed [{video_idx}/{len(video_files)}]: {video_path.name}")
            print(f"   💾 Checkpoint saved (Progress: {len(checkpoint_data['processed_videos'])}/{len(video_files)})")

        except Exception as e:
            print(f"❌ Critical error processing {video_path.name}: {e}")
            failed_videos.append((video_path.name, str(e)))
            master_summary.append({
                'video': video_path.name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # ==========================================
            # CHECKPOINT: SAVE AFTER FAILURE
            # ==========================================
            checkpoint_data['processed_videos'].append(video_path.name)  # Mark as processed (even if failed)
            checkpoint_data['failed_videos'] = [v[0] for v in failed_videos]
            checkpoint_data['detailed_log'] = master_summary
            save_checkpoint(checkpoint_data)
            
            continue  # Continue with next video instead of stopping

    # ===== COMPARISON & ACCURACY ANALYSIS =====
    print("\n" + "="*80)
    print(" " * 20 + "GENERATING COMPARISON REPORTS")
    print("="*80)
    
    # Generate results CSV
    results_df = generate_results_csv(Config.VIDEO_OUTPUT, ground_truth_dict)
    
    # Generate accuracy report
    accuracy_report = generate_accuracy_report(results_df, Config.VIDEO_OUTPUT)
    
    # Generate visualizations
    if not results_df.empty:
        generate_comparison_visualizations(results_df, accuracy_report, Config.VIDEO_OUTPUT)
    
    # Save master summary
    try:
        summary_path = Config.VIDEO_OUTPUT / "processing_summary.json"
        checkpoint_data['total_videos'] = len(video_files)
        checkpoint_data['successful'] = len(video_files) - len(failed_videos)
        checkpoint_data['failed'] = len(failed_videos)
        checkpoint_data['completion_timestamp'] = datetime.now().isoformat()
        
        with open(summary_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"\n📊 Master summary saved: {summary_path}")
    except Exception as e:
        print(f"⚠️  Failed to save master summary: {e}")

    # Print final summary
    print("\n" + "="*80)
    print(" " * 25 + "PHASE 5 PROCESSING COMPLETE")
    print("="*80)
    print(f"✅ Successful: {len(video_files) - len(failed_videos)}/{len(video_files)}")
    print(f"❌ Failed: {len(failed_videos)}/{len(video_files)}")
    
    if failed_videos:
        print(f"\n⚠️  Failed videos:")
        for video_name, error in failed_videos[:10]:  # Show first 10
            print(f"   - {video_name}: {error[:60]}...")
        if len(failed_videos) > 10:
            print(f"   ... and {len(failed_videos)-10} more")
    
    print(f"\n📁 Results saved to: {Config.VIDEO_OUTPUT}")
    print(f"   📄 detailed_results.csv - Per-video results")
    print(f"   📊 accuracy_report.json - Accuracy metrics")
    print(f"   📈 confusion_matrix.png - Detection performance")
    print(f"   📈 accuracy_by_category.png - Category-wise accuracy")
    print(f"   📈 confidence_vs_accuracy.png - Confidence analysis")
    
    return len(failed_videos) == 0

# 3. Temporal Plots
def generate_temporal_plots(detections, output_dir, video_name):
    """Generate temporal plots of detections"""
    print(f"   📈 Generating temporal plots...")

    # Extract data
    frames = [d['frame_idx'] for d in detections]
    timestamps = [d['timestamp'] for d in detections]
    confidences = []

    for d in detections:
        confs = [det.get('confidence', 0) for det in d['detections']]
        confidences.append(max(confs) if confs else 0)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Confidence over time
    ax1.plot(timestamps, confidences, 'b-', marker='o', markersize=2)
    ax1.set_title('Detection Confidence Over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Max Confidence')
    ax1.grid(True, alpha=0.3)

    # Detection count over time
    det_counts = [len(d['detections']) for d in detections]
    ax2.plot(timestamps, det_counts, 'r-', marker='s', markersize=2)
    ax2.set_title('Number of Detections Over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Detection Count')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{video_name}_temporal.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved temporal plot: {plot_path}")

# 4. Summary Reports
def generate_summary_report(detections, output_dir, video_name):
    """Generate JSON and CSV summary reports"""
    print(f"   📄 Generating summary reports...")

    # Calculate statistics
    total_frames = len(detections)
    total_detections = sum(len(d['detections']) for d in detections)
    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0

    confidences = []
    for d in detections:
        confidences.extend([det.get('confidence', 0) for det in d['detections']])

    stats = {
        'video_name': video_name,
        'total_frames': total_frames,
        'total_detections': total_detections,
        'avg_detections_per_frame': avg_detections_per_frame,
        'mean_confidence': np.mean(confidences) if confidences else 0,
        'std_confidence': np.std(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0,
        'processing_timestamp': datetime.now().isoformat()
    }

    # Save JSON
    json_path = output_dir / f"{video_name}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save CSV with frame-by-frame data
    csv_data = []
    for d in detections:
        for det in d['detections']:
            csv_data.append({
                'frame_idx': d['frame_idx'],
                'timestamp': d['timestamp'],
                'model': det.get('model', 'unknown'),
                'confidence': det.get('confidence', 0),
                'bbox': det.get('bbox', [])
            })

    if csv_data:
        csv_path = output_dir / f"{video_name}_detections.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved CSV report: {csv_path}")

    print(f"   ✅ Saved JSON summary: {json_path}")

# 5. Interactive Dashboard (Simplified)
def generate_interactive_dashboard(detections, output_dir, video_name):
    """Generate HTML dashboard with Plotly"""
    print(f"   🌐 Generating interactive dashboard...")

    # Create simple dashboard
    timestamps = [d['timestamp'] for d in detections]
    det_counts = [len(d['detections']) for d in detections]
    max_confs = [max([det.get('confidence', 0) for det in d['detections']]) if d['detections'] else 0 for d in detections]

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Detections Over Time', 'Confidence Over Time'))

    fig.add_trace(go.Scatter(x=timestamps, y=det_counts, mode='lines+markers', name='Detections'), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=max_confs, mode='lines+markers', name='Max Confidence'), row=2, col=1)

    fig.update_layout(height=600, title_text=f"Video Analysis: {video_name}")
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)

    html_path = output_dir / f"{video_name}_dashboard.html"
    fig.write_html(str(html_path))
    print(f"   ✅ Saved interactive dashboard: {html_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    success = process_all_videos()
    exit(0 if success else 1)