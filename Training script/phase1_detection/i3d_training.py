# -*- coding: utf-8 -*-
"""
PHASE 1D: I3D Video Detection Training (70-30 Split)
Trains I3D model on actual video dataset (373 videos from Apply Video folder)
70% (260 videos) for training, 30% (113 videos) for validation
Uses ground truth labels from detailed_results.csv
"""

import os
import sys
import csv
import random

# Suppress ffmpeg warnings at the environment level
os.environ['FFREPORT'] = 'file=/dev/null'
os.environ['FFMPEG_SUPPRESS_WARNINGS'] = '1'

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter

print("=" * 80)
print(" " * 25 + "PHASE 1D: I3D VIDEO DETECTION TRAINING")
print(" " * 20 + "(I3D on 373 Actual Videos - 70-30 Split)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    APPLY_VIDEO_ROOT = THESIS_ROOT / 'NeSy' / 'Dataset i3d'  # I3D training dataset
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    DETECTION_OUTPUT = OUTPUT_ROOT / 'detection_models'
    VIDEO_INFERENCE_RESULTS = OUTPUT_ROOT / 'video_inference_results'
    GROUND_TRUTH_CSV = VIDEO_INFERENCE_RESULTS / 'detailed_results.csv'

    # Training Config (RTX 4080 Ti Super 16 GB VRAM, 32 GB RAM)
    EPOCHS = 50
    I3D_BATCH = 44  # Optimized for 16GB VRAM - larger batch = fewer disk reads = faster training
    IMG_SIZE = (224, 224)  # I3D standard input size
    CLIP_LENGTH = 16  # Frames per clip
    CLIP_STRIDE = 8   # Stride for extracting overlapping clips
    DEVICE = 'cuda'
    NUM_WORKERS = 7  # Balanced: avoid disk I/O thrashing, but still parallel loading
    PREFETCH_FACTOR = 3  # Queue more batches to prevent GPU stalling
    PERSISTENT_WORKERS = True  # Keep workers alive (less overhead)
    LR = 1e-4         # Lower LR for fine-tuning on actual videos
    AMP = True        # Mixed precision
    
    # Checkpoint/Resume Config
    SAVE_CHECKPOINT_EVERY = 1  # Save checkpoint after every epoch
    CHECKPOINT_DIR = DETECTION_OUTPUT / 'i3d_polyp' / 'checkpoints'
    CHECKPOINT_PREFIX = 'i3d_checkpoint'
    CHECKPOINT_METADATA = CHECKPOINT_DIR / 'checkpoint_metadata.json'
    
    # Data split
    TRAIN_RATIO = 0.70  # 70% for training (260 videos)
    VAL_RATIO = 0.30   # 30% for validation (113 videos)
    
    # Loss weighting (polyp is positive class)
    POLYP_CLASS_WEIGHT = 4.0  # Weighted loss to balance classes
    
    # Random seed for reproducibility
    SEED = 42

# Set random seeds
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)

# Create directories
os.makedirs(Config.DETECTION_OUTPUT / 'i3d_polyp', exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# SUPPRESS LIBAV/FFMPEG CODEC WARNINGS
# ==========================================
import contextlib
@contextlib.contextmanager
def suppress_libav_warnings():
    """Suppress ffmpeg/libav codec warnings from OpenCV VideoCapture"""
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

# ==========================================
# I3D MODEL IMPLEMENTATION
# ==========================================
class I3D(nn.Module):
    def __init__(self, num_classes=2):  # Polyp detection: background, polyp
        super(I3D, self).__init__()
        # Simplified I3D architecture
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1))
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool4(self.bn4(torch.relu(self.conv4(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==========================================
# LOAD GROUND TRUTH FROM CSV
# ==========================================
def load_ground_truth_labels():
    """Load ground truth labels from detailed_results.csv"""
    labels = {}
    
    if not Config.GROUND_TRUTH_CSV.exists():
        print(f"   ⚠️  Ground truth CSV not found at {Config.GROUND_TRUTH_CSV}")
        return labels
    
    try:
        with open(Config.GROUND_TRUTH_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row['video_name']
                has_polyp = row['ground_truth_has_polyp'].lower() == 'true'
                labels[video_name] = int(has_polyp)  # 1 = polyp, 0 = no polyp
        
        print(f"   ✅ Loaded ground truth for {len(labels)} videos")
        class_counts = Counter(labels.values())
        print(f"      Polyp videos: {class_counts[1]}")
        print(f"      Non-polyp videos: {class_counts[0]}")
    except Exception as e:
        print(f"   ⚠️  Error loading ground truth: {e}")
    
    return labels

# ==========================================
# CREATE 70-30 TRAIN-VAL SPLIT
# ==========================================
def create_train_val_split(ground_truth_labels):
    """
    Split videos into 70% train (260) and 30% validation (113)
    Maintains class balance in both splits
    """
    video_names = list(ground_truth_labels.keys())
    print(f"\n   📊 Total videos: {len(video_names)}")
    
    # Separate by class for stratified split
    polyp_videos = [v for v in video_names if ground_truth_labels[v] == 1]
    non_polyp_videos = [v for v in video_names if ground_truth_labels[v] == 0]
    
    # Shuffle with seed for reproducibility
    random.shuffle(polyp_videos)
    random.shuffle(non_polyp_videos)
    
    # Split each class
    polyp_train_count = int(len(polyp_videos) * Config.TRAIN_RATIO)
    non_polyp_train_count = int(len(non_polyp_videos) * Config.TRAIN_RATIO)
    
    train_videos = (polyp_videos[:polyp_train_count] + 
                   non_polyp_videos[:non_polyp_train_count])
    val_videos = (polyp_videos[polyp_train_count:] + 
                 non_polyp_videos[non_polyp_train_count:])
    
    random.shuffle(train_videos)
    random.shuffle(val_videos)
    
    print(f"   📚 Train split: {len(train_videos)} videos")
    print(f"      Polyp: {sum(1 for v in train_videos if ground_truth_labels[v] == 1)}")
    print(f"      Non-polyp: {sum(1 for v in train_videos if ground_truth_labels[v] == 0)}")
    print(f"   📚 Val split: {len(val_videos)} videos")
    print(f"      Polyp: {sum(1 for v in val_videos if ground_truth_labels[v] == 1)}")
    print(f"      Non-polyp: {sum(1 for v in val_videos if ground_truth_labels[v] == 0)}")
    
    return {'train': train_videos, 'val': val_videos}

# ==========================================
# DATASET FOR VIDEO CLIPS
# ==========================================
class VideoDataset(Dataset):
    """Load 16-frame clips from actual videos with ground truth labels"""
    
    def __init__(self, video_names, ground_truth_labels, is_training=True):
        self.video_names = video_names
        self.ground_truth_labels = ground_truth_labels
        self.is_training = is_training
        self.clips = []  # List of (video_name, start_frame, label)
        
        # Extract all clips from videos
        print(f"\n   📹 Extracting {len(video_names)} videos...")
        for video_name in tqdm(video_names, desc="Extracting clips"):
            video_path = Config.APPLY_VIDEO_ROOT / f"{video_name}.mp4"
            if not video_path.exists():
                # Try common extensions
                for ext in ['.avi', '.mov', '.mkv', '.flv']:
                    alt_path = Config.APPLY_VIDEO_ROOT / f"{video_name}{ext}"
                    if alt_path.exists():
                        video_path = alt_path
                        break
            
            if not video_path.exists():
                continue
            
            try:
                with suppress_libav_warnings():
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                
                label = self.ground_truth_labels.get(video_name, 0)
                
                # Extract overlapping clips with stride
                for start_frame in range(0, total_frames - Config.CLIP_LENGTH, Config.CLIP_STRIDE):
                    self.clips.append((video_name, start_frame, label))
                
            except Exception as e:
                print(f"      ⚠️  Error processing {video_name}: {e}")
        
        print(f"   ✅ Extracted {len(self.clips)} clips total")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        video_name, start_frame, label = self.clips[idx]
        
        video_path = Config.APPLY_VIDEO_ROOT / f"{video_name}.mp4"
        if not video_path.exists():
            # Try common extensions
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                alt_path = Config.APPLY_VIDEO_ROOT / f"{video_name}{ext}"
                if alt_path.exists():
                    video_path = alt_path
                    break
        
        with suppress_libav_warnings():
            cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(Config.CLIP_LENGTH):
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame if video is too short
                if frames:
                    frame = frames[-1]
                else:
                    frame = np.zeros((Config.IMG_SIZE[1], Config.IMG_SIZE[0], 3), dtype=np.uint8)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, Config.IMG_SIZE)
            frames.append(frame)
        
        cap.release()
        
        # Convert to torch tensor (C, T, H, W)
        clip = np.stack(frames, axis=0)  # (T, H, W, C)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        
        # Apply augmentation during training
        if self.is_training:
            # Random brightness
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                clip = clip * brightness
            
            # Random contrast
            if random.random() > 0.5:
                contrast = random.uniform(0.8, 1.2)
                clip = clip * contrast + (1 - contrast) * 0.5
        
        clip = torch.clamp(clip, 0.0, 1.0)
        
        return clip, label

# ==========================================
# TRAINING AND VALIDATION
# ==========================================
def validate(model, val_loader, criterion, device):
    """Validate the model on validation set"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy

# ==========================================
# CHECKPOINT/RESUME UTILITIES
# ==========================================
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, val_acc_history, scaler=None):
    """Save training checkpoint to resume later"""
    checkpoint_path = Config.CHECKPOINT_DIR / f"{Config.CHECKPOINT_PREFIX}_epoch_{epoch+1:03d}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'val_acc_history': val_acc_history,
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    
    # Update metadata
    metadata = {
        'last_epoch': epoch,
        'last_checkpoint': str(checkpoint_path),
        'best_val_acc': best_val_acc,
        'total_epochs_trained': epoch + 1
    }
    
    with open(Config.CHECKPOINT_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler):
    """Load last checkpoint if it exists and return starting epoch"""
    if not Config.CHECKPOINT_METADATA.exists():
        return 0, 0.0, [], None  # Start from epoch 0
    
    try:
        with open(Config.CHECKPOINT_METADATA, 'r') as f:
            metadata = json.load(f)
        
        checkpoint_path = Path(metadata['last_checkpoint'])
        if not checkpoint_path.exists():
            print(f"   ⚠️  Checkpoint file not found: {checkpoint_path}")
            return 0, 0.0, [], None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        scaler = None
        if 'scaler_state_dict' in checkpoint and Config.AMP:
            scaler = torch.amp.GradScaler('cuda')
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        val_acc_history = checkpoint['val_acc_history']
        
        print(f"   ✅ Resumed from checkpoint: Epoch {start_epoch}/{Config.EPOCHS}")
        print(f"      Previous best validation accuracy: {best_val_acc:.2f}%")
        
        return start_epoch, best_val_acc, val_acc_history, scaler
    
    except Exception as e:
        print(f"   ⚠️  Could not load checkpoint: {e}")
        return 0, 0.0, [], None

def train_i3d():
    """Train I3D model on 70% of 373 videos, validate on 30%"""
    
    print("\n🔧 Initializing I3D model...")
    model = I3D(num_classes=2).to(Config.DEVICE)
    
    # Load ground truth labels
    print("\n📖 Loading ground truth labels...")
    ground_truth_labels = load_ground_truth_labels()
    
    if not ground_truth_labels:
        print("   ❌ No ground truth labels loaded. Exiting.")
        return False
    
    # Create train-val split
    print("\n🔀 Creating 70-30 train-validation split...")
    splits = create_train_val_split(ground_truth_labels)
    
    # Create datasets
    print("\n🎬 Creating datasets...")
    train_dataset = VideoDataset(splits['train'], ground_truth_labels, is_training=True)
    val_dataset = VideoDataset(splits['val'], ground_truth_labels, is_training=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("   ❌ No clips extracted. Check video paths.")
        return False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.I3D_BATCH, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.I3D_BATCH, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    
    # Loss and optimizer with class weighting
    class_weights = torch.tensor([1.0, Config.POLYP_CLASS_WEIGHT], device=Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # AMP scaler (using new torch.amp API)
    scaler = torch.amp.GradScaler('cuda') if Config.AMP else None
    
    # ⚡ Enable cuDNN optimizations for faster convolutions
    torch.backends.cudnn.benchmark = True  # Find optimal convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matrix ops (RTX 4090/4080 support)
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch size: {Config.I3D_BATCH}")
    print(f"   Learning rate: {Config.LR}")
    print(f"   Clip length: {Config.CLIP_LENGTH} frames")
    print(f"   Train clips: {len(train_dataset)}")
    print(f"   Val clips: {len(val_dataset)}")
    print(f"   Polyp class weight: {Config.POLYP_CLASS_WEIGHT}")
    print(f"   Device: {Config.DEVICE}")
    
    # Try to resume from checkpoint
    print(f"\n🔍 Checking for existing checkpoints...")
    start_epoch, best_val_acc, val_acc_history, loaded_scaler = load_checkpoint(model, optimizer, scheduler)
    
    if loaded_scaler is not None:
        scaler = loaded_scaler
    
    best_model_path = Config.DETECTION_OUTPUT / 'i3d_polyp' / 'i3d_best.pth'
    
    print(f"\n🚀 Starting training on {Config.DEVICE}...")
    print(f"   ⚙️  Batch size: {Config.I3D_BATCH}")
    print(f"   ⚙️  Workers: {Config.NUM_WORKERS} (reduced I/O contention)")
    print(f"   ⚙️  Prefetch factor: {Config.PREFETCH_FACTOR}")
    print(f"   ⚙️  Persistent workers: {Config.PERSISTENT_WORKERS}")
    print(f"   ⚙️  Expected VRAM usage: ~15.5GB (from ~10GB)")
    print(f"   ⚡ cuDNN Benchmark: Enabled (find optimal conv kernels)")
    print(f"   ⚡ TF32: Enabled (faster matrix ops on RTX 4080)\n")
    
    if torch.cuda.is_available():
        print(f"   📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for clips, labels in pbar:
            # Non-blocking GPU transfer allows computation overlap
            clips, labels = clips.to(Config.DEVICE, non_blocking=True), labels.to(Config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if Config.AMP:
                with torch.amp.autocast('cuda'):
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(clips)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss / (train_total / Config.I3D_BATCH)})
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Clear GPU cache to prevent memory fragmentation
        torch.cuda.empty_cache()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Track validation accuracy
        val_acc_history.append(val_acc)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (BEST)")
        else:
            print(f"   Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every N epochs (for resumable training)
        if (epoch + 1) % Config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, val_acc_history, scaler)
            print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {best_model_path}")
    
    return True

if __name__ == "__main__":
    print("\n🚀 Starting I3D training on Dataset i3d (70-30 split)...")
    print(f"   Dataset i3d folder: {Config.APPLY_VIDEO_ROOT}")
    print(f"   Ground truth CSV: {Config.GROUND_TRUTH_CSV}")
    
    success = train_i3d()
    
    if success:
        print("\n✅ I3D training completed successfully!")
    else:
        print("\n❌ I3D training failed!")