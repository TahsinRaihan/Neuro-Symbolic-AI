# -*- coding: utf-8 -*-
"""
PHASE 2: Self-Supervised Learning (SSL)
Trains Vision Transformer on Dataset 2 using SimCLR + Reconstruction

Training Config: 3 epochs
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import gc
import glob
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

print("=" * 80)
print(" " * 20 + "PHASE 2: SELF-SUPERVISED LEARNING")
print(" " * 15 + "(ViT on Dataset 3 - Treating as Unlabeled)")
print("=" * 80)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    THESIS_ROOT = Path(__file__).parent.parent.parent.absolute()
    # Use Dataset 3 for SSL training
    DATASET_3_ROOT = THESIS_ROOT / 'NeSy' / 'Dataset 3'
    OUTPUT_ROOT = THESIS_ROOT / 'thesis_outputs'
    SSL_OUTPUT = OUTPUT_ROOT / 'ssl_outputs'
    VISUAL_OUTPUT = OUTPUT_ROOT / 'visualizations'

    # Training Config (RTX 4080 Ti Super 16 GB VRAM, 64 GB RAM)
    EPOCHS = 100
    BATCH_SIZE = 192  # ViT-Small @ 256px, 16 GB VRAM with AMP
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.5
    ALPHA = 0.5  # Balance contrastive vs reconstruction
    IMG_SIZE = 256  # Match ROI size from Phase 1
    PROJECTION_DIM = 128
    NUM_WORKERS = 8  # 64 GB RAM – increased for better CPU utilization

    DEVICE = torch.device("cuda")  # Force GPU usage
    CHECKPOINT_FREQ = 10  # Save every 10 epochs

Config.SSL_OUTPUT.mkdir(parents=True, exist_ok=True)
Config.VISUAL_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\n[Configuration]:")
print(f"   Device: {Config.DEVICE}")
print(f"   Epochs: {Config.EPOCHS}")
print(f"   Batch Size: {Config.BATCH_SIZE}")
print(f"   Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
print(f"   Dataset: {Config.DATASET_3_ROOT}")

# ==========================================
# DATASET
# ==========================================
class SSLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for ext in ['**/*.jpg', '**/*.jpeg', '**/*.png']:
            self.image_paths.extend(list(root_dir.glob(ext)))

        self.image_paths = sorted(list(set(self.image_paths)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        print(f"   Found {len(self.image_paths):,} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            view1 = self.transform(img)
            view2 = self.transform(img)
            return view1, view2

        return transforms.ToTensor()(img), transforms.ToTensor()(img)

# ==========================================
# MODEL
# ==========================================
class NeuroSymbolicSSL(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # Create ViT with 256x256 image size to match our ROIs
        self.backbone = timm.create_model('vit_small_patch16_224',
                                         pretrained=True,
                                         num_classes=0,
                                         img_size=256)  # Match ROI size
        self.backbone_dim = 384

        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.backbone_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        reconstruction = self.decoder(features)
        return features, projections, reconstruction

# ==========================================
# LOSS FUNCTION
# ==========================================
class CombinedSSLLoss(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def nt_xent_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        features = torch.cat([z1, z2], dim=0)
        batch_size = z1.shape[0]

        similarity = torch.matmul(features, features.T) / self.temperature
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity = similarity[~mask].view(similarity.shape[0], -1)

        positives = similarity[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity[~labels.bool()].view(similarity.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        targets = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)

        return F.cross_entropy(logits, targets)

    def forward(self, z1, z2, feat1, rec1):
        loss_contrastive = self.nt_xent_loss(z1, z2)
        loss_reconstruction = self.mse(feat1, rec1)
        total_loss = (1 - self.alpha) * loss_contrastive + self.alpha * loss_reconstruction
        return total_loss, loss_contrastive, loss_reconstruction

# ==========================================
# TRAINING
# ==========================================
def train_ssl():
    print("\n" + "=" * 80)
    print(" " * 30 + "STARTING TRAINING")
    print("=" * 80)

    # Heavy Augmentations for small dataset (1k samples)
    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),  # Aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # Added vertical flip
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),  # Camera invariance
        transforms.RandomRotation(degrees=15),  # Rotation augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset & DataLoader
    dataset = SSLDataset(Config.DATASET_3_ROOT, transform=ssl_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,      # 64 GB RAM – prefetch workers
        pin_memory=True,                     # Faster transfer to GPU
        persistent_workers=True,
        drop_last=True
    )

    print(f"\n📊 Training Details:")
    print(f"   Images: {len(dataset):,}")
    print(f"   Batches/Epoch: {len(dataloader):,}")
    print(f"   Total Iterations: {len(dataloader) * Config.EPOCHS:,}")

    # Model
    model = NeuroSymbolicSSL(feature_dim=Config.PROJECTION_DIM).to(Config.DEVICE)
    criterion = CombinedSSLLoss(temperature=Config.TEMPERATURE, alpha=Config.ALPHA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Initialize loss history first (needed for checkpoint resume)
    loss_history = {'total': [], 'contrastive': [], 'reconstruction': []}

    # Check for latest checkpoint to resume training
    start_epoch = 0
    if Config.SSL_OUTPUT.exists():
        checkpoints = list(Config.SSL_OUTPUT.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loss_history = checkpoint.get('loss_history', loss_history)
            start_epoch = checkpoint['epoch'] + 1
            print(f"   ✅ Resumed from epoch {start_epoch}")

    print(f"\n🚀 Training...\n")
    model.train()

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_loss = 0
        epoch_contrastive = 0
        epoch_reconstruction = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", ncols=100)

        for view1, view2 in pbar:
            view1 = view1.to(Config.DEVICE)
            view2 = view2.to(Config.DEVICE)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    feat1, proj1, rec1 = model(view1)
                    feat2, proj2, rec2 = model(view2)
                    loss, loss_clr, loss_rec = criterion(proj1, proj2, feat1, rec1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                feat1, proj1, rec1 = model(view1)
                feat2, proj2, rec2 = model(view2)
                loss, loss_clr, loss_rec = criterion(proj1, proj2, feat1, rec1)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_contrastive += loss_clr.item()
            epoch_reconstruction += loss_rec.item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_clr = epoch_contrastive / len(dataloader)
        avg_rec = epoch_reconstruction / len(dataloader)

        loss_history['total'].append(avg_loss)
        loss_history['contrastive'].append(avg_clr)
        loss_history['reconstruction'].append(avg_rec)

        scheduler.step()

        print(f"   Loss: {avg_loss:.4f} | Contrastive: {avg_clr:.4f} | Reconstruction: {avg_rec:.4f}")

        # Checkpoint
        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0 or (epoch + 1) == Config.EPOCHS:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_history': loss_history,
                'loss': avg_loss
            }
            torch.save(checkpoint, str(Config.SSL_OUTPUT / f'checkpoint_epoch_{epoch+1}.pth'))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    encoder_path = Config.SSL_OUTPUT / 'ssl_encoder_final.pth'
    torch.save(model.backbone.state_dict(), str(encoder_path))

    full_model_path = Config.SSL_OUTPUT / 'ssl_model_final.pth'
    torch.save(model.state_dict(), str(full_model_path))

    print(f"\n✅ Training Complete!")
    print(f"   Encoder saved: {encoder_path}")
    print(f"   Full model saved: {full_model_path}")

    # Plot curves
    plot_training_curves(loss_history)

    return model, loss_history

def plot_training_curves(loss_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(loss_history['total']) + 1)

    axes[0].plot(epochs, loss_history['total'], 'b-', linewidth=2, marker='o')
    axes[0].set_title('Total Loss', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss_history['contrastive'], 'r-', linewidth=2, marker='s')
    axes[1].set_title('Contrastive Loss', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss_history['reconstruction'], 'g-', linewidth=2, marker='^')
    axes[2].set_title('Reconstruction Loss', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Config.VISUAL_OUTPUT / 'ssl_training_curves.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Training curves saved: {save_path}")

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    # CRITICAL for Windows multiprocessing compatibility
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        if not Config.DATASET_3_ROOT.exists():
            print(f"[X] Dataset 3 not found at {Config.DATASET_3_ROOT}")
            print("   Please run Phase 1 first!")
            sys.exit(1)

        print("[OK] Cropped ROIs found\n")

        # Initial memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, loss_history = train_ssl()

        print("\n" + "=" * 80)
        print(" " * 25 + "PHASE 2 COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Final Loss: {loss_history['total'][-1]:.4f}")
        print(f"   Models saved to: {Config.SSL_OUTPUT}")

        # Final memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n➡️  Next Steps:")
        print("   Phase 2.5: python phase2_5_neopolyp/neopolyp_preparation.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
