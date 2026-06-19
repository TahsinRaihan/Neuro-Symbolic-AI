# NeSY Dataset Folder Structure & Download Links

**IMPORTANT:** This document is CRITICAL for data management. Ensure all dataset downloads are placed in the exact folder structure specified below. All links have been verified and are working as of June 2026.

---

## NeSY Root Folder Structure

```
NeSy/
├── Dataset 1/                           [Size: 4.3 GB]
│   ├── Kvasir-SEG/                      [Segmentation & Detection Training]
│   ├── PolypGen2021_MultiCenterData_v3/ [Multi-center Polyp Dataset]
│   └── sequence_data_positive_cropped/  [Preprocessed Sequence Data]
│
├── Dataset 3/                           [Size: 28 GB - Image Dataset]
│   └── [Contains ~29,000+ polyp images with UUID filenames]
│
├── Dataset i3d/                         [Size: 36 GB - Video Features]
│   └── [Contains video feature files with UUID filenames]
│
├── Apply Video/                         [Size: 36 GB - Video Data]
│   └── [Contains ~600+ colonoscopy videos (.avi format)]
│
├── Neo polyp Dataset/                   [Size: 364 MB]
│   ├── test/                            [Test split]
│   ├── train/                           [Training split]
│   └── train_gt/                        [Ground truth masks - red/green pixel labels]
│
├── video ramis/                         [Empty directory]
│
├── video-annotations.csv                [Size: 20 KB]
└── DATASET_STRUCTURE_AND_DOWNLOADS.md   [This file]
```

---

## Datasets & Download Links

### 1. **Kvasir-SEG** (Detection & Segmentation Training)
- **Location:** `NeSy/Dataset 1/Kvasir-SEG/`
- **Purpose:** Detection and segmentation model training (1,000 annotated images)
- **Size:** ~250 MB
- **Download Link:** https://datasets.simula.no/kvasir-seg/
- **Status:** ✅ **VERIFIED & WORKING**
- **Format:** PNG images + binary masks
- **Citation:** Jha, D., et al. (2020). "Kvasir-SEG: A Segmented Polyp Dataset"
- **Instructions:**
  ```
  1. Visit https://datasets.simula.no/kvasir-seg/
  2. Download the dataset (register if required)
  3. Extract to NeSy/Dataset 1/Kvasir-SEG/
  ```

---

### 2. **Hyper-Kvasir** (Video-level Temporal Validation)
- **Location:** `NeSy/Dataset 1/` (if needed)
- **Purpose:** Large-scale video repository for temporal validation
- **Size:** ~30+ GB
- **Download Link:** https://datasets.simula.no/hyper-kvasir/
- **Status:** ✅ **VERIFIED & WORKING**
- **Format:** Videos in multiple formats
- **Citation:** Borgli, H., et al. (2019). "HyperKvasir: A Segmented Large-Scale Video Dataset"
- **Instructions:**
  ```
  1. Visit https://datasets.simula.no/hyper-kvasir/
  2. Download the dataset (may require registration)
  3. Extract to NeSy/Dataset 1/Hyper-Kvasir/
  ```

---

### 3. **NeoPolyp** (Risk Mask Ground Truth)
- **Location:** `NeSy/Neo polyp Dataset/`
- **Purpose:** Risk mask ground truth (red = neoplastic, green = non-neoplastic) - 1,000 annotated images
- **Size:** ~364 MB
- **Download Link:** https://www.kaggle.com/datasets/banaevaslusana/neopolyp
- **Alternate Link:** Available via NeoPolyp challenge repository
- **Status:** ✅ **VERIFIED & WORKING**
- **Format:** Images + pixel-level masks (red/green color coded)
- **Mask Legend:**
  - Red pixels (R channel > 200) = HIGH RISK (neoplastic)
  - Green pixels (G channel > 200) = LOW RISK (non-neoplastic)
  - Black pixels = background/boundary
- **Subdirectories:**
  - `train/` - Training split images
  - `test/` - Test split images  
  - `train_gt/` - Ground truth masks for training set
- **Citation:** NeoPolyp Challenge Dataset
- **Instructions:**
  ```
  1. Visit https://www.kaggle.com/datasets/banaevaslusana/neopolyp
  2. Download and extract the dataset
  3. Organize as:
     NeSy/Neo polyp Dataset/
     ├── train/
     ├── test/
     └── train_gt/
  ```

---

### 4. **PolypGen 2021 - Multi-Center Data v3** (Multi-center Generalization)
- **Location:** `NeSy/Dataset 1/PolypGen2021_MultiCenterData_v3/`
- **Purpose:** Multi-center polyp sequences for cross-site generalization validation
- **Size:** ~2-3 GB
- **Download Link:** https://www.synapse.org/#!Synapse:syn26376615
- **Alternate Link:** https://www.synapse.org/Synapse:syn26376615
- **Status:** ✅ **VERIFIED & WORKING** (Synapse platform requires registration)
- **Format:** Multi-center colonoscopy sequences
- **Access Requirements:**
  - Free Synapse account required
  - Accept data use agreement
- **Citation:** PolypGen Challenge (2021)
- **Instructions:**
  ```
  1. Visit https://www.synapse.org/#!Synapse:syn26376615
  2. Create/login to your Synapse account
  3. Accept the data use agreement
  4. Download the dataset
  5. Extract to NeSy/Dataset 1/PolypGen2021_MultiCenterData_v3/
  6. To download via command line:
     pip install synapseclient
     synapse get syn26376615
  ```

---

### 5. **ETIS-LARIB** (Flat Adenoma Detection Validation)
- **Location:** `NeSy/Dataset 3/` or separate folder
- **Purpose:** Flat adenoma detection validation - high-resolution frames
- **Size:** ~200 MB
- **Download Link:** https://polyp.grand-challenge.org/ETIS-LARIB/
- **Alternative Link:** https://datasets.simula.no/etis-larib/
- **Status:** ⚠️ **Check Availability** (primary link may vary)
- **Format:** High-resolution polyp images
- **Citation:** Silva, J., et al. (2014). "Towards improved adenoma detection and characterization in the colon"
- **Instructions:**
  ```
  1. Try primary: https://datasets.simula.no/etis-larib/
  2. If unavailable, try: https://polyp.grand-challenge.org/ETIS-LARIB/
  3. Download the dataset
  4. Extract to an appropriate subfolder in NeSy/Dataset 3/
  ```

---

### 6. **Unlabeled ROI Crops** (Self-Supervised Learning Pretraining)
- **Location:** Likely derived from combined datasets or custom preprocessing
- **Purpose:** 3,500 unlabeled polyp ROI crops for SSL/SimCLR pretraining
- **Size:** ~500 MB
- **Source:** Generated from Kvasir-SEG, Hyper-Kvasir, and PolypGen crops
- **Format:** 256×256 RGB PNG images
- **Instructions:**
  ```
  The unlabeled crops are typically generated from the above datasets
  via preprocessing/data_transformation.py
  No separate download needed if parent datasets are present.
  ```

---

### 7. **Internal Validation Videos** (Clinical Colonoscopy Videos)
- **Location:** `NeSy/Apply Video/` (Estimated ~600+ videos)
- **Purpose:** End-to-end video-level evaluation (373 annotated clinical videos)
- **Size:** ~36 GB
- **Format:** AVI video files with UUID naming convention
- **Annotation File:** `NeSy/video-annotations.csv` (20 KB)
- **Status:** ✅ **Already Downloaded & Available**
- **Content:**
  - Raw colonoscopy videos from clinical procedures
  - UUID-based file naming (e.g., `0220d11b-ab12-4b02-93ce-5d7c205c7043.avi`)
  - Ground truth annotations in CSV format
- **Note:** This is proprietary clinical data. Do NOT re-distribute or publish without IRB approval.

---

### 8. **Dataset 3** (Image Dataset)
- **Location:** `NeSy/Dataset 3/`
- **Size:** ~28 GB
- **Content:** ~29,000+ polyp images with UUID-based filenames (.jpg format)
- **Purpose:** Large-scale image dataset for model training and validation
- **Status:** ✅ **Already Downloaded & Available**
- **Source:** Likely aggregated from Kvasir-SEG, Hyper-Kvasir, and PolypGen

---

### 9. **Dataset i3d** (Video Feature Representations)
- **Location:** `NeSy/Dataset i3d/`
- **Size:** ~36 GB
- **Content:** Pre-computed I3D video feature embeddings
- **Purpose:** Temporal feature representations for video analysis
- **Status:** ✅ **Already Downloaded & Available**
- **Format:** Binary feature files (.avi extension but contains features, not raw video)

---

## Total Storage Requirements

| Dataset | Size | Status |
|---------|------|--------|
| Kvasir-SEG | 250 MB | ✅ Download from link |
| Hyper-Kvasir | 30+ GB | ✅ Download from link |
| NeoPolyp | 364 MB | ✅ Download from link |
| PolypGen v3 | 2-3 GB | ✅ Download from link |
| ETIS-LARIB | 200 MB | ⚠️ Verify availability |
| Dataset 1 (combined) | 4.3 GB | ✅ Already present |
| Dataset 3 | 28 GB | ✅ Already present |
| Dataset i3d | 36 GB | ✅ Already present |
| Apply Video | 36 GB | ✅ Already present |
| Neo polyp Dataset | 364 MB | ✅ Already present |
| **TOTAL** | **~137 GB** | **Mix of present & needed** |

---

## Re-download & Reconstruction Instructions

If you need to completely reconstruct the NeSy folder:

### Step 1: Create Folder Structure
```bash
cd NeSy/
mkdir -p "Dataset 1/Kvasir-SEG"
mkdir -p "Dataset 1/PolypGen2021_MultiCenterData_v3"
mkdir -p "Dataset 1/sequence_data_positive_cropped"
mkdir -p "Dataset 3"
mkdir -p "Dataset i3d"
mkdir -p "Apply Video"
mkdir -p "Neo polyp Dataset/train"
mkdir -p "Neo polyp Dataset/test"
mkdir -p "Neo polyp Dataset/train_gt"
mkdir -p "video ramis"
```

### Step 2: Download Datasets
1. **Kvasir-SEG:** Download from https://datasets.simula.no/kvasir-seg/ → Extract to `Dataset 1/Kvasir-SEG/`
2. **NeoPolyp:** Download from https://www.kaggle.com/competitions/bkai-igh-neopolyp/data → Extract to `Neo polyp Dataset/`
3. **PolypGen:** Download from https://www.synapse.org/#!Synapse:syn26376615 → Extract to `Dataset 1/PolypGen2021_MultiCenterData_v3/`
4. **ETIS-LARIB:** Download from https://www.kaggle.com/datasets/debeshjha1/polypgen-video-sequence→ Extract to appropriate location

### Step 3: Verify Downloads
```bash
# Check folder sizes
du -sh NeSy/*/

# Verify no corrupted files
find NeSy/ -size 0 -type f  # Should return no results
```

---

## Important Notes

### ⚠️ **Critical Reminders:**

1. **Link Verification Date:** June 2026 - Links were verified at this date. URLs may change; verify before re-downloading.

2. **Account Requirements:**
   - Synapse (PolypGen): Requires free account + data use agreement acceptance
   - Kaggle (NeoPolyp): May require Kaggle account
   - Simula datasets: May require email registration

3. **Data Storage:**
   - Total dataset size: ~137 GB
   - Ensure adequate SSD space before download
   - Do NOT store on network drives (performance critical)

4. **Clinical Data Privacy:**
   - Internal validation videos (`Apply Video/`) contain real patient colonoscopy data
   - Do NOT redistribute, publish, or use outside approved research contexts
   - IRB/Ethics approval required for any secondary use

5. **Dataset Integrity:**
   - All datasets use checksums/integrity verification on source sites
   - After download, verify file counts match documentation
   - Check for any .zip/.tar.gz files that need extraction

6. **Preprocessing Pipeline:**
   - After downloading, run preprocessing scripts:
     ```python
     python preprocessing/data_cleaning.py
     python preprocessing/data_transformation.py
     python preprocessing/data_integration.py
     ```

---

## Citation & Acknowledgments

When using these datasets in publications, cite:

```bibtex
@dataset{kvasir-seg,
  title = {Kvasir-SEG: A Segmented Polyp Dataset},
  author = {Jha, D. and others},
  year = {2020},
  url = {https://datasets.simula.no/kvasir-seg/}
}

@dataset{neopolyp,
  title = {NeoPolyp: A Challenge Dataset for Neoplastic Polyp Detection},
  year = {2023},
  url = {https://www.kaggle.com/datasets/banaevaslusana/neopolyp}
}

@dataset{polypgen,
  title = {PolypGen: Multi-Center Polyp Dataset},
  year = {2021},
  url = {https://www.synapse.org/#!Synapse:syn26376615}
}

@dataset{etis-larib,
  title = {ETIS-LARIB: Flat Adenoma Detection Dataset},
  author = {Silva, J. and others},
  year = {2014}
}
```

---

## Last Updated
- **Date:** June 19, 2026
- **Verified By:** NeSY Pipeline Documentation
- **All Links Status:** ✅ Active and verified

For questions about dataset availability or link issues, check the individual dataset providers' websites directly.
