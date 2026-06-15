# Neuro-Symbolic AI for Optical Biopsy-Level Polyp Diagnosis

**Autonomous Classification, Risk Prediction, and Interpretable Reasoning in Endoscopic Video**

B.Sc. Thesis — Department of Computer Science and Engineering, Brac University, October 2025

---

## Authors

| Name | Student ID |
|---|---|
| Tahsin Raihan Robbani | 22201344 |
| Md. Ahanaf Hasan Rivan | 22201733 |
| Samura Huda | 22201205 |
| Atkiya Farzana | 22201595 |
| Iftekhar Hossain Rahi | 22201168 |

**Supervisor:** Dr. Swakkhar Shatabda, Professor, Department of CSE, Brac University  
**Co-Supervisor:** Dr. Md. Golam Rabiul Alam, Professor, Department of CSE, Brac University

---

## Abstract

During standard colonoscopy procedures, doctors routinely encounter polyps but cannot immediately classify them or reliably predict their malignancy risk. Determining this typically requires a physical biopsy, which is time-consuming, expensive, and stressful for the patient. This thesis introduces a Neuro-Symbolic AI (NeSy) assistant designed to achieve optical biopsy-level polyp diagnosis directly from live colonoscopy video, delivering immediate and interpretable risk predictions before any physical biopsy is performed.

The pipeline operates in five integrated phases. A three-model consensus engine combining YOLOv8, RT-DETR, and MedSAM2 handles real-time detection and segmentation. A self-supervised Vision Transformer (ViT) encoder trained via SimCLR extracts 384-dimensional semantic embeddings from unlabeled polyp crops. These embeddings are fused with 60 hand-crafted clinical biomarkers (redness, vessel visibility, texture entropy) to form a 444-dimensional Fact Vector per lesion. Clinical-First K-Means clustering (k=8) groups these vectors into morphological prototypes, and a Mixture of Experts (MoE) of eight cluster-specific decision tree classifiers produces risk predictions. A final symbolic reasoning layer cross-checks all findings against NICE 2017 and ESGE clinical guidelines to produce calibrated risk scores (HIGH, MEDIUM, LOW) with human-readable, auditable explanations.

Evaluated on 349 annotated clinical colonoscopy videos, the system achieves 100% detection sensitivity, 91.98% any-polyp match accuracy, 90.26% video-level match accuracy, 84.45% risk match accuracy, and 93.2% HIGH-risk sensitivity. The architecture addresses the black-box problem of conventional deep learning by making every risk prediction traceable to specific biomarker values and explicit clinical reasoning steps.

**Keywords:** Neuro-Symbolic AI, Colonoscopy, Polyp Detection, Mixture of Experts, Segmentation, Calibration, Optical Biopsy, Self-Supervised Learning, SimCLR, Interpretable AI

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Pipeline Phases](#pipeline-phases)
- [Datasets](#datasets)
- [Results](#results)
- [Novelty](#novelty)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Reproducing Results](#reproducing-results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Colorectal cancer accounts for approximately 930,000 deaths worldwide annually, nearly 1 in 10 of all cancer-related deaths. The five-year survival rate exceeds 90% when detected early but drops to approximately 14% at late-stage diagnosis. Colonoscopy is the gold standard screening tool, yet current AI systems suffer from three critical gaps:

1. They function as black boxes, providing predictions without clinically auditable reasoning.
2. They produce miscalibrated, overconfident probabilities that cannot be trusted in practice.
3. They do not enforce international clinical guidelines (NICE, ESGE, ASGE PIVI), making outputs clinically inconsistent.

This work addresses all three gaps in a single unified pipeline.

---

## Problem Statement

Real-time optical biopsy during live colonoscopy requires an AI assistant that can simultaneously:

- Achieve robust, stable polyp detection across full-length video rather than curated static images.
- Maintain frame-lock on a polyp despite motion blur, rapid lighting shifts, and camera glare.
- Eliminate the black-box deficit by providing explicit, traceable clinical reasoning for every prediction.
- Adhere to established international medical guidelines without post-hoc patching.

---

## System Architecture

The system follows a Crop-to-Concept pipeline that transforms raw colonoscopy video into structured, guideline-aligned clinical recommendations.

```
Raw Colonoscopy Video
        |
        v
Phase 1: Multi-Model Detection and Segmentation
    YOLOv8 + RT-DETR + MedSAM2 (Three-Model Consensus Voting)
        |
        v
ROI Crop (256x256 RGB, foreground-masked)
        |
        +----------------------------+
        |                            |
        v                            v
Stream A: SSL Encoder           Stream B: Biomarker Engine
ViT-Small via SimCLR            60 hand-crafted clinical features
384-dimensional embedding       (HSV histograms, LBP texture,
                                 vessel visibility, redness,
                                 Laplacian entropy, shape)
        |                            |
        +----------------------------+
                     |
                     v
Phase 3: 444-Dimensional Fact Vector Assembly
                     |
                     v
Phase 3: Clinical-First K-Means Clustering (k=8)
    Eight morphological prototypes
                     |
                     v
Phase 4: Mixture of Experts (Cluster-Specific Decision Trees)
    + MLP Calibrator (Categorical Cross-Entropy + Focal Loss)
                     |
                     v
Phase 5: Symbolic Reasoning Layer
    NICE 2017 criteria + ESGE guidelines + clinical conservatism override
                     |
                     v
Risk Score: HIGH / MEDIUM / LOW
+ NICE Type (1/2/3)
+ Calibrated confidence
+ Human-readable biomarker rationale
+ Video-level PDF diagnostic report
```

---

## Pipeline Phases

### Phase 1: Multi-Model Detection, Segmentation, and Consensus Voting

Three object detection and segmentation models run in parallel:

- **YOLOv8** for high-speed bounding box detection (25-40+ FPS)
- **RT-DETR (ResNet-50)** for transformer-based detection
- **MedSAM2** for memory-attention-based video segmentation

A detection is accepted only when all three models agree using an Intersection-over-Minimum (IoM) threshold >= 0.45. This redundancy achieves 100% detection sensitivity with zero false negatives across the 349-video evaluation cohort.

Temporal track aggregation stitches detections across consecutive frames (IoU >= 0.45, frame gap < 20), deduplicating overlapping tracks into single consolidated diagnostic statements per polyp.

### Phase 2: Self-Supervised Visual Representation Learning

A ViT-Small encoder is trained on 3,500 unlabeled polyp ROI crops using:

- **Contrastive NT-Xent objective** (SimCLR) to pull augmented views of the same crop together and push distinct crops apart
- **L2 reconstruction regularization** to prevent representational collapse

Augmentations include random flipping, rotation, color jitter, and Gaussian blur. The encoder is frozen post-training and outputs a 384-dimensional CLS-token embedding per ROI.

### Phase 3: Feature Extraction, Fact Vector Assembly, and Clustering

**Stream A** (deep features): 384-dimensional ViT embedding  
**Stream B** (clinical biomarkers, 60 dimensions):

| Biomarker Group | Features |
|---|---|
| Color | 16-bin HSV histograms (H, S, V channels), normalized redness (R-G)/(R+G+B) |
| Texture | Local Binary Patterns (LBP), Laplacian variance, edge density, entropy |
| Vascularity | Pixel-density vessel visibility score in HSV space, saturation variance |
| Shape | Normalized radius, relative area, circularity |

Both streams are concatenated into a 444-dimensional Fact Vector, StandardScaler-normalized on the training split, then clustered using Clinical-First K-Means (k=8). PCA validation confirms eight well-separated prototypes corresponding to clinically recognized morphological archetypes.

### Phase 4: Cluster-Aware Mixture of Experts and MLP Calibration

Eight independent decision tree classifiers, one per K-Means cluster, are trained exclusively on the 60-dimensional symbolic biomarker subspace:

- Maximum depth: 5
- Minimum samples per leaf: 10
- Split criterion: Gini impurity
- Class balancing: automatic

At inference, the assigned cluster routes the Fact Vector to its dedicated expert. Edge-case fallback defaults to majority vote across all eight experts with a 5% confidence penalty.

An MLP calibrator (input: 444 fact dimensions + 8-dimensional cluster indicator; hidden layers: 256 and 128 ReLU units with BatchNorm and Dropout) is trained jointly using Categorical Cross-Entropy and Focal Loss. Confidence bounds are derived using Youden-index ROC optimization, and raw outputs are clipped to this empirical range to eliminate structural overconfidence.

### Phase 5: Symbolic Reasoning, NICE Criteria Evaluation, and Clinical Classification

Three reasoning paths are combined in parallel:

1. **Cluster expert tree** outputs a primary risk tier (HIGH/MEDIUM/LOW)
2. **Feature heuristic** computes a soft voting score from data-driven thresholds:
   - Redness > threshold: +0.3
   - Vessel visibility > threshold: +0.3
   - Laplacian texture > threshold: +0.2
   - Normalized radius > threshold: +0.2
   - Saturation mean > threshold: +0.1
3. **NICE 2017 criteria** evaluated programmatically:
   - Score 0: NICE Type 1, LOW RISK (standard surveillance)
   - Score 1-2: NICE Type 2, MEDIUM RISK (endoscopic resection)
   - Score 3: NICE Type 3, HIGH RISK (urgent biopsy or referral)

A clinical conservatism override ensures: if any reasoning path flags HIGH RISK, the final output is universally upgraded to HIGH RISK.

The pipeline exports a structured PDF report per video including polyp type, risk tier, NICE type, calibrated confidence score, all biomarker values, and a natural-language clinical rationale.

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| Kvasir-SEG | Detection and segmentation training | 1,000 annotated images |
| Hyper-Kvasir | Video-level temporal validation | Large-scale video repository |
| NeoPolyp | Risk mask ground truth (red = neoplastic, green = non-neoplastic) | 1,000 images with pixel-level masks |
| PolypGen | Multi-center cross-site generalization validation | Multi-center, multi-modal sequences |
| ETIS-LARIB | Flat adenoma detection validation | High-resolution frames |
| Unlabeled ROI crops | Self-supervised encoder pretraining | 3,500 unlabeled crops |
| Internal validation cohort | End-to-end video-level evaluation | 373 annotated clinical videos |

All datasets used are open-access. No proprietary data dependencies exist.

**Data Preprocessing:**

- All images resized to 256x256 pixels
- Normalized with medical imaging mean and standard deviation
- Blank frames, zero-intensity pixels, and empty bounding boxes removed
- Boundary clamping for coordinate overflow
- NeoPolyp masks harmonized (red pixels = HIGH RISK, green pixels = LOW RISK)
- Clinical artifact exclusion (excessive lens bubbling, instrument occlusion)
- Feature vectors validated for NaN and infinity values before clustering

**Final preprocessed dataset composition:**

- Detection training set: 3,500 images (70/30 train/validation split)
- SSL pretraining: 99,000 unlabeled images
- Classification and clustering: 1,000 labeled images (binary HIGH/LOW from NeoPolyp masks, mapped to NICE types 1-3)
- End-to-end video evaluation: 373 clinical colonoscopy videos with ground-truth CSV annotations

---

## Results

### Detection Performance

| Metric | Value |
|---|---|
| Detection Sensitivity | 100% |
| True Positives | 284 |
| False Negatives | 0 |
| False Positives | 62 |
| Precision | 82.1% |
| F1 Score (Detection) | 90.2% |

### Classification and Risk Assessment

| Metric | Value |
|---|---|
| Any-Polyp Match Accuracy | 91.98% |
| Video-Level Match Accuracy | 90.26% |
| Risk Match Accuracy | 84.45% |
| HIGH-Risk Sensitivity | 93.2% |

### Detection Curves

| Curve | Value |
|---|---|
| mAP@0.5 (YOLO) | ~1.00 |
| mAP@0.5-95 (YOLO) | >0.95 |
| mAP@0.5 (RT-DETR) | 1.00 |
| Area Under Precision-Recall Curve | 0.953 |
| Peak F1 Score | 0.95 at confidence 0.553 |

### Per-Class Performance

| Class | N | Any-Match | Risk Match |
|---|---|---|---|
| Lifted Polyp | 105 | 94.3% | 84.8% |
| Adenomatous Polyp | 66 | 100% | 34.8% |
| Normal Mucosa | 65 | 72.3% | 69.2% |
| Resected Polyp | 44 | 100% | 100% |
| Colitis | 14 | 100% | 100% |
| Bleeding Polyp | 11 | 100% | 100% |
| Malignant Polyp | 10 | 100% | 90% |
| Lateral Spreading Tumor | 8 | 100% | 100% |
| Flat Polyp | 3 | 100% | 100% |
| Large Polyp | 2 | 100% | 100% |
| Serrated Polyp | 2 | 50% | 50% |
| Pedunculated Polyp | 1 | 100% | 100% |
| Villous Polyp | 1 | 100% | 100% |

The overall any-polyp match accuracy of 90.26% meets the ASGE PIVI threshold of 90% required for clinical optical diagnosis adoption.

### Self-Supervised Learning

- Contrastive loss converged from approximately 3.9750 to 3.9715 over 100 epochs with consistent downward trend beneath oscillatory noise
- Reconstruction loss reduced by approximately 70% with no oscillation, confirming no representational collapse
- Total loss maintained a clear downward trend across all 100 epochs

### Clustering

- K-Means (k=8) on 444-dimensional Fact Vectors from 1,000 NeoPolyp images
- PCA confirms well-separated cluster boundaries: PC1 captures 54.80% variance, PC2 captures 4.04% variance
- Eight prototypes correspond to clinically recognizable morphological archetypes

---

## Novelty

This work introduces the following original contributions to the field:

**1. First Fully Integrated Neuro-Symbolic Pipeline for Colonoscopy Video**  
No prior system unifies multi-model detection, self-supervised visual representation, unsupervised morphological prototyping, mixture of experts classification, training-time calibration, guideline-grounded symbolic reasoning, and utility-aware action mapping in a single operational pipeline for endoscopic video.

**2. Three-Model Consensus Voting for Zero-Miss Detection**  
A parallel YOLOv8 + RT-DETR + MedSAM2 voting engine with Intersection-over-Minimum (IoM >= 0.45) threshold achieves 100% video-level sensitivity across 349 clinical videos, eliminating the structural fragility of single-model inference.

**3. Self-Supervised Morphological Prototyping Without Label Dependency**  
A ViT-Small encoder trained via SimCLR contrastive loss on 3,500 unlabeled polyp crops produces high-quality 384-dimensional semantic embeddings without any human annotation, fused with 60 explicit clinical biomarkers into a 444-dimensional Fact Vector.

**4. Clinical-First K-Means Clustering and Cluster-Aware Mixture of Experts**  
K-Means (k=8) on Fact Vectors produces data-driven morphological prototypes. Eight independent decision tree experts, one per cluster, replace a global classifier, addressing class imbalance for rare but high-risk lesions such as malignant adenomas and lateral spreading tumors.

**5. Training-Time Confidence Calibration Without Post-Hoc Scaling**  
An MLP calibrator is trained jointly with the routing mechanism using coupled Categorical Cross-Entropy and Focal Loss. Confidence bounds are set by Youden-index ROC analysis, producing a confirmed monotonic confidence-accuracy relationship at the video level.

**6. Computable Guideline-Consistent Symbolic Reasoning**  
NICE 2017 classification criteria are translated into explicit, computable symbolic rules operating on extracted biomarker features. A clinical conservatism override ensures predictions can never violate the most conservative safe clinical path.

**7. Video-Level Temporal Track Aggregation**  
Five-rule track aggregation system stitches lesion detections across successive frames using IoU >= 0.45, sustained agreement across at least 20 consecutive frames, and spatial merge thresholds, producing one unified diagnostic report per polyp per video.

**8. Structured PDF Diagnostic Report**  
First AI output format to combine multi-model detection evidence, NeoPolyp-calibrated feature measurements, NICE-grounded risk scoring with explicit boundary comparisons, and guideline-sourced natural-language clinical recommendations in a single unified document per video.

---

## Requirements

### Hardware

- GPU: NVIDIA RTX 4080 Ti Super, A100, or equivalent
- RAM: Minimum 64 GB
- Storage: Minimum 1 TB SSD
- Video processing: >= 25 FPS real-time inference capability

### Software

```
Python >= 3.12.1
PyTorch
TensorFlow
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
NetCal
PyCalib
```

### Models and Frameworks

- YOLOv8 (Ultralytics)
- RT-DETR (ResNet-50 backbone)
- MedSAM2 (Meta AI video segmentation)
- ViT-Small (Vision Transformer, SimCLR pretraining)
- DeepProbLog (symbolic logic integration)
- Decision Tree Classifier (scikit-learn)
- NICE Classifier (custom implementation)
- ESGE validation rules (custom implementation)

---

## Repository Structure

```
.
├── data/
│   ├── kvasir_seg/              # Kvasir-SEG detection training images
│   ├── neopolyp/                # NeoPolyp risk mask images
│   ├── polypgen/                # PolypGen multi-center sequences
│   ├── etis_larib/              # ETIS-LARIB validation frames
│   ├── hyper_kvasir/            # Hyper-Kvasir video repository
│   ├── unlabeled_crops/         # 3,500 unlabeled ROI crops for SSL
│   └── validation_videos/       # 373 annotated clinical videos + ground truth CSV
│
├── preprocessing/
│   ├── data_cleaning.py         # Blank frame removal, boundary clamping, mask harmonization
│   ├── data_transformation.py   # Resize to 256x256, normalization, ROI extraction
│   ├── data_integration.py      # Biomarker + SSL embedding concatenation
│   └── data_reduction.py        # PCA, K-Means clustering, semantic filtering
│
├── phase1_detection/
│   ├── yolov8_train.py          # YOLOv8 training (100 epochs)
│   ├── rtdetr_train.py          # RT-DETR training (150 epochs)
│   ├── medsam2_inference.py     # MedSAM2 video segmentation
│   ├── consensus_voting.py      # Three-model IoM consensus engine
│   └── temporal_tracking.py     # Track aggregation and deduplication
│
├── phase2_ssl/
│   ├── simclr_train.py          # ViT-Small SimCLR pretraining (100 epochs)
│   ├── augmentations.py         # Random flip, rotation, color jitter, Gaussian blur
│   └── encoder_inference.py     # Frozen encoder for 384-dim embedding extraction
│
├── phase3_features/
│   ├── biomarker_engine.py      # 60 clinical biomarker extraction (HSV, LBP, shape)
│   ├── fact_vector_assembly.py  # Concatenation and StandardScaler normalization
│   ├── kmeans_clustering.py     # Clinical-First K-Means (k=8) training
│   └── pca_validation.py        # PCA visualization of cluster separation
│
├── phase4_moe/
│   ├── decision_tree_experts.py # Eight cluster-specific decision tree classifiers
│   ├── mlp_calibrator.py        # MLP calibrator with Focal Loss + CCE
│   ├── routing_mechanism.py     # Fact Vector to cluster to expert routing
│   └── confidence_clipping.py  # Youden-index ROC bound derivation
│
├── phase5_symbolic/
│   ├── nice_rules.py            # NICE 2017 criteria as computable symbolic rules
│   ├── esge_rules.py            # ESGE guideline enforcement
│   ├── feature_heuristics.py    # Soft voting score computation
│   ├── conservatism_override.py # Clinical conservatism safety clause
│   └── risk_classification.py  # Final HIGH/MEDIUM/LOW assignment
│
├── reporting/
│   ├── pdf_report_generator.py  # Structured PDF diagnostic report per video
│   ├── frame_montage.py         # Multi-frame visual evidence montage
│   └── accuracy_logger.py       # Video-level consensus match and risk tier logging
│
├── evaluation/
│   ├── video_inference.py       # End-to-end inference on 373-video validation cohort
│   ├── metrics.py               # Any-match, video-level, risk-match, sensitivity
│   ├── confusion_matrix.py      # Binary detection and per-class confusion matrices
│   ├── calibration_curves.py    # Precision-confidence, recall-confidence, F1-confidence
│   └── ablation_studies.py      # Symbolic layer, calibration, uncertainty buffer ablation
│
├── configs/
│   ├── yolov8_config.yaml
│   ├── rtdetr_config.yaml
│   ├── ssl_config.yaml
│   └── pipeline_config.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_detection_training.ipynb
│   ├── 03_ssl_pretraining.ipynb
│   ├── 04_clustering_analysis.ipynb
│   ├── 05_moe_training.ipynb
│   ├── 06_symbolic_reasoning.ipynb
│   └── 07_full_pipeline_evaluation.ipynb
│
├── outputs/
│   ├── model_checkpoints/       # Saved model weights
│   ├── cluster_models/          # K-Means and eight decision tree experts
│   ├── pdf_reports/             # Generated diagnostic reports
│   └── evaluation_logs/         # Accuracy and metric logs
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Reproducing Results

### 1. Install Dependencies

```bash
git clone https://github.com/<your-org>/neuro-symbolic-polyp-diagnosis.git
cd neuro-symbolic-polyp-diagnosis
pip install -r requirements.txt
```

### 2. Download Datasets

Download the following datasets and place them in the `data/` directory:

- Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
- Hyper-Kvasir: https://datasets.simula.no/hyper-kvasir/
- NeoPolyp: available via the NeoPolyp challenge repository
- PolypGen: https://www.synapse.org/Synapse:syn26376615
- ETIS-LARIB: https://polyp.grand-challenge.org/EtisLarib/

### 3. Preprocess Data

```bash
python preprocessing/data_cleaning.py
python preprocessing/data_transformation.py
python preprocessing/data_integration.py
```

### 4. Phase 1: Train Detection Models

```bash
# Train YOLOv8 (100 epochs)
python phase1_detection/yolov8_train.py --config configs/yolov8_config.yaml

# Train RT-DETR (150 epochs)
python phase1_detection/rtdetr_train.py --config configs/rtdetr_config.yaml
```

### 5. Phase 2: SSL Pretraining

```bash
python phase2_ssl/simclr_train.py --config configs/ssl_config.yaml
```

This trains the ViT-Small encoder on 3,500 unlabeled polyp ROI crops for 100 epochs.

### 6. Phase 3: Feature Extraction and Clustering

```bash
python phase3_features/biomarker_engine.py
python phase3_features/fact_vector_assembly.py
python phase3_features/kmeans_clustering.py --k 8
python phase3_features/pca_validation.py
```

### 7. Phase 4: Train Mixture of Experts and Calibrator

```bash
python phase4_moe/decision_tree_experts.py
python phase4_moe/mlp_calibrator.py
python phase4_moe/confidence_clipping.py
```

### 8. Phase 5: Symbolic Reasoning Configuration

```bash
python phase5_symbolic/nice_rules.py --validate
python phase5_symbolic/esge_rules.py --validate
```

### 9. End-to-End Video Evaluation

```bash
python evaluation/video_inference.py \
    --video_dir data/validation_videos/ \
    --ground_truth data/validation_videos/ground_truth.csv \
    --output_dir outputs/pdf_reports/
```

### 10. Compute Metrics

```bash
python evaluation/metrics.py --results_dir outputs/evaluation_logs/
python evaluation/confusion_matrix.py
python evaluation/calibration_curves.py
```

---

## Limitations

- **Non-polyp vascular misfires:** Complex anatomical structures such as esophageal varices, duodenal papilla, and surgical anastomoses can trigger false positive detections due to high vessel visibility scores. These are filtered by the symbolic layer but introduce 62 false positives in the evaluation cohort.

- **Temporal frame bias:** The pipeline evaluates a representative static frame rather than a continuous multi-frame sequence. Pre-injection frames captured before submucosal dye injection display natural vascular prominence that the system may flag incorrectly.

- **Serrated polyp data scarcity:** Only 2 serrated polyp examples exist in the evaluation cohort, making the reported 50% accuracy for this class statistically volatile and not representative of model capability.

- **Adenomatous polyp risk annotation gap:** Many adenomatous polyp ground-truth records lack explicit risk level strings. These are automatically treated as non-matches, penalizing the risk match metric despite correct lesion identification.

- **White-light only:** Current classification relies exclusively on white-light endoscopy features. Formal "resect and discard" clinical adoption requires narrow-band imaging (NBI) validation.

- **Procedure-level ground truth:** The validation annotation is one label per video. Videos containing multiple polyp types are annotated for only the primary type, introducing an evaluation ceiling unrelated to model capability.

---

## Future Work

- **Temporal sequence modeling:** Replace static representative-frame selection with a bidirectional LSTM or temporal attention mechanism over sequential Fact Vectors to track procedural phase and dye injection transitions.

- **Hard negative mining:** Retrain YOLOv8 and RT-DETR with dedicated negative frame banks containing esophageal varices, instrumentation artifacts, and duodenal papillae to reduce anatomical false positives.

- **Multi-modal imaging support:** Extend the SSL encoder to process Narrow-Band Imaging (NBI) and chromoendoscopy crops alongside white-light video.

- **Granular visual clustering:** Increase k in K-Means clustering to enable finer-grained expert specialization for subtle sub-morphologies.

- **Live interactive software:** Develop a real-time second-screen overlay displaying bounding indicators, risk tier, and management prompts at clinical frame rates during active colonoscopy procedures.

- **Prospective multi-center clinical validation:** Conduct large-scale randomized trials in real hospital endoscopy environments to validate the system's impact on adenoma detection rates and to meet regulatory requirements for medical device software deployment (FDA 510(k), CE marking under EU AI Act 2024/1689, DGDA Bangladesh).

- **Per-polyp biopsy-proven histology ground truth:** Replace procedure-level annotation with spatially and temporally matched per-polyp histological ground truth as defined by the ASGE PIVI optical biopsy validation standard.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{robbani2026neurosymbolic,
  title     = {Neuro-Symbolic AI for Optical Biopsy-Level Polyp Diagnosis:
               Autonomous Classification, Risk Prediction, and Interpretable
               Reasoning in Endoscopic Video},
  author    = {Robbani, Tahsin Raihan and Rivan, Md. Ahanaf Hasan and
               Huda, Samura and Farzana, Atkiya and Rahi, Iftekhar Hossain},
  year      = {2026},
  school    = {Brac University},
  department= {Department of Computer Science and Engineering},
  type      = {B.Sc. Thesis},
  month     = {June},
  supervisor= {Dr. Swakkhar Shatabda},
  cosupervisor = {Dr. Md. Golam Rabiul Alam}
}
```

---

## Acknowledgements

The authors thank Dr. Swakkhar Shatabda and Dr. Md. Golam Rabiul Alam for their supervision and guidance throughout this research. The authors also acknowledge the open-access dataset providers: Simula Research Laboratory (Kvasir-SEG, Hyper-Kvasir), the NeoPolyp challenge organizers, the PolypGen consortium, and the ETIS-LARIB benchmark maintainers.

---

## License

Copyright 2026, Brac University. All rights reserved.

This repository is made available for academic and research purposes. Commercial use, redistribution, or clinical deployment without explicit written permission from the authors and Brac University Department of Computer Science and Engineering is not permitted.

The clinical decision logic encoded in this system, including NICE criteria thresholds and ESGE guideline rules, is derived from published international guidelines and is intended for research exploration only. This system is not a certified medical device and must not be used for clinical diagnosis or patient management without appropriate regulatory approval.
