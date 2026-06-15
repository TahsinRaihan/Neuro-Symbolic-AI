# Phase 5 Fixes Applied

## Why These Changes Were Made
The phase 5 pipeline had several mismatches between the intended thesis rules and the code that actually generated results. The main problems were:

- Final CSV results were using raw detection summaries instead of the consensus decision.
- MedSAM was effectively acting as a dependent ROI step instead of an independent detector.
- Temporal consensus was accepting too-short tracks.
- Masked ROI handling was not strict enough, so background pixels could still influence SSL and symbolic reasoning.
- Report generation was reading incomplete metadata and was not receiving segmentation data.
- The polyp-type classifier assumed the wrong color space and used a split that was not group-aware.

## What Changed

### `phase5_video_inference.py`
- Switched Rule 1 to the smarter MedSAM prompt path guided by YOLO/RT-DETR overlap, with a strict grid fallback.
- Stopped double-converting already RGB frames back into RGB.
- Added consensus metadata to the detections object so reports and CSV output use one source of truth.
- Made the final prediction extractor consensus-driven instead of raw-box-count-driven.
- Routed symbolic reasoning through a pure ROI cropped from the MedSAM mask before SSL feature extraction and type classification.
- Updated the report metadata so Rule 1, Rule 2, Rule 4, and Rule 5 reflect the actual pipeline behavior.
- Added a final IoU-based unique-polyp deduplication pass before report serialization so the PDF shows one representative entry per physical polyp instead of repeated temporal fragments.

Why: this fixes the CSV anomaly, aligns the final verdict with consensus, makes Rule 4 and Rule 5 actually use the polyp ROI instead of background pixels, and prevents MedSAM from locking onto colon folds or glare when the blind grid lands in the wrong place.

Why the extra dedup pass was needed: temporal track merging reduces obvious duplicates, but the report was still counting fragmented views of the same lesion as separate tracks when the camera motion or track gaps prevented a full merge. The final dedup stage collapses those repeated views into one unique-polyp record.

### `rules_engine_5.py`
- Added `rule1_medsam_independent_detection()` using grid-point prompts without YOLO/RT-DETR boxes.
- Raised temporal consensus requirements to 20 consecutive frames by default.
- Tightened track merging so unrelated tracks are less likely to be merged together.

Why: this implements the intended MedSAM independence and enforces a real sustained consensus rule instead of a short flicker-based track.

### `polyp_feature_extractor.py`
- Added `crop_pure_roi_from_mask()`.
- Made `extract_all_features()` use the mask-filtered ROI when a segmentation mask is available.

Why: this removes background influence from redness, radius, texture, vessel visibility, and color features.

### `medical_report_generator.py`
- Passed segmentation data into the report generator.
- Removed the extra BGR to RGB conversion on images that are already RGB.
- Fixed the consensus count so it reads the actual number of consensus runs.

Why: this keeps the PDF report consistent with the real segmentation data and avoids corrupting the displayed ROI colors.

### `polyp_type_classifier.py`
- Changed ROI feature extraction to use RGB color conversions.
- Replaced the random split in `train_from_annotations()` with a `GroupShuffleSplit` by video ID.

Why: the classifier now matches the frame format used by the pipeline and respects the video-level 70-30 split.

### `symbolic_reasoning_integrator.py`
- Changed the 70-30 split in `implement_70_30_split()` to `GroupShuffleSplit` by video ID.

Why: this makes the symbolic reasoning split explicitly group-aware and consistent with the applied dataset requirement.

## Validation Performed
- Syntax checks passed for all edited Python files.
- `py_compile` succeeded for:
  - `phase5_video_inference.py`
  - `rules_engine_5.py`
  - `polyp_feature_extractor.py`
  - `medical_report_generator.py`
  - `polyp_type_classifier.py`
  - `symbolic_reasoning_integrator.py`

## Notes
- I did not run the full end-to-end dataset inference job in this session.
- The pipeline is now syntactically valid and the core rule wiring is corrected, but full clinical/batch verification still depends on running the phase 5 script on the real video set.
