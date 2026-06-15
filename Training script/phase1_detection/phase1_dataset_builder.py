# -*- coding: utf-8 -*-
"""Shared phase 1 dataset preparation utilities."""

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
MIN_CONTOUR_AREA = 20


@dataclass(frozen=True)
class Phase1DatasetPaths:
    root: Path
    train_images: Path
    train_labels: Path
    train_masks: Path
    val_images: Path
    val_labels: Path
    val_masks: Path
    data_yaml: Path
    manifest_path: Path


def _safe_token(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    cleaned = cleaned.strip('_.-')
    return cleaned or 'sample'


def _target_token(value: str) -> str:
    digest = hashlib.sha1(value.encode('utf-8', errors='ignore')).hexdigest()[:8]
    return f'{_safe_token(value)}__{digest}'


def _candidate_stems(stem: str) -> List[str]:
    candidates = [stem]
    if stem.endswith('_mask'):
        candidates.append(stem[:-5])
    else:
        candidates.append(f'{stem}_mask')

    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def find_first_matching_file(directory: Path, stem: str, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> Optional[Path]:
    if not directory.exists():
        return None

    for candidate_stem in _candidate_stems(stem):
        for extension in extensions:
            candidate = directory / f'{candidate_stem}{extension}'
            if candidate.exists():
                return candidate

    return None


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    try:
        os.link(str(source), str(target))
    except OSError:
        shutil.copy2(str(source), str(target))


def _write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')


def _load_image(image_path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image


def _mask_has_foreground(mask_path: Path) -> bool:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return bool(np.any(mask > 10))


def _bbox_to_yolo_line(x1: float, y1: float, x2: float, y2: float, image_width: int, image_height: int) -> str:
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))

    box_width = max(x2 - x1, 0.0)
    box_height = max(y2 - y1, 0.0)
    if image_width <= 0 or image_height <= 0 or box_width <= 0 or box_height <= 0:
        return ''

    x_center = max(0.0, min(1.0, ((x1 + x2) / 2.0) / image_width))
    y_center = max(0.0, min(1.0, ((y1 + y2) / 2.0) / image_height))
    norm_width = max(0.0, min(1.0, box_width / image_width))
    norm_height = max(0.0, min(1.0, box_height / image_height))

    if norm_width == 0.0 or norm_height == 0.0:
        return ''

    return f'0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}'


def _bbox_text_to_yolo_lines(bbox_path: Path, image_width: int, image_height: int) -> List[str]:
    try:
        text = bbox_path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return []

    yolo_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) < 4:
            continue

        try:
            x1, y1, x2, y2 = map(float, tokens[-4:])
        except ValueError:
            continue

        yolo_line = _bbox_to_yolo_line(x1, y1, x2, y2, image_width, image_height)
        if yolo_line:
            yolo_lines.append(yolo_line)

    return yolo_lines


def _mask_to_yolo_lines(mask_path: Path, image_width: int, image_height: int) -> List[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines: List[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        yolo_line = _bbox_to_yolo_line(x, y, x + width, y + height, image_width, image_height)
        if yolo_line:
            yolo_lines.append(yolo_line)

    if not yolo_lines and np.any(binary):
        ys, xs = np.where(binary > 0)
        if xs.size > 0 and ys.size > 0:
            yolo_line = _bbox_to_yolo_line(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, image_width, image_height)
            if yolo_line:
                yolo_lines.append(yolo_line)

    return yolo_lines


def _build_annotation_lines(image: np.ndarray, mask_path: Optional[Path], bbox_path: Optional[Path]) -> Tuple[List[str], str, bool]:
    image_height, image_width = image.shape[:2]

    if bbox_path is not None and bbox_path.exists():
        bbox_lines = _bbox_text_to_yolo_lines(bbox_path, image_width, image_height)
        if bbox_lines:
            return bbox_lines, 'bbox', True

    if mask_path is not None and mask_path.exists():
        mask_lines = _mask_to_yolo_lines(mask_path, image_width, image_height)
        if mask_lines:
            return mask_lines, 'mask', True
        return [], 'mask', _mask_has_foreground(mask_path)

    return [], 'empty', False


def _write_blank_mask(target_mask_path: Path, image: np.ndarray) -> None:
    blank_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    target_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_mask_path), blank_mask)


def _materialize_sample(
    *,
    source_name: str,
    split: str,
    source_type: str,
    source_center: Optional[str],
    source_sequence_id: Optional[int],
    sample_type: str,
    image_path: Path,
    mask_path: Optional[Path],
    bbox_path: Optional[Path],
    output_root: Path,
    records: List[Dict[str, object]],
    summary: DefaultDict[str, int],
) -> None:
    image = _load_image(image_path)
    if image is None:
        return

    target_split_dir = output_root / split
    target_split_dir.mkdir(parents=True, exist_ok=True)

    safe_source_name = _safe_token(source_name)
    target_stem = f'{safe_source_name}__{_target_token(image_path.stem)}'

    target_image_path = target_split_dir / 'images' / f'{target_stem}{image_path.suffix.lower()}'
    target_label_path = target_split_dir / 'labels' / f'{target_stem}.txt'
    if mask_path is not None and mask_path.exists():
        target_mask_suffix = mask_path.suffix.lower()
    else:
        target_mask_suffix = '.png'
    target_mask_path = target_split_dir / 'masks' / f'{target_stem}{target_mask_suffix}'

    _link_or_copy(image_path, target_image_path)

    if mask_path is not None and mask_path.exists():
        _link_or_copy(mask_path, target_mask_path)
    else:
        _write_blank_mask(target_mask_path, image)

    annotation_lines, annotation_source, positive = _build_annotation_lines(image, mask_path, bbox_path)
    _write_text(target_label_path, '\n'.join(annotation_lines))

    summary['total'] += 1
    summary[f'{split}_total'] += 1
    summary['positive' if positive else 'negative'] += 1
    summary[f'{source_name}_total'] += 1

    records.append({
        'source': source_name,
        'source_type': source_type,
        'split': split,
        'sample_type': sample_type,
        'center': source_center,
        'sequence_id': source_sequence_id,
        'positive': positive,
        'annotation_source': annotation_source,
        'source_image': image_path.as_posix(),
        'source_mask': mask_path.as_posix() if mask_path is not None else None,
        'source_bbox': bbox_path.as_posix() if bbox_path is not None else None,
        'target_image': target_image_path.as_posix(),
        'target_mask': target_mask_path.as_posix(),
        'target_label': target_label_path.as_posix(),
        'target_stem': target_stem,
    })


def _iter_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([
        file_path for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ])


def _ingest_kvasir(source_root: Path, output_root: Path, records: List[Dict[str, object]], summary: DefaultDict[str, int]) -> None:
    kvasir_root = source_root / 'Kvasir-SEG'
    masks_dir = kvasir_root / 'masks'

    for split in ('train', 'val'):
        image_dir = kvasir_root / f'{split}_images'
        for image_path in _iter_files(image_dir):
            mask_path = find_first_matching_file(masks_dir, image_path.stem, IMAGE_EXTENSIONS)
            _materialize_sample(
                source_name=f'kvasir_{split}',
                split=split,
                source_type='kvasir',
                source_center=None,
                source_sequence_id=None,
                sample_type='full_frame',
                image_path=image_path,
                mask_path=mask_path,
                bbox_path=None,
                output_root=output_root,
                records=records,
                summary=summary,
            )


def _ingest_polypgen_centers(source_root: Path, output_root: Path, records: List[Dict[str, object]], summary: DefaultDict[str, int]) -> None:
    polypgen_root = source_root / 'PolypGen2021_MultiCenterData_v3' / 'PolypGen2021_MultiCenterData_v3'

    for center in range(1, 7):
        center_root = polypgen_root / f'data_C{center}'
        image_dir = center_root / f'images_C{center}'
        mask_dir = center_root / f'masks_C{center}'
        bbox_dir = center_root / f'bbox_C{center}'
        split = 'train' if center < 6 else 'val'

        for image_path in _iter_files(image_dir):
            mask_path = find_first_matching_file(mask_dir, image_path.stem, IMAGE_EXTENSIONS)
            bbox_path = find_first_matching_file(bbox_dir, image_path.stem, ('.txt',))
            _materialize_sample(
                source_name=f'polypgen_center_C{center}',
                split=split,
                source_type='polypgen_center',
                source_center=f'C{center}',
                source_sequence_id=None,
                sample_type='full_frame',
                image_path=image_path,
                mask_path=mask_path,
                bbox_path=bbox_path,
                output_root=output_root,
                records=records,
                summary=summary,
            )


def _ingest_polypgen_sequences(source_root: Path, output_root: Path, records: List[Dict[str, object]], summary: DefaultDict[str, int]) -> None:
    sequence_root = source_root / 'PolypGen2021_MultiCenterData_v3' / 'PolypGen2021_MultiCenterData_v3' / 'sequenceData'

    positive_root = sequence_root / 'positive'
    negative_root = sequence_root / 'negativeOnly'

    for sequence_id in range(1, 24):
        split = 'train' if sequence_id <= 18 else 'val'

        positive_sequence_root = positive_root / f'seq{sequence_id}'
        image_dir = positive_sequence_root / f'images_seq{sequence_id}'
        mask_dir = positive_sequence_root / f'masks_seq{sequence_id}'
        bbox_dir = positive_sequence_root / f'bbox_seq{sequence_id}'

        for image_path in _iter_files(image_dir):
            mask_path = find_first_matching_file(mask_dir, image_path.stem, IMAGE_EXTENSIONS)
            bbox_path = find_first_matching_file(bbox_dir, image_path.stem, ('.txt',))
            _materialize_sample(
                source_name=f'polypgen_sequence_positive_seq{sequence_id}',
                split=split,
                source_type='polypgen_sequence_positive',
                source_center=None,
                source_sequence_id=sequence_id,
                sample_type='sequence',
                image_path=image_path,
                mask_path=mask_path,
                bbox_path=bbox_path,
                output_root=output_root,
                records=records,
                summary=summary,
            )

        negative_sequence_root = negative_root / f'seq{sequence_id}_neg'
        for image_path in _iter_files(negative_sequence_root):
            _materialize_sample(
                source_name=f'polypgen_sequence_negative_seq{sequence_id}',
                split=split,
                source_type='polypgen_sequence_negative',
                source_center=None,
                source_sequence_id=sequence_id,
                sample_type='sequence',
                image_path=image_path,
                mask_path=None,
                bbox_path=None,
                output_root=output_root,
                records=records,
                summary=summary,
            )


def _ingest_cropped_positive_sequences(source_root: Path, output_root: Path, records: List[Dict[str, object]], summary: DefaultDict[str, int]) -> None:
    cropped_root = source_root / 'sequence_data_positive_cropped' / 'positive_cropped'

    for sequence_id in range(1, 24):
        split = 'train' if sequence_id <= 18 else 'val'
        sequence_root = cropped_root / f'seq{sequence_id}'
        image_dir = sequence_root / 'images'
        mask_dir = sequence_root / 'masks'

        for image_path in _iter_files(image_dir):
            mask_path = find_first_matching_file(mask_dir, image_path.stem, IMAGE_EXTENSIONS)
            _materialize_sample(
                source_name=f'cropped_positive_seq{sequence_id}',
                split=split,
                source_type='cropped_positive_sequence',
                source_center=None,
                source_sequence_id=sequence_id,
                sample_type='crop',
                image_path=image_path,
                mask_path=mask_path,
                bbox_path=None,
                output_root=output_root,
                records=records,
                summary=summary,
            )


def build_phase1_combined_dataset(thesis_root: Path) -> Phase1DatasetPaths:
    thesis_root = Path(thesis_root).resolve()
    source_root = thesis_root / 'NeSy' / 'Dataset 1'
    if not source_root.exists():
        raise FileNotFoundError(f'Phase 1 source data not found at {source_root}')

    output_root = thesis_root / 'thesis_outputs' / 'phase1_combined_dataset'
    train_images = output_root / 'train' / 'images'
    train_labels = output_root / 'train' / 'labels'
    train_masks = output_root / 'train' / 'masks'
    val_images = output_root / 'val' / 'images'
    val_labels = output_root / 'val' / 'labels'
    val_masks = output_root / 'val' / 'masks'
    manifest_path = output_root / 'manifest.json'
    data_yaml_path = output_root / 'data.yaml'

    for directory in (train_images, train_labels, train_masks, val_images, val_labels, val_masks):
        directory.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    summary: DefaultDict[str, int] = defaultdict(int)

    _ingest_kvasir(source_root, output_root, records, summary)
    _ingest_polypgen_centers(source_root, output_root, records, summary)
    _ingest_polypgen_sequences(source_root, output_root, records, summary)
    _ingest_cropped_positive_sequences(source_root, output_root, records, summary)

    manifest = {
        'root': output_root.as_posix(),
        'summary': {
            'total': summary['total'],
            'train': summary['train_total'],
            'val': summary['val_total'],
            'positive': summary['positive'],
            'negative': summary['negative'],
            'by_source': {
                key: value for key, value in sorted(summary.items())
                if key.endswith('_total') and key not in {'total', 'train_total', 'val_total'}
            },
        },
        'samples': records,
    }

    _write_text(manifest_path, json.dumps(manifest, indent=2))
    _write_text(data_yaml_path, f"""path: {output_root.as_posix()}
train: train/images
val: val/images
nc: 1
names: ['polyp']
""")

    print("\n[Phase 1 dataset]")
    print(f"   Root: {output_root}")
    print(f"   Train samples: {summary['train_total']}")
    print(f"   Val samples: {summary['val_total']}")
    print(f"   Positive samples: {summary['positive']}")
    print(f"   Negative samples: {summary['negative']}")
    print(f"   Manifest: {manifest_path}")

    return Phase1DatasetPaths(
        root=output_root,
        train_images=train_images,
        train_labels=train_labels,
        train_masks=train_masks,
        val_images=val_images,
        val_labels=val_labels,
        val_masks=val_masks,
        data_yaml=data_yaml_path,
        manifest_path=manifest_path,
    )