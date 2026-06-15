# -*- coding: utf-8 -*-
"""
CONSENSUS VOTING ENGINE
Implements consensus voting mechanism where only frames with agreement from 
ALL THREE models (YOLO + RTDETR + MedSAM2) are considered high-confidence detections.

Rule 2: Agreement between models significantly improves precision and reduces false positives.
Expected improvement: +20-35% accuracy
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection over Minimum (IoM) so a tight SAM2 mask fully inside a
    loose detector box scores 1.0 instead of ~0.2 with standard IoU."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    min_area = min(box1_area, box2_area)

    return inter_area / min_area if min_area > 0 else 0.0


def find_overlapping_boxes(yolo_boxes: List[float],
                          rtdetr_boxes: List[float],
                          medsam_boxes: List[float],
                          iou_threshold: float = 0.3) -> Tuple[bool, Dict]:
    """
    Check if all three models have a genuine 3-way overlapping detection in the
    same frame.  Requires a single (YOLO, RTDETR, MedSAM2) triplet where every
    pair exceeds iou_threshold — avoids false consensus when separate pairs
    happen to overlap independently at different locations.
    """
    consensus_info = {
        'has_consensus': False,
        'yolo_detected': len(yolo_boxes) > 0,
        'rtdetr_detected': len(rtdetr_boxes) > 0,
        'medsam_detected': len(medsam_boxes) > 0,
        'consensus_boxes': [],
        'yolo_rtdetr_iou': 0.0,
        'yolo_medsam_iou': 0.0,
        'rtdetr_medsam_iou': 0.0,
        'avg_consensus_iou': 0.0,
    }

    if not yolo_boxes or not rtdetr_boxes or not medsam_boxes:
        return False, consensus_info

    # Require all three pairs to overlap on the SAME triplet of boxes
    for y_box in yolo_boxes:
        for r_box in rtdetr_boxes:
            y_r_iou = calculate_iou(y_box, r_box)
            if y_r_iou < iou_threshold:
                continue
            for m_box in medsam_boxes:
                y_m_iou = calculate_iou(y_box, m_box)
                r_m_iou = calculate_iou(r_box, m_box)
                if y_m_iou >= iou_threshold and r_m_iou >= iou_threshold:
                    consensus_info['has_consensus'] = True
                    consensus_info['yolo_rtdetr_iou'] = y_r_iou
                    consensus_info['yolo_medsam_iou'] = y_m_iou
                    consensus_info['rtdetr_medsam_iou'] = r_m_iou
                    consensus_info['avg_consensus_iou'] = float(np.mean([y_r_iou, y_m_iou, r_m_iou]))
                    consensus_info['consensus_boxes'].append(y_box)

    if consensus_info['has_consensus']:
        return True, consensus_info

    return False, consensus_info


def aggregate_consensus_frames(detections: Dict, 
                              min_consensus_frames: int = 20,
                              iou_threshold: float = 0.3) -> Dict:
    """
    Aggregate detections across frames, identifying consensus runs.
    
    A "consensus run" is a sequence of frames where all 3 models agree.
    Only consider high-confidence runs with 20+ consecutive consensus frames.
    
    Args:
        detections: Dict with keys 'yolo', 'rtdetr', 'medsam' (MedSAM2) containing frame detections
        min_consensus_frames: Minimum consecutive frames for high confidence
        iou_threshold: IoU threshold for model agreement
    
    Returns:
        aggregated_result: Dict containing:
            - consensus_runs: List of consensus run info
            - overall_confidence: Average confidence across consensus runs
            - polyp_present: Boolean - is polyp high-confidence present?
    """
    
    # Create frame-indexed detection lookup
    yolo_by_frame = {d['frame']: d for d in detections.get('yolo', [])}
    rtdetr_by_frame = {d['frame']: d for d in detections.get('rtdetr', [])}
    medsam_by_frame = {d['frame']: d for d in detections.get('medsam', [])}
    
    # Get all frame indices
    all_frames = set(yolo_by_frame.keys()) | set(rtdetr_by_frame.keys()) | set(medsam_by_frame.keys())
    
    if not all_frames:
        return {
            'consensus_runs': [],
            'overall_confidence': 0.0,
            'polyp_present': False,
            'consensus_frame_count': 0,
            'non_consensus_frame_count': 0,
        }
    
    # Find consensus runs
    sorted_frames = sorted(all_frames)
    consensus_runs = []
    current_run = []
    
    for frame_idx in sorted_frames:
        yolo_det = yolo_by_frame.get(frame_idx, {})
        rtdetr_det = rtdetr_by_frame.get(frame_idx, {})
        medsam_det = medsam_by_frame.get(frame_idx, {})
        
        yolo_boxes = yolo_det.get('boxes', [])
        rtdetr_boxes = rtdetr_det.get('boxes', [])
        medsam_boxes = medsam_det.get('boxes', [])
        
        has_consensus, consensus_info = find_overlapping_boxes(
            yolo_boxes, rtdetr_boxes, medsam_boxes, iou_threshold
        )
        
        if has_consensus:
            current_run.append({
                'frame': frame_idx,
                'consensus_info': consensus_info,
                'yolo_conf': np.mean(yolo_det.get('confidences', [0])),
                'rtdetr_conf': np.mean(rtdetr_det.get('confidences', [0])),
                'medsam_conf': np.mean(medsam_det.get('confidences', [0])),  # MedSAM2
            })
        else:
            # End current run if it exists and is long enough
            if len(current_run) >= min_consensus_frames:
                consensus_runs.append({
                    'start_frame': current_run[0]['frame'],
                    'end_frame': current_run[-1]['frame'],
                    'duration': len(current_run),
                    'frames': current_run,
                    'avg_yolo_conf': np.mean([f['yolo_conf'] for f in current_run]),
                    'avg_rtdetr_conf': np.mean([f['rtdetr_conf'] for f in current_run]),
                    'avg_medsam_conf': np.mean([f['medsam_conf'] for f in current_run]),  # MedSAM2
                    'avg_consensus_iou': np.mean([f['consensus_info']['avg_consensus_iou'] for f in current_run]),
                })
            current_run = []
    
    # Handle final run
    if len(current_run) >= min_consensus_frames:
        consensus_runs.append({
            'start_frame': current_run[0]['frame'],
            'end_frame': current_run[-1]['frame'],
            'duration': len(current_run),
            'frames': current_run,
            'avg_yolo_conf': np.mean([f['yolo_conf'] for f in current_run]),
            'avg_rtdetr_conf': np.mean([f['rtdetr_conf'] for f in current_run]),
            'avg_medsam_conf': np.mean([f['medsam_conf'] for f in current_run]),
            'avg_consensus_iou': np.mean([f['consensus_info']['avg_consensus_iou'] for f in current_run]),
        })
    
    # Calculate overall metrics
    total_consensus_frames = sum(len(run['frames']) for run in consensus_runs)
    total_frames = len(sorted_frames)
    
    # Overall confidence: average confidence from all consensus runs
    overall_confidence = 0.0
    if consensus_runs:
        all_confidences = []
        for run in consensus_runs:
            all_confidences.extend([
                run['avg_yolo_conf'],
                run['avg_rtdetr_conf'],
                run['avg_medsam_conf']
            ])
        overall_confidence = np.mean(all_confidences) if all_confidences else 0.0
    
    # Polyp present if we have at least one significant consensus run
    polyp_present = len(consensus_runs) > 0 and overall_confidence > 0.5
    
    return {
        'consensus_runs': consensus_runs,
        'overall_confidence': float(overall_confidence),
        'polyp_present': polyp_present,
        'consensus_frame_count': total_consensus_frames,
        'non_consensus_frame_count': total_frames - total_consensus_frames,
        'consensus_percentage': (total_consensus_frames / total_frames * 100) if total_frames > 0 else 0.0,
        'num_consensus_runs': len(consensus_runs),
    }


def merge_consensus_detections(consensus_result: Dict) -> Dict:
    """
    Create final detection result from consensus analysis.
    
    Returns detections that should be used for final prediction.
    """
    merged = {
        'detected': consensus_result['polyp_present'],
        'confidence': consensus_result['overall_confidence'],
        'method': 'consensus_voting',
        'consensus_runs': len(consensus_result['consensus_runs']),
        'consensus_frames': consensus_result['consensus_frame_count'],
        'consensus_percentage': consensus_result['consensus_percentage'],
    }
    
    # Add details from primary consensus run (longest)
    if consensus_result['consensus_runs']:
        longest_run = max(consensus_result['consensus_runs'], key=lambda x: x['duration'])
        merged['primary_run'] = {
            'start_frame': longest_run['start_frame'],
            'end_frame': longest_run['end_frame'],
            'duration': longest_run['duration'],
            'avg_iou': longest_run['avg_consensus_iou'],
        }
    
    return merged
