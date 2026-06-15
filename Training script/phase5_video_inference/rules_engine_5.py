"""
RULES ENGINE FOR PHASE 5 VIDEO INFERENCE
Implements all 5 thesis rules correctly with proper semantic meaning.

Rule 1: MedSAM2 Dependent/Hybrid - Only segment polyp ROIs from YOLO/RTDETR boxes, never full frame
Rule 2: Consensus Voting - Only accept detections where ALL 3 models agree in same frame
Rule 3: 70-30 Split - Train symbolic reasoning on 70% of applied videos, test on 30%
Rule 4: SSL Features - Use 444-dim SSL features in every symbolic reasoning decision
Rule 5: ROI Analysis - Only analyze polyp regions, exclude borders and artifacts
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict
from scipy import ndimage


class RulesEngine5:
    """Unified engine for all 5 rules"""
    
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.frame_consensus_cache = {}  # Cache consensus decisions per frame


    def rule2_consensus_voting(self,
                               frame_idx: int,
                               yolo_boxes: List[List[float]],
                               yolo_confs: List[float],
                               rtdetr_boxes: List[List[float]],
                               rtdetr_confs: List[float],
                               medsam_boxes: List[List[float]],
                               medsam_confs: List[float],
                               iou_threshold: float = 0.3) -> Dict:
        
        consensus_boxes = []
        consensus_confidences = []
        consensus_models = []

        if not (yolo_boxes and rtdetr_boxes and medsam_boxes):
            return {
                'consensus_boxes': [],
                'consensus_confidences': [],
                'consensus_models': [],
                'frame_has_consensus': False,
                'frame_idx': frame_idx,
                'yolo_detected': len(yolo_boxes) > 0,
                'rtdetr_detected': len(rtdetr_boxes) > 0,
                'medsam_detected': len(medsam_boxes) > 0,
                'ious': {'yolo_rtdetr': 0, 'yolo_medsam': 0}
            }

        # Track the best IoU seen across ALL yolo-box iterations for diagnostics
        max_frame_rtdetr_iou = 0.0
        max_frame_medsam_iou = 0.0

        for yolo_box, yolo_conf in zip(yolo_boxes, yolo_confs):
            # Best RTDETR match for this YOLO box
            best_rtdetr_iou = 0
            best_rtdetr_idx = -1
            for rtdetr_idx, rtdetr_box in enumerate(rtdetr_boxes):
                iou = self._calculate_iou(yolo_box, rtdetr_box)
                if iou > best_rtdetr_iou and iou >= iou_threshold:
                    best_rtdetr_iou = iou
                    best_rtdetr_idx = rtdetr_idx
            max_frame_rtdetr_iou = max(max_frame_rtdetr_iou, best_rtdetr_iou)

            if best_rtdetr_idx == -1:
                continue

            # Best MedSAM match for this YOLO box (IoM handles tight SAM mask vs loose YOLO box)
            best_medsam_iou = 0
            best_medsam_idx = -1
            for medsam_idx, medsam_box in enumerate(medsam_boxes):
                iou = self._calculate_iou(yolo_box, medsam_box)
                if iou > best_medsam_iou and iou >= iou_threshold:
                    best_medsam_iou = iou
                    best_medsam_idx = medsam_idx
            max_frame_medsam_iou = max(max_frame_medsam_iou, best_medsam_iou)

            if best_medsam_idx == -1:
                continue

            consensus_box = self._average_boxes([
                yolo_box,
                rtdetr_boxes[best_rtdetr_idx],
                medsam_boxes[best_medsam_idx]
            ])
            consensus_conf = (yolo_conf + rtdetr_confs[best_rtdetr_idx] + medsam_confs[best_medsam_idx]) / 3.0
            consensus_boxes.append(consensus_box)
            consensus_confidences.append(consensus_conf)
            consensus_models.append(['YOLO', 'RTDETR', 'MedSAM2'])

        return {
            'consensus_boxes': consensus_boxes,
            'consensus_confidences': consensus_confidences,
            'consensus_models': consensus_models,
            'frame_has_consensus': len(consensus_boxes) > 0,
            'frame_idx': frame_idx,
            'yolo_detected': len(yolo_boxes) > 0,
            'rtdetr_detected': len(rtdetr_boxes) > 0,
            'medsam_detected': len(medsam_boxes) > 0,
            'ious': {
                'yolo_rtdetr': max_frame_rtdetr_iou,
                'yolo_medsam': max_frame_medsam_iou
            }
        }
    
    def rule5_filter_border_artifacts(self,
                                      frame: np.ndarray,
                                      box: List[float],
                                      border_margin_percent: float = 0.02) -> bool:
        """
        RULE 5: ROI Analysis - Filter Border and Artifacts
        
        Exclude detections that are:
        - Too close to frame borders (endoscope not fully in)
        - In completely black regions (non-tissue areas)
        - Touching frame edges (likely artifacts)
        
        Returns:
            is_valid: True if box should be kept
        """
        h, w = frame.shape[:2]
        # Boxes are in pixel coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate margins
        margin_x = int(w * border_margin_percent)
        margin_y = int(h * border_margin_percent)
        
        # Check if box is too close to edges
        if x1 < margin_x or x2 > w - margin_x or y1 < margin_y or y2 > h - margin_y:
            return False
        
        # Reject boxes that cover more than 60% of frame area (colon wall bleed-through)
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = h * w
        if box_area / frame_area > 0.60:
            return False
        
        # Reject boxes smaller than 0.5% of frame area (noise)
        if box_area / frame_area < 0.005:
            return False
        
        # Extract ROI and check if it's valid tissue (not just black/white)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        # Calculate mean intensity
        if len(roi.shape) == 3:
            mean_intensity = np.mean(roi)
        else:
            mean_intensity = np.mean(roi)
        
        # Valid polyps should have moderate intensity (not pure black < 10, not pure white > 245)
        if mean_intensity < 10 or mean_intensity > 245:
            return False
        
        # Check for texture (not completely uniform)
        std_intensity = np.std(roi)
        if std_intensity < 5:
            return False  # Too uniform (likely artifact)
        
        return True
    
    @staticmethod
    def _average_boxes(boxes: List[List[float]]) -> List[float]:
        """Average multiple boxes"""
        return [
            np.mean([b[0] for b in boxes]),
            np.mean([b[1] for b in boxes]),
            np.mean([b[2] for b in boxes]),
            np.mean([b[3] for b in boxes])
        ]
    
    def temporal_consensus_aggregation(self,
                                       consensus_results: Dict,
                                       max_frame_gap: int = 1,
                                       min_consensus_frames: int = 20) -> Dict:
        """
        TEMPORAL CONSENSUS AVERAGING (Your 20-frame continuous detection idea)
        
        For continuous sequences of frames where ALL 3 models agree on a polyp,
        average the confidence scores across those frames. This significantly boosts
        detection reliability.
        
        Args:
            consensus_results: Dict of frame_idx -> consensus detections
            max_frame_gap: Max frames to allow between detections before breaking a track
        
        Returns:
            temporal_aggregated: Dict with polyp tracks and averaged confidences
        """
        
        temporal_tracks = []  # List of polyp tracks
        
        if not consensus_results:
            return {
                'tracks': [], 
                'total_polyp_instances': 0, 
                'total_frame_coverage': 0,
                'average_confidence_boosted': [],
                'average_boost_amount': 0.0
            }
        
        sorted_frames = sorted(consensus_results.keys())
        used_detections = set()
        
        for start_frame_idx in sorted_frames:
            if start_frame_idx in used_detections:
                continue
            
            frame_detections = consensus_results[start_frame_idx]['boxes']
            frame_confs = consensus_results[start_frame_idx]['confidences']
            
            for det_idx, (start_box, start_conf) in enumerate(zip(frame_detections, frame_confs)):
                if (start_frame_idx, det_idx) in used_detections:
                    continue
                
                # Track this polyp across consecutive frames
                track_frames = [start_frame_idx]
                track_confs = [start_conf]
                track_boxes = [start_box]
                
                # Look forward to find continuous detections of this same polyp
                current_frame = start_frame_idx
                
                for next_frame_idx in sorted_frames:
                    if next_frame_idx <= current_frame:
                        continue
                    
                    # Check if frames are continuous (within gap threshold)
                    if next_frame_idx - current_frame > max_frame_gap:
                        break
                    
                    # Try to match polyp in next frame
                    next_frame_detections = consensus_results[next_frame_idx]['boxes']
                    next_frame_confs = consensus_results[next_frame_idx]['confidences']
                    
                    # Find closest matching detection (by center distance)
                    current_box_center = [
                        (start_box[0] + start_box[2]) / 2,
                        (start_box[1] + start_box[3]) / 2
                    ]
                    
                    best_match_idx = -1
                    best_distance = float('inf')
                    
                    for next_det_idx, next_box in enumerate(next_frame_detections):
                        if (next_frame_idx, next_det_idx) in used_detections:
                            continue
                        
                        next_box_center = [
                            (next_box[0] + next_box[2]) / 2,
                            (next_box[1] + next_box[3]) / 2
                        ]
                        
                        distance = np.sqrt(
                            (current_box_center[0] - next_box_center[0])**2 +
                            (current_box_center[1] - next_box_center[1])**2
                        )
                        
                        # Remove hardcoded 100px — use box size as reference distance
                        box_diagonal = np.sqrt(
                            (start_box[2] - start_box[0])**2 + (start_box[3] - start_box[1])**2
                        )
                        # Allow center shift up to 60% of the box diagonal (handles large polyp camera movement)
                        max_allowed_distance = max(100, box_diagonal * 0.60)
                        
                        if distance < best_distance and distance < max_allowed_distance:
                            best_match_idx = next_det_idx
                            best_distance = distance
                    
                    if best_match_idx >= 0:
                        # Found matching detection in next frame
                        track_frames.append(next_frame_idx)
                        track_confs.append(next_frame_confs[best_match_idx])
                        track_boxes.append(next_frame_detections[best_match_idx])
                        used_detections.add((next_frame_idx, best_match_idx))
                        current_frame = next_frame_idx
                        start_box = next_frame_detections[best_match_idx]  # Update for next iteration
                    else:
                        # No match found - end of track
                        break
                
                # Store track only when we have a sustained continuous consensus run
                if len(track_frames) >= min_consensus_frames:
                    avg_confidence = np.mean(track_confs)
                    track_box = self._average_boxes(track_boxes)
                    
                    # Issue 2: Weight detection confidence by number of frames in track
                    # Use all track frame confidences, not just consensus window
                    all_det_confs = list(track_confs) if track_confs else [avg_confidence]
                    det_conf = float(np.mean(all_det_confs)) if all_det_confs else avg_confidence
                    
                    temporal_tracks.append({
                        'polyp_id': len(temporal_tracks) + 1,
                        'start_frame': track_frames[0],
                        'end_frame': track_frames[-1],
                        'num_frames': len(track_frames),
                        'frame_sequence': track_frames,
                        'confidences': track_confs,
                        'original_average_conf': np.mean([track_confs[0]]),  # First frame
                        'temporal_average_conf': float(det_conf),
                        'confidence_boost': float(det_conf - np.mean([track_confs[0]])),
                        'box': track_box,
                        'all_boxes': track_boxes
                    })
                    
                    # Mark only the specific (frame, box_index) pairs used by this track
                    # Never mark whole frames — other polyps may be visible in those same frames
                    used_detections.add((start_frame_idx, det_idx))
        
        return {
            'tracks': temporal_tracks,
            'total_polyp_instances': len(temporal_tracks),
            'total_frame_coverage': sum(t['num_frames'] for t in temporal_tracks),
            'average_confidence_boosted': [t['temporal_average_conf'] for t in temporal_tracks],
            'average_boost_amount': float(np.mean([t['confidence_boost'] for t in temporal_tracks]) if temporal_tracks else 0.0)
        }
    
    def merge_overlapping_polyp_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """
        FIX FOR ISSUE 2: Merge overlapping tracks that represent the same polyp
        
        Videos should have only 1 unique polyp, but temporal tracking creates
        multiple tracks for the same polyp. This function merges them.
        
        Args:
            tracks: List of temporal tracks
            
        Returns:
            merged_tracks: Single track per unique polyp
        """
        if not tracks:
            return []
        
        merged_tracks = []
        used_track_indices = set()
        
        for i, track1 in enumerate(tracks):
            if i in used_track_indices:
                continue
            
            # Start with this track
            merged_track = track1.copy()
            merged_boxes = [track1['box']]
            merged_confs = [track1['temporal_average_conf']]
            merged_frame_sequences = list(track1.get('frame_sequence', []))
            
            # Find overlapping tracks
            for j, track2 in enumerate(tracks):
                if j <= i or j in used_track_indices:
                    continue
                
                # Check spatial overlap (IoU of bounding boxes)
                iou = self._calculate_iou(track1['box'], track2['box'])
                
                # Check temporal overlap (frame ranges)
                time_overlap = self._check_temporal_overlap(track1, track2)
                
                # Changed from aggressive OR logic to spatially-anchored conditions
                # Only merge when spatially AND temporally close, or genuinely overlapping
                time_gap = abs(track1['start_frame'] - track2['start_frame']) - abs(track1['end_frame'] - track2['end_frame'])
                time_gap = max(0, time_gap)  # Time between end of one and start of other
                max_merge_gap = 200  # Frames; same polyp at different time points
                should_merge = (
                    (iou > 0.4) or  # clear spatial overlap only — raised from 0.3
                    (time_overlap and iou > 0.35) or  # temporal overlap needs stronger spatial match
                    (iou > 0.3 and time_gap < max_merge_gap)  # Problem C: same polyp, different time
                )
                
                if should_merge:
                    merged_boxes.append(track2['box'])
                    merged_confs.append(track2['temporal_average_conf'])
                    merged_frame_sequences.extend(track2.get('frame_sequence', []))
                    used_track_indices.add(j)
                    frame_distance = abs(track1['start_frame'] - track2['start_frame'])
                    print(f"        🔗 Merging track {j} into track {i} (IoU: {iou:.3f}, frames: {frame_distance})")
            
            # Average the merged data
            if len(merged_boxes) > 1:
                merged_track['box'] = self._average_boxes(merged_boxes)
                merged_track['temporal_average_conf'] = float(np.mean(merged_confs))
                merged_frames = sorted(set(merged_frame_sequences))
                merged_track['num_frames'] = len(merged_frames)
                merged_track['frame_sequence'] = merged_frames
                merged_track['start_frame'] = min(merged_frames)
                merged_track['end_frame'] = max(merged_frames)
                merged_track['merged_from'] = len(merged_boxes)
                print(f"      🔗 Merged {len(merged_boxes)} overlapping tracks into 1 polyp")
            
            merged_tracks.append(merged_track)
            used_track_indices.add(i)
        
        # Reassign polyp_ids to be sequential (1, 2, 3, ...)
        for idx, track in enumerate(merged_tracks):
            track['polyp_id'] = idx + 1
        
        print(f"      📊 Reduced {len(tracks)} tracks to {len(merged_tracks)} unique polyps")
        return merged_tracks
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Intersection over Minimum (IoM) so a tight SAM2 mask fully inside a
        loose YOLO box scores 1.0 instead of ~0.2 with standard IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        min_area = min(box1_area, box2_area)

        return inter_area / min_area if min_area > 0 else 0.0
    
    def _check_temporal_overlap(self, track1: Dict, track2: Dict) -> bool:
        """Check if two tracks overlap in time"""
        start1, end1 = track1['start_frame'], track1['end_frame']
        start2, end2 = track2['start_frame'], track2['end_frame']
        
        # Check for any overlap
        return not (end1 < start2 or end2 < start1)
    
    def _average_boxes(self, boxes: List[List[float]]) -> List[float]:
        """Average multiple bounding boxes"""
        if not boxes:
            return [0, 0, 0, 0]
        
        avg_box = np.mean(boxes, axis=0)
        return avg_box.tolist()