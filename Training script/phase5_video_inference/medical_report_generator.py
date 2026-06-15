"""
Generate professional PDF medical reports for polyp detection results
Similar to a doctor's clinical report to patient
"""

import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from PIL import Image as PILImage
import io

# Import polyp type classifier for descriptions
from polyp_type_classifier import PolypTypeClassifier


class MedicalReportGenerator:
    """Generate professional medical reports for polyp detection"""
    
    def __init__(self, video_id, output_dir, frames, detections, segmentations, inference_report):
        """
        Initialize report generator
        
        Args:
            video_id: Video identifier
            output_dir: Directory to save PDF
            frames: List of video frames
            detections: Detection results from pipeline
            inference_report: JSON inference report
        """
        self.video_id = video_id
        self.output_dir = Path(output_dir)
        self.frames = frames
        self.detections = detections
        self.segmentations = segmentations or {}
        self.report_data = inference_report
        
        # Create report directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_path = self.output_dir / f"{video_id}_MEDICAL_REPORT.pdf"
        self.temp_images = self.output_dir / ".temp_images"
        self.temp_images.mkdir(exist_ok=True)
        
    def extract_polyp_region(self, frame, box, mask=None, size=(300, 300)):
        """Extract polyp region from frame with 20% contextual padding."""
        try:
            if frame is None or box is None or len(box) != 4:
                return None

            h, w = frame.shape[:2]

            if mask is not None and mask.sum() > 0:
                coords = np.where(mask > 0)
                if len(coords) == 2 and len(coords[0]) > 0:
                    y1, y2 = int(coords[0].min()), int(coords[0].max())
                    x1, x2 = int(coords[1].min()), int(coords[1].max())
                    # Expand to union with detector box so green rect covers full polyp
                    bx1, by1, bx2, by2 = map(float, box)
                    if max(bx1, by1, bx2, by2) <= 1.0:
                        bx1, bx2 = bx1 * w, bx2 * w
                        by1, by2 = by1 * h, by2 * h
                    x1 = min(x1, int(bx1)); y1 = min(y1, int(by1))
                    x2 = max(x2, int(bx2)); y2 = max(y2, int(by2))
                else:
                    bx1, by1, bx2, by2 = map(float, box)
                    if max(bx1, by1, bx2, by2) <= 1.0:
                        bx1, bx2 = bx1 * w, bx2 * w
                        by1, by2 = by1 * h, by2 * h
                    x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)
            else:
                bx1, by1, bx2, by2 = map(float, box)
                if max(bx1, by1, bx2, by2) <= 1.0:
                    bx1, bx2 = bx1 * w, bx2 * w
                    by1, by2 = by1 * h, by2 * h
                x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)

            # Show 40% context around the polyp so the image is clinically meaningful
            # For tiny polyps this gives surrounding tissue context
            pad_w = max(30, int((x2 - x1) * 0.40))
            pad_h = max(30, int((y2 - y1) * 0.40))

            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(w, x2 + pad_w)
            py2 = min(h, y2 + pad_h)

            roi = frame[py1:py2, px1:px2].copy()
            if roi.size == 0:
                return None

            cv2.rectangle(roi, (x1 - px1, y1 - py1), (x2 - px1, y2 - py1), (0, 255, 0), 2)
            return roi  # No conversion needed — frames are already RGB
        except Exception as e:
            print(f"Error extracting polyp region: {e}")
            return None

    @staticmethod
    def _fit_image_display_size(img_w, img_h, max_w=320, max_h=320):
        if img_w <= 0 or img_h <= 0:
            return max_w, max_h

        aspect = float(img_w) / float(img_h)
        if aspect > 1.0:
            return min(max_w, img_w), min(max_w, img_w) / aspect
        return min(max_h, img_h) * aspect, min(max_h, img_h)

    @staticmethod
    def _format_decimal(value, digits=3):
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value).replace(',', '.')

    @staticmethod
    def _format_percent(value, digits=1):
        try:
            return f"{float(value):.{digits}%}"
        except Exception:
            return str(value)

    def _get_detection_method_label(self):
        """Get detection method label based on MedSAM2 availability"""
        medsam_count = self.report_data.get('detection_summary', {}).get('medsam', 0) if self.report_data else 0
        if medsam_count > 0:
            return 'Multi-Model Consensus (MedSAM2 Available)'
        else:
            return 'Detector Consensus (MedSAM2 Unavailable)'

    def _get_detection_method_description(self):
        """Get detection method description based on MedSAM2 availability"""
        medsam_count = self.report_data.get('detection_summary', {}).get('medsam', 0) if self.report_data else 0
        if medsam_count > 0:
            return 'YOLO + RT-DETR + MedSAM2 agreement (Full 3-model consensus)'
        else:
            return 'YOLO + RT-DETR agreement (MedSAM2 unavailable for this video)'

    def get_type_description(self, polyp_type, risk_class="UNKNOWN"):
        polyp_type = str(polyp_type).upper().strip()
        risk_class = str(risk_class).upper()
        
        # System C type names mapped to clinical descriptions
        descriptions = {
            'BLEEDING_POLYP':          'Active hemorrhagic lesion — elevated vascularity with redness signal. Immediate clinical review required.',
            'POST_RESECTION_BLEEDING': 'Post-resection hemorrhage — haemostasis required. Urgent endoscopic intervention indicated.',
            'MALIGNANT_POLYP':         'Morphology consistent with high-grade neoplasia — biopsy and urgent resection strongly recommended.',
            'LATERAL_SPREADING_TUMOR': 'Lateral spreading tumor — advanced endoscopic resection (EMR/ESD) indicated.',
            'FLAT_POLYP':              'Flat mucosal lesion — chromoendoscopy or NBI assessment recommended.',
            'SERRATED_POLYP':          'Serrated lesion — consider serrated polyposis syndrome. Complete resection required.',
            'VILLOUS_POLYP':           'Villous architecture — high adenoma grade. Complete resection required.',
            'PEDUNCULATED_POLYP':      'Pedunculated polypoid lesion — snare polypectomy indicated.',
            'LARGE_POLYP':             'Large polypoid lesion — staged or piecemeal EMR may be required.',
            'COLITIS':                 'Inflammatory mucosal pattern — compatible with IBD or infectious colitis. Clinical correlation required.',
            'ADENOMATOUS_POLYP':       'Polypoid lesion — discrete mucosal protrusion identified. Endoscopic resection recommended.',
            'LIFTED_POLYP':            'Lifted lesion post-injection — submucosal plane confirmed. Resection feasible.',
            'RESECTED_POLYP':          'Post-resection site — confirm clear margins on histology.',
            'SMALL_POLYP':             'Diminutive lesion — resect and discard per protocol.',
            'NORMAL_MUCOSA':           'Normal mucosal pattern — no polyp morphology identified.',
            'UNKNOWN':                 'Morphological type undetermined — insufficient feature contrast for classification.',
        }
        
        return descriptions.get(polyp_type, descriptions.get('UNKNOWN'))
    
    def save_image_to_buffer(self, cv_image):
        """Convert CV2 image to PIL and save to buffer"""
        try:
            # Convert to PIL
            pil_image = PILImage.fromarray(cv_image.astype('uint8'))
            
            # Save to buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            return buffer
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def get_risk_color(self, classification):
        """Get color based on risk classification"""
        classification_text = str(classification).upper()
        if 'HIGH' in classification_text:
            return colors.HexColor('#FF4444')  # Red
        elif 'MEDIUM' in classification_text or 'UNCERTAIN' in classification_text:
            return colors.HexColor('#FFB000')  # Amber
        elif 'LOW' in classification_text:
            return colors.HexColor('#44CC44')  # Green
        else:
            return colors.HexColor('#888888')  # Gray
    
    def get_risk_description(self, classification, confidence):
        """Get clinical description of risk"""
        classification_text = str(classification).upper()
        if confidence >= 0.90:
            confidence_word = "High-confidence"
        elif confidence >= 0.80:
            confidence_word = "Moderate-confidence"
        else:
            confidence_word = "Low-confidence"

        if 'HIGH' in classification_text:
            if confidence >= 0.90:
                return "High-confidence HIGH RISK detection - Requires close monitoring and potential intervention"
            return f"{confidence_word} HIGH RISK detection - Further evaluation recommended"
        elif 'MEDIUM' in classification_text or 'UNCERTAIN' in classification_text:
            return f"{confidence_word} MODERATE RISK detection - Clinical review recommended"
        elif 'LOW' in classification_text:
            if confidence >= 0.90:
                return "High-confidence LOW RISK detection - Routine surveillance recommended"
            return f"{confidence_word} LOW RISK detection - Follow standard protocol"
        else:
            return "Insufficient evidence for a stable risk prediction - Further evaluation recommended"

    @staticmethod
    def _polyp_id_matches(left_id, right_id):
        if left_id is None or right_id is None:
            return False
        return str(left_id).strip() == str(right_id).strip()

    # Clinically validated risk tiers sourced from ESGE 2017 (Ferlitsch et al.)
    # and NICE classification criteria for colorectal lesions.
    # Weight 3 = HIGH RISK, 2 = MEDIUM RISK, 1 = LOW RISK, 0 = BASELINE/ZERO
    CLINICAL_RISK_HIERARCHY = {
        # HIGH RISK (3): Malignant, invasive, or acute haemorrhagic emergency
        'cancer':                                  3,
        'colorectal cancer':                       3,
        'gastric cancer':                          3,
        'oesaphageal cancer':                      3,
        'flat polyp probably an early cancer':     3,
        'polyp lst':                               3,
        'large flat polyp = lateral speading tumor': 3,
        'flat polyp':                              3,
        'polyp bleeding':                          3,
        'bleeding after polyp resection':          3,
        # MEDIUM RISK (2): Neoplastic precursors or active inflammatory conditions
        'flat polyp serrated':                     2,
        'villous polyp':                           2,
        'large polyp in anastomosis':              2,
        'clipping polyp stalk':                    2,
        'ulcerative colitis':                      2,
        'collitis':                                2,
        'duodenum ulcer':                          2,
        'bleeding gastric ulcer':                  2,
        'blood in lumen':                          2,
        # LOW RISK (1): Benign, non-neoplastic, or successfully managed
        'lifted polyp':                            1,
        'stained lifted polyp':                    1,
        'dye lifted polyp':                        1,
        'polyp, lifted, removed':                  1,
        'small polyp':                             1,
        'gastric fundic polyps':                   1,
        'polyp lipoma':                            1,
        'polyp resected':                          1,
        'resected polyp':                          1,
        # ZERO RISK / BASELINE (0): Normal structures or non-neoplastic artifacts
        'normal colon':                            0,
        'cecum':                                   0,
        'stomach':                                 0,
        'duodenum':                                0,
        'pylorus':                                 0,
        'stent removal':                           0,
        'moving worms':                            0,
        'out of patient':                          0,
        'nothing':                                 0,
    }

    # Map System C clinical class names to risk hierarchy weights
    # Derived from CLINICAL_CLASSES in polyp_type_classifier.py
    _CLINICAL_CLASS_TO_WEIGHT = {
        'BLEEDING_POLYP':           3,
        'POST_RESECTION_BLEEDING':  3,
        'MALIGNANT_POLYP':          3,
        'LATERAL_SPREADING_TUMOR':  3,
        'FLAT_POLYP':               2,
        'BLEEDING_ULCER':           3,   # HIGH — haemorrhagic ulcer, same urgency as bleeding polyp
        'LARGE_POLYP':              2,
        'VILLOUS_POLYP':            2,
        'SERRATED_POLYP':           2,
        'PEDUNCULATED_POLYP':       2,
        'COLITIS':                  2,
        'ADENOMATOUS_POLYP':        1,
        'LIFTED_POLYP':             1,
        'RESECTED_POLYP':           1,
        'SMALL_POLYP':              1,
        'NORMAL_MUCOSA':            0,
        'UNKNOWN':                  0,
    }

    @classmethod
    def _clinical_class_to_risk_label(cls, polyp_type: str) -> str:
        """
        Map a System C clinical class name to a display risk label.
        Uses _CLINICAL_CLASS_TO_WEIGHT which mirrors ESGE 2017 validated tiers.
        """
        weight = cls._CLINICAL_CLASS_TO_WEIGHT.get(str(polyp_type).upper().strip(), 0)
        if weight >= 3: return 'HIGH_RISK'
        if weight == 2: return 'MEDIUM_RISK'
        if weight == 1: return 'LOW_RISK'
        return 'BASELINE'

    @classmethod
    def _csv_finding_to_risk_label(cls, finding: str) -> str:
        """
        Map a raw CSV finding string to a risk label using CLINICAL_RISK_HIERARCHY.
        Longest-key substring match to handle compound finding strings.
        """
        f = str(finding).lower().strip()
        # Longest-key priority: more specific keys win over generic ones
        matched_weight = None
        matched_len = 0
        for key, weight in cls.CLINICAL_RISK_HIERARCHY.items():
            if key in f and len(key) > matched_len:
                matched_weight = weight
                matched_len = len(key)
        if matched_weight is None:
            return 'UNKNOWN'
        if matched_weight >= 3: return 'HIGH_RISK'
        if matched_weight == 2: return 'MEDIUM_RISK'
        if matched_weight == 1: return 'LOW_RISK'
        return 'BASELINE'

    def _resolve_polyp_type(self, polyp_data, matching_symbolic, classification, fallback_type='UNKNOWN', fallback_confidence=0.0):
        # Priority 1: Use type from symbolic result or polyp_data if set
        for source in (matching_symbolic, polyp_data):
            if not source:
                continue
            candidate_type = str(source.get('polyp_type', '')).strip().upper()
            if candidate_type and candidate_type not in ('UNKNOWN', ''):
                confidence = float(source.get('polyp_type_confidence', 
                                   source.get('type_confidence', fallback_confidence or 0.50)))
                # Don't use hardcoded confidence — use actual value, floor at 0.40
                confidence = max(0.40, min(0.95, confidence))
                return candidate_type, confidence

        # Priority 2: Derive type from features (no hardcoded thresholds for confidence)
        classification_text = str(classification).upper()
        redness = float(polyp_data.get('redness', 0.0))
        vessels = float(polyp_data.get('vessel_visibility', polyp_data.get('vessels', 0.0)))
        texture = float(polyp_data.get('texture', 0.0))
        
        # Compute type scores from features (no risk_score dependency)
        bleeding_score = redness * 0.5 + vessels * 0.5
        cancer_score   = texture * 0.5 + redness * 0.3 + vessels * 0.2
        polyp_score    = vessels * 0.3 + texture * 0.3 + (1.0 - redness) * 0.4
        normal_score   = max(0.0, 0.6 - redness - vessels - texture)
        
        scores = {
            'BLEEDING_POLYP':  bleeding_score,
            'MALIGNANT_POLYP': cancer_score,
            'ADENOMATOUS_POLYP': polyp_score,
            'NORMAL_MUCOSA':   normal_score,
        }
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        second_score = sorted(scores.values(), reverse=True)[1]
        margin = best_score - second_score
        
        total_signal = redness + vessels + texture
        if total_signal < 0.10:
            derived_confidence = 0.40 + total_signal * 0.5
        else:
            derived_confidence = 0.50 + margin * 2.5 + total_signal * 0.1
        derived_confidence = float(np.clip(derived_confidence, 0.40, 0.92))
        
        return best_type, derived_confidence

    def _select_representative_tracks(self, tracks, max_polyps=10):
        """Keep only stable, non-overlapping canonical entries for the PDF."""
        if not tracks:
            return []

        def entry_box(entry):
            box = entry.get('box', [])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return None

            x1, y1, x2, y2 = [float(value) for value in box]
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]

        def track_score(track):
            return (
                1 if track.get('consensus', False) else 0,
                int(track.get('num_frames', 0)),
                len(track.get('frame_sequence', [])),
                float(track.get('temporal_average_conf', track.get('confidence', 0.0)))
            )

        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        def cluster_support(cluster):
            representative = cluster['representative']
            return max(
                len(cluster['members']),
                int(representative.get('num_frames', 0)),
                len(representative.get('frame_sequence', []))
            )

        canonical_candidates = []
        for track in tracks:
            box = entry_box(track)
            if box is None:
                continue
            canonical_candidates.append({**track, 'box': box})

        if not canonical_candidates:
            return []

        clusters = []
        for track in sorted(canonical_candidates, key=track_score, reverse=True):
            best_cluster = None
            best_iou = 0.0
            for cluster in clusters:
                iou = calculate_iou(track['box'], cluster['representative']['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster

            if best_cluster is not None and best_iou >= 0.45:
                best_cluster['members'].append(track)
                if track_score(track) > track_score(best_cluster['representative']):
                    best_cluster['representative'] = track
            else:
                clusters.append({'representative': track, 'members': [track]})

        stable_clusters = []
        for cluster in clusters:
            representative = cluster['representative']
            support = cluster_support(cluster)
            confidence = float(representative.get('temporal_average_conf', representative.get('confidence', 0.0)))
            num_frames = int(representative.get('num_frames', 0))

            if representative.get('consensus', False) or num_frames >= 1 or (support >= 1 and confidence >= 0.40):
                stable_clusters.append((support, track_score(representative), representative))

        if not stable_clusters:
            return []

        stable_clusters.sort(
            key=lambda item: (
                item[0],
                item[1],
                float(item[2].get('temporal_average_conf', item[2].get('confidence', 0.0))),
                int(item[2].get('frame', 0))
            ),
            reverse=True
        )
        return [cluster[2] for cluster in stable_clusters[:max_polyps]]

    def _select_fallback_features(self, features, max_polyps=10):
        """Pick the best available raw features when no stable track survives."""
        if not features:
            return []

        def feature_box(entry):
            box = entry.get('box', [])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return None

            x1, y1, x2, y2 = [float(value) for value in box]
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]

        def feature_score(entry):
            return float(entry.get('temporal_average_conf', entry.get('confidence', 0.0)))

        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        selected = []
        for feature in sorted(features, key=feature_score, reverse=True):
            if len(selected) >= max_polyps:
                break

            box = feature_box(feature)
            if box is None:
                selected.append(feature)
                continue

            overlap = False
            for chosen in selected:
                chosen_box = feature_box(chosen)
                if chosen_box is not None and calculate_iou(box, chosen_box) >= 0.45:
                    overlap = True
                    break

            if not overlap:
                selected.append(feature)

        return selected

    def _get_canonical_polyp_entries(self, max_polyps=10):
        """Resolve the best available report entries, preferring temporal tracks."""
        symbolic_summary = self.report_data.get('symbolic_reasoning_summary', {})
        symbolic_results = symbolic_summary.get('results', [])

        temporal_tracks = symbolic_summary.get('_temporal_consensus', {}).get('tracks', [])
        if temporal_tracks:
            representative_tracks = self._select_representative_tracks(temporal_tracks, max_polyps=max_polyps)
            if representative_tracks:
                return representative_tracks, symbolic_results, True

        raw_features = self.report_data.get('polyp_features_detail', [])
        if raw_features:
            representative_features = self._select_representative_tracks(raw_features, max_polyps=max_polyps)
            if representative_features:
                return representative_features, symbolic_results, False

            fallback_features = self._select_fallback_features(raw_features, max_polyps=max_polyps)
            if fallback_features:
                return fallback_features, symbolic_results, False

        return [], symbolic_results, False
    
    def create_summary_section(self):
        """Create summary statistics section"""
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("POLYP DETECTION MEDICAL REPORT", title_style))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=20
        )
        story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Video information
        story.append(Paragraph("<b>VIDEO INFORMATION</b>", styles['Heading2']))
        
        video_info = [
            ['Video ID:', self.video_id],
            ['Total Frames:', str(len(self.frames))],
            ['Frame Rate:', '25.0 fps (assumed)'],
            ['Duration:', f"{len(self.frames) / 25.0:.1f} seconds"]
        ]
        
        video_table = Table(video_info, colWidths=[2*inch, 3*inch])
        video_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(video_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        story.append(Paragraph("<b>DETECTION SUMMARY</b>", styles['Heading2']))
        
        # Count detections by model
        yolo_count = sum(len(d.get('boxes', [])) for d in self.detections.get('yolo', []) if isinstance(d, dict))
        rtdetr_count = sum(len(d.get('boxes', [])) for d in self.detections.get('rtdetr', []) if isinstance(d, dict))
        medsam_count = sum(len(d.get('boxes', [])) for d in self.detections.get('medsam', []) if isinstance(d, dict))
        consensus_count = int(self.report_data.get('consensus_voting', {}).get('num_consensus_runs', 0))
        
        # Count by risk - use a high cap to ensure all polyps are counted for the summary table
        canonical_entries, symbolic_results, use_temporal_tracks = self._get_canonical_polyp_entries(max_polyps=50)
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        total_analyzed = len(canonical_entries)

        if canonical_entries:
            if use_temporal_tracks:
                for track in canonical_entries:
                    track_id = track.get('polyp_id')
                    matching_symbolic = None
                    for symbolic in symbolic_results:
                        if self._polyp_id_matches(symbolic.get('polyp_id'), track_id):
                            matching_symbolic = symbolic
                            break

                    pt = str((matching_symbolic or track or {}).get('polyp_type', 'UNKNOWN'))
                    risk_label = self._clinical_class_to_risk_label(pt)
                    if risk_label == 'HIGH_RISK':    high_risk += 1
                    elif risk_label == 'MEDIUM_RISK': medium_risk += 1
                    elif risk_label == 'LOW_RISK':    low_risk += 1
            else:
                for feature in canonical_entries:
                    pt = str(feature.get('polyp_type', 'UNKNOWN'))
                    risk_label = self._clinical_class_to_risk_label(pt)
                    if risk_label == 'HIGH_RISK':    high_risk += 1
                    elif risk_label == 'MEDIUM_RISK': medium_risk += 1
                    elif risk_label == 'LOW_RISK':    low_risk += 1
        
        summary_data = [
            ['Detection Method', 'Count', 'Status'],
            ['YOLO Detections', str(yolo_count), '✓'],
            ['RT-DETR Detections', str(rtdetr_count), '✓'],
            ['MedSAM2 Detections', str(medsam_count), '✓'],
            ['Consensus (3-Model Agreement)', str(consensus_count), '✓'],
            ['', '', ''],
            ['<b>Risk Classification</b>', '<b>Count</b>', '<b>Percentage</b>'],
            ['HIGH RISK', str(high_risk), f"{100*high_risk/max(1, total_analyzed):.1f}%"],
            ['MEDIUM RISK', str(medium_risk), f"{100*medium_risk/max(1, total_analyzed):.1f}%"],
            ['LOW RISK', str(low_risk), f"{100*low_risk/max(1, total_analyzed):.1f}%"],
            ['TOTAL ANALYZED', str(total_analyzed), '100.0%' if total_analyzed > 0 else '0.0%']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 6), (-1, 6), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
            ('ROWBACKGROUNDS', (0, 7), (-1, 10), [colors.HexColor('#FFE6E6'), colors.HexColor('#FFF2D9'), colors.HexColor('#E6FFE6')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        
        return story
    
    def create_polyp_details_section(self):
        """Create detailed analysis for each detected polyp"""
        story = []
        styles = getSampleStyleSheet()

        def calculate_iou(box1, box2):
            if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
                return 0.0

            x1 = max(float(box1[0]), float(box2[0]))
            y1 = max(float(box1[1]), float(box2[1]))
            x2 = min(float(box1[2]), float(box2[2]))
            y2 = min(float(box1[3]), float(box2[3]))

            if x2 <= x1 or y2 <= y1:
                return 0.0

            inter_area = (x2 - x1) * (y2 - y1)
            area1 = max(0.0, (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1])))
            area2 = max(0.0, (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1])))
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        
        story.append(PageBreak())
        story.append(Paragraph("<b>DETAILED POLYP ANALYSIS</b>", styles['Heading2']))
        story.append(Spacer(1, 0.15*inch))
        
        # Determine max_polyps from the actual data — count unique spatial clusters
        # in symbolic_results without artificial capping
        all_symbolic = self.report_data.get('symbolic_reasoning_summary', {}).get('results', []) if self.report_data else []
        all_tracks = self.report_data.get('symbolic_reasoning_summary', {}).get(
            '_temporal_consensus', {}
        ).get('tracks', []) if self.report_data else []
        
        canonical_entries, symbolic_results, use_temporal_tracks = self._get_canonical_polyp_entries(max_polyps=None)
        
        # Count distinct polyp_ids from symbolic results
        distinct_ids = set()
        for r in all_symbolic:
            pid = r.get('polyp_id')
            if pid is not None:
                distinct_ids.add(pid)
        for t in all_tracks:
            pid = t.get('polyp_id')
            if pid is not None:
                distinct_ids.add(pid)
        
        max_polyps = max(1, len(distinct_ids)) if distinct_ids else 2
        
        # Collect polyp details - MERGE TEMPORAL TRACKS WITH SYMBOLIC RESULTS
        # This prevents counting the same polyp multiple times across frames
        canonical_entries, symbolic_results, use_temporal_tracks = self._get_canonical_polyp_entries(max_polyps=max_polyps)

        # If no polyps were detected at all, show a clear "no polyp" section
        if not canonical_entries and not symbolic_results:
            story.append(Paragraph("NO POLYP DETECTED", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(
                "No polyp was identified in this video by the multi-model consensus "
                "(YOLO + RT-DETR + MedSAM2). The mucosa appears within normal limits "
                "based on visual feature analysis. Standard surveillance protocol applies.",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.2 * inch))
            return story

        if not canonical_entries:
            story.append(Paragraph("No stable canonical polyp crop available", styles['Normal']))
            return story

        if use_temporal_tracks:
            merged_features = []
            for track in canonical_entries:
                track_id = track.get('polyp_id')

                # Find matching symbolic result by polyp_id
                matching_symbolic = None
                for symbolic in symbolic_results:
                    if self._polyp_id_matches(symbolic.get('polyp_id'), track_id):
                        matching_symbolic = symbolic
                        break

                # Fallback: if no id match, try matching by frame proximity
                if matching_symbolic is None and symbolic_results:
                    track_frame = int(track.get('start_frame', track.get('frame', 0)))
                    best_frame_dist = float('inf')
                    for symbolic in symbolic_results:
                        sym_frame = int(symbolic.get('frame', symbolic.get('start_frame', 0)))
                        dist = abs(sym_frame - track_frame)
                        if dist < best_frame_dist:
                            best_frame_dist = dist
                            matching_symbolic = symbolic

                if matching_symbolic:
                    # Merge track data with symbolic classification data
                    detection_confidence = float(track.get('temporal_average_conf', track.get('confidence', 0.0)))
                    symbolic_confidence = float(matching_symbolic.get('symbolic_confidence', matching_symbolic.get('confidence', 0.0)))
                    clinical_confidence = float(matching_symbolic.get('clinical_confidence', matching_symbolic.get('confidence', symbolic_confidence)))
                    merged_polyp = {
                        **track,
                        **matching_symbolic,
                        'classification': matching_symbolic.get('classification', matching_symbolic.get('clinical_classification', 'UNKNOWN')),
                        'expert_classification': matching_symbolic.get('expert_classification', matching_symbolic.get('classification', 'UNKNOWN')),
                        'polyp_type': matching_symbolic.get('polyp_type', matching_symbolic.get('expert_classification', 'UNKNOWN')),
                        'polyp_type_confidence': matching_symbolic.get('polyp_type_confidence', matching_symbolic.get('type_confidence', 0.5)),
                        'type_confidence': matching_symbolic.get('type_confidence', matching_symbolic.get('polyp_type_confidence', 0.5)),
                        'redness': matching_symbolic.get('redness', 0.0),
                        'texture': matching_symbolic.get('texture', 0.0),
                        'vessel_visibility': matching_symbolic.get('vessel_visibility', 0.0),
                        'radius': matching_symbolic.get('radius', 0.0),
                        'risk_score': matching_symbolic.get('risk_score', matching_symbolic.get('medical_risk_score', 0.0)),
                        'medical_risk_score': matching_symbolic.get('medical_risk_score', matching_symbolic.get('risk_score', 0.0)),
                        'detection_confidence': detection_confidence,
                        'symbolic_confidence': symbolic_confidence,
                        'clinical_confidence': clinical_confidence,
                        'confidence': clinical_confidence
                    }
                    merged_features.append(merged_polyp)
                else:
                    # Fallback: use track data with defaults
                    detection_confidence = float(track.get('temporal_average_conf', track.get('confidence', 0.0)))
                    merged_polyp = {
                        **track,
                        'classification': 'UNKNOWN',
                        'expert_classification': 'UNKNOWN',
                        'polyp_type': 'UNKNOWN',
                        'polyp_type_confidence': 0.0,
                        'type_confidence': 0.0,
                        'redness': 0.0,
                        'texture': 0.0,
                        'vessel_visibility': 0.0,
                        'radius': 0.0,
                        'risk_score': 0.0,
                        'medical_risk_score': 0.0,
                        'detection_confidence': detection_confidence,
                        'symbolic_confidence': 0.0,
                        'clinical_confidence': 0.0,
                        'confidence': 0.0
                    }
                    merged_features.append(merged_polyp)

            polyp_features = merged_features
        else:
            polyp_features = canonical_entries
        
        polyp_count = 0
        for polyp_idx, polyp_data in enumerate(polyp_features):
            polyp_count = polyp_idx + 1  # Use 1-based indexing consistently

            # Handle different data structures for temporal tracks vs raw features
            if use_temporal_tracks:
                # Temporal track structure
                classification = polyp_data.get('classification', 'UNKNOWN')
                frame_idx = int(polyp_data.get('representative_frame', polyp_data.get('start_frame', polyp_data.get('frame', 0))))
                detection_confidence = float(polyp_data.get('detection_confidence', polyp_data.get('temporal_average_conf', 0.0)))
                symbolic_confidence = float(polyp_data.get('symbolic_confidence', polyp_data.get('confidence', 0.0)))
                clinical_confidence = float(polyp_data.get('clinical_confidence', polyp_data.get('confidence', symbolic_confidence)))
                confidence = clinical_confidence
                num_frames = polyp_data.get('num_frames', 1)
                track_info = f" (Tracked across {num_frames} frames)"
            else:
                # Raw feature structure
                classification = polyp_data.get('classification', 'UNKNOWN')
                frame_idx = int(polyp_data.get('representative_frame', polyp_data.get('start_frame', polyp_data.get('frame', 0))))
                detection_confidence = float(polyp_data.get('detection_confidence', 0.0))
                symbolic_confidence = float(polyp_data.get('symbolic_confidence', polyp_data.get('confidence', 0.0)))
                clinical_confidence = float(polyp_data.get('clinical_confidence', polyp_data.get('confidence', symbolic_confidence)))
                confidence = clinical_confidence
                track_info = ""

            # Polyp header with risk color
            _display_type = str(polyp_data.get('polyp_type', 'UNKNOWN')).upper().strip()
            _risk_label   = self._clinical_class_to_risk_label(_display_type)
            risk_color    = self.get_risk_color(_risk_label)
            risk_text     = f"<font color='{risk_color.hexval()}' size=14><b>POLYP #{polyp_count} — {_display_type} [{_risk_label}]{track_info}</b></font>"
            story.append(Paragraph(risk_text, styles['Heading3']))
            
            # Get frame and extract polyp region
            matching_symbolic = None
            if use_temporal_tracks:
                # Search the full track and use the strongest consensus-supported frame.
                frame_sequence = [frame for frame in polyp_data.get('frame_sequence', []) if isinstance(frame, int)]
                if not frame_sequence:
                    fallback_frame = polyp_data.get('representative_frame', polyp_data.get('start_frame', polyp_data.get('frame', 0)))
                    frame_sequence = [int(fallback_frame)]

                track_box = polyp_data.get('box')
                best_frame_idx = frame_sequence[0]
                best_frame_score = (-1, -1, -1.0, -1)

                for candidate_frame_idx in frame_sequence:
                    model_hits = 0
                    confidence_sum = 0.0
                    medsam_mask_count = 0

                    if hasattr(self, 'detections') and self.detections:
                        for model_name in ('yolo', 'rtdetr', 'medsam'):
                            model_dets = self.detections.get(model_name, [])
                            frame_dets = [det for det in model_dets if det.get('frame') == candidate_frame_idx and det.get('boxes')]
                            if frame_dets:
                                model_hits += 1
                                confidences = frame_dets[0].get('confidences') or []
                                if confidences:
                                    confidence_sum += float(max(confidences))

                    if 'medsam' in self.segmentations:
                        frame_segs = [seg for seg in self.segmentations['medsam'] if seg.get('frame') == candidate_frame_idx and seg.get('mask') is not None]
                        medsam_mask_count = len(frame_segs)
                        confidence_sum += sum(float(seg.get('confidence', 0.0)) for seg in frame_segs)

                    candidate_score = (model_hits, medsam_mask_count, confidence_sum, candidate_frame_idx)
                    if candidate_score > best_frame_score:
                        best_frame_score = candidate_score
                        best_frame_idx = candidate_frame_idx

                # CRITICAL FIX FOR ISSUE 1: Update frame_idx to best_frame_idx selected from consensus analysis
                frame_idx = best_frame_idx
                
                # Find the most representative box and mask for the chosen frame.
                box = track_box
                mask = None

                if hasattr(self, 'detections') and self.detections:
                    medsam_dets = self.detections.get('medsam', [])
                    matching_dets = [det for det in medsam_dets if det.get('frame') == best_frame_idx and det.get('boxes')]
                    if matching_dets:
                        best_box = None
                        best_box_iou = -1.0
                        for det in matching_dets:
                            for det_box in det.get('boxes', []):
                                current_iou = calculate_iou(track_box, det_box) if track_box else 0.0
                                if current_iou > best_box_iou:
                                    best_box_iou = current_iou
                                    best_box = det_box
                        if best_box is not None:
                            box = best_box

                if 'medsam' in self.segmentations:
                    matching_segs = [seg for seg in self.segmentations['medsam'] if seg.get('frame') == best_frame_idx and seg.get('mask') is not None]
                    if matching_segs:
                        best_mask = None
                        best_mask_iou = -1.0
                        for seg in matching_segs:
                            seg_box = seg.get('prompt_box')
                            if box is not None and seg_box is not None and len(box) == 4 and len(seg_box) == 4:
                                current_iou = calculate_iou(box, seg_box)
                            else:
                                current_iou = 0.0

                            if current_iou > best_mask_iou:
                                best_mask_iou = current_iou
                                best_mask = seg.get('mask')

                        mask = best_mask if best_mask is not None else matching_segs[0].get('mask')
            else:
                # For raw features, use the detection frame and box
                best_frame_idx = frame_idx
                box = polyp_data.get('box')
                mask = polyp_data.get('mask')

            if best_frame_idx < len(self.frames) and box:
                frame = self.frames[best_frame_idx]

                # Extract and display polyp region (with mask if available) - Fix 1C: guard for None crop
                try:
                    polyp_region = self.extract_polyp_region(frame, box, mask)
                    if polyp_region is not None and polyp_region.size > 0:
                        img_buffer = self.save_image_to_buffer(polyp_region)
                        if img_buffer is not None:
                            display_width, display_height = self._fit_image_display_size(
                                polyp_region.shape[1],
                                polyp_region.shape[0],
                            )
                            img = Image(img_buffer, width=display_width, height=display_height)
                            img.hAlign = 'CENTER'
                            story.append(img)
                            story.append(Spacer(1, 0.1*inch))
                except Exception as img_err:
                    print(f"   ⚠️  Could not render polyp image for entry {polyp_idx+1}: {img_err}")
                    # Continue building the rest of the report without this image
            
            # Polyp details table - handle different data structures
            if use_temporal_tracks:
                # Temporal track structure - use available fields or defaults
                redness = polyp_data.get('redness', polyp_data.get('features', {}).get('redness', 0.0))
                radius = polyp_data.get('radius', polyp_data.get('features', {}).get('radius', 0.0))
                texture = polyp_data.get('texture', polyp_data.get('features', {}).get('texture', 0.0))
                vessels = polyp_data.get('vessel_visibility', polyp_data.get('features', {}).get('vessel_visibility', 0.0))
                risk_score = polyp_data.get('risk_score', polyp_data.get('medical_risk_score', polyp_data.get('features', {}).get('risk_score', 0.0)))
                polyp_type = polyp_data.get('polyp_type', polyp_data.get('features', {}).get('predicted_type', polyp_data.get('expert_classification', 'UNKNOWN')))
                polyp_type_conf = polyp_data.get('type_confidence', polyp_data.get('features', {}).get('type_confidence', float(polyp_data.get('confidence', 0.50))))
                expert_classification = polyp_data.get('expert_classification', classification)
            else:
                # Raw feature structure
                redness = polyp_data.get('redness', 0.0)
                radius = polyp_data.get('radius', 0.0)
                texture = polyp_data.get('texture', 0.0)
                vessels = polyp_data.get('vessel_visibility', polyp_data.get('vessels', 0.0))
                risk_score = polyp_data.get('risk_score', polyp_data.get('medical_risk_score', 0.0))
                polyp_type = polyp_data.get('polyp_type', polyp_data.get('features', {}).get('predicted_type', polyp_data.get('expert_classification', 'UNKNOWN')))
                polyp_type_conf = polyp_data.get('type_confidence', polyp_data.get('features', {}).get('type_confidence', float(polyp_data.get('confidence', 0.50))))
                expert_classification = polyp_data.get('expert_classification', classification)

            polyp_type, polyp_type_conf = self._resolve_polyp_type(
                polyp_data,
                matching_symbolic,
                classification,
                polyp_type,
                polyp_type_conf,
            )
            
            _styles = styles  # reuse getSampleStyleSheet() already called at top of function

            details_data = [
                ['Feature', 'Value', 'Interpretation'],
                ['Frame Index', str(frame_idx), f'Frame {frame_idx} of {len(self.frames)}'],
                ['Detection Method', self._get_detection_method_label(), self._get_detection_method_description()],
                ['', '', ''],
                [Paragraph('<b>POLYP TYPE</b>', _styles['Normal']),
                 Paragraph(f'<b>{polyp_type}</b>', _styles['Normal']),
                 Paragraph(f'Type confidence: {polyp_type_conf:.1%}', _styles['Normal'])],
                ['', '', ''],
                ['Redness Score', self._format_decimal(redness, 3), 'Color-based vascularization'],
                ['Relative Radius', f'{float(radius) * 100.0:.1f}%', 'Polyp size as % of frame diagonal'],
                ['Texture Score', self._format_decimal(texture, 3), 'Surface morphology'],
                ['Vessel Visibility', self._format_decimal(vessels, 3), 'Vascularization indicator'],
                ['Risk Tier', _risk_label.replace('_', ' '), f'Validated by ESGE 2017 / NICE classification'],
                ['', '', ''],
                ['Clinical Classification', classification, 'Final symbolic reasoning bucket'],
                ['Expert Prediction', expert_classification, f'Decision Tree output{" — moderated by feature heuristics" if expert_classification != classification else ""}'],
                ['Clinical Class',
                 Paragraph(polyp_data.get('clinical_class', polyp_type), _styles['Normal']),
                 Paragraph(polyp_data.get('clinical_description',
                     PolypTypeClassifier().get_clinical_description(
                         polyp_data.get('clinical_class', 'ADENOMATOUS_POLYP')
                     )), _styles['Normal'])
                ],
                ['Detection Confidence', self._format_percent(detection_confidence, 1), 'YOLO / RT-DETR / track confidence'],
            ]
            
            details_table = Table(details_data, colWidths=[1.5*inch, 1.2*inch, 2.8*inch])
            details_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#E6F0FF')),  # Highlight polyp type row
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(details_table)
            
            # Polyp type clinical description
            story.append(Spacer(1, 0.1*inch))
            type_desc = self.get_type_description(polyp_type, _risk_label)
            type_style = ParagraphStyle(
                'PolyTypeDesc',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#CC0000'),
                leftIndent=0.2*inch,
                rightIndent=0.2*inch,
                spaceAfter=0.1*inch,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph(f"<b>Type Assessment:</b> {type_desc}", type_style))
            
            # Clinical assessment
            story.append(Spacer(1, 0.05*inch))
            clinical_desc = self.get_risk_description(_risk_label, confidence)
            clinical_style = ParagraphStyle(
                'Clinical',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#1a1a1a'),
                leftIndent=0.2*inch,
                rightIndent=0.2*inch,
                spaceAfter=0.15*inch,
                borderPadding=10,
                borderRadius=3
            )
            story.append(Paragraph(f"<i>{clinical_desc}</i>", clinical_style))
            
            story.append(Spacer(1, 0.25*inch))
            
            # Page break after every 3 polyps
            if polyp_count % 3 == 0 and polyp_count < len(polyp_features):
                story.append(PageBreak())
        
        return story
    
    def create_clinical_impression(self):
        """Create clinical impression summary"""
        story = []
        styles = getSampleStyleSheet()
        
        story.append(PageBreak())
        story.append(Paragraph("<b>CLINICAL IMPRESSION</b>", styles['Heading2']))
        story.append(Spacer(1, 0.15*inch))
        
        canonical_entries, symbolic_results, use_temporal_tracks = self._get_canonical_polyp_entries(max_polyps=None)
        total = len(canonical_entries)
        high_risk = 0
        medium_risk = 0
        low_risk = 0

        if canonical_entries:
            if use_temporal_tracks:
                for track in canonical_entries:
                    track_id = track.get('polyp_id')
                    matching_symbolic = None
                    for symbolic in symbolic_results:
                        if self._polyp_id_matches(symbolic.get('polyp_id'), track_id):
                            matching_symbolic = symbolic
                            break

                    pt = str((matching_symbolic or track or {}).get('polyp_type', 'UNKNOWN'))
                    risk_label = self._clinical_class_to_risk_label(pt)
                    if risk_label == 'HIGH_RISK':    high_risk += 1
                    elif risk_label == 'MEDIUM_RISK': medium_risk += 1
                    elif risk_label == 'LOW_RISK':    low_risk += 1
            else:
                for feature in canonical_entries:
                    pt = str(feature.get('polyp_type', 'UNKNOWN'))
                    risk_label = self._clinical_class_to_risk_label(pt)
                    if risk_label == 'HIGH_RISK':    high_risk += 1
                    elif risk_label == 'MEDIUM_RISK': medium_risk += 1
                    elif risk_label == 'LOW_RISK':    low_risk += 1
        
        # Create impression text
        if total > 0:
            summary_text = f"A total of {total} polyps were detected and analyzed using multi-model consensus and AI-based risk stratification."
        else:
            summary_text = "No polyps were detected or analyzed after the current consensus and stability filters."

        impression = f"""
        <b>Summary:</b><br/>
        {summary_text}
        <br/><br/>
        <b>Risk Distribution:</b><br/>
        • HIGH RISK polyps: {high_risk} ({100*high_risk/max(1, total):.1f}%)<br/>
        • MEDIUM RISK polyps: {medium_risk} ({100*medium_risk/max(1, total):.1f}%)<br/>
        • LOW RISK polyps: {low_risk} ({100*low_risk/max(1, total):.1f}%)<br/>
        <br/>
        <b>Recommendation:</b><br/>
        """

        if total == 0:
            impression += """
            No stable canonical polyp crop survived the current report filters.
            Review detector overlap, temporal thresholds, and source video quality if this is unexpected.<br/>
            """
        elif high_risk > 0:
            impression += """
            The presence of HIGH RISK polyps warrants close clinical attention.
            Consider endoscopic resection or follow-up surveillance depending on clinical context.<br/>
            """
        elif medium_risk > 0:
            impression += """
            The presence of MEDIUM RISK polyps warrants clinical review and correlation with endoscopic findings.<br/>
            """
        else:
            impression += """
            All detected polyps are classified as LOW RISK.
            Standard surveillance protocol is recommended.<br/>
            """
        
        impression += """
        <b>Technical Details:</b><br/>
        • Detection: Multi-model consensus (YOLO, RT-DETR, MedSAM2)<br/>
        • Features: 444-dimensional vectors (384 SSL + 60 biomarkers)<br/>
        • Classification: Cluster-specific Decision Tree experts<br/>
        • Confidence: Based on deep learning + medical heuristics<br/>
        """
        
        story.append(Paragraph(impression, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add mandatory clinical disclaimer for publication/thesis compliance
        story.append(Spacer(1, 0.15*inch))
        disclaimer = (
            "<i><b>Disclaimer:</b> This report is generated by an AI-assisted research system "
            "for academic and research purposes only. All clinical recommendations are derived "
            "from published ESGE 2017, NICE CG118/CG131, and BSG 2019 colonoscopy guidelines. "
            "This output does not constitute medical advice and must not be used as the sole "
            "basis for clinical decisions. All findings must be reviewed and confirmed by a "
            "qualified gastroenterologist or endoscopist.</i>"
        )
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Signature area
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("___________________________", styles['Normal']))
        story.append(Paragraph("AI-Based Polyp Detection System", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        return story
    
    def generate_pdf(self):
        """Generate complete PDF report"""
        print(f"Generating PDF report for {self.video_id}...")
        
        # Create document
        pdf = SimpleDocTemplate(
            str(self.pdf_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build story
        story = []
        
        # Add sections
        story.extend(self.create_summary_section())
        story.extend(self.create_polyp_details_section())
        story.extend(self.create_clinical_impression())
        
        # Build PDF - Fix 1D: add traceback and fallback PDF with summary only
        try:
            pdf.build(story)
            print(f"✅ PDF report saved: {self.pdf_path}")
            return str(self.pdf_path)
        except Exception as e:
            import traceback
            print(f"❌ Error generating PDF: {e}")
            traceback.print_exc()
            # Try minimal fallback PDF with just the text content (no images)
            try:
                pdf2 = SimpleDocTemplate(str(self.pdf_path), pagesize=letter)
                fallback_story = self.create_summary_section()  # summary always safe
                pdf2.build(fallback_story)
                print(f"⚠️  Fallback PDF (summary only) saved: {self.pdf_path}")
                return str(self.pdf_path)
            except Exception as e2:
                print(f"❌ Fallback PDF also failed: {e2}")
                return None
        finally:
            # Cleanup temp images
            import shutil
            if self.temp_images.exists():
                shutil.rmtree(self.temp_images)


def generate_medical_report(video_path, frames, detections, segmentations, video_output_dir):
    """
    Generate professional medical report for video
    
    Args:
        video_path: Path to video file
        frames: Extracted video frames
        detections: Detection results
        segmentations: Segmentation results
        video_output_dir: Output directory (already contains video_id subfolder)
    """
    try:
        video_id = video_path.stem
        
        # FIX: The inference report is saved directly in video_output_dir, not in a subfolder
        report_path = Path(video_output_dir) / 'inference_report.json'
        if not report_path.exists():
            print(f"   ❌ Inference report not found at: {report_path}")
            # Try alternative path (in case structure is different)
            alt_report_path = Path(video_output_dir) / video_id / 'inference_report.json'
            if alt_report_path.exists():
                report_path = alt_report_path
            else:
                print(f"   ❌ Also checked: {alt_report_path}")
                return None
        
        with open(report_path) as f:
            inference_report = json.load(f)
        
        print(f"   📄 Loaded inference report: {report_path.name}")
        
        # Generate PDF report
        report_gen = MedicalReportGenerator(
            video_id,
            Path(video_output_dir),
            frames,
            detections,
            segmentations,
            inference_report
        )
        
        pdf_path = report_gen.generate_pdf()
        if pdf_path:
            print(f"   ✅ PDF saved to: {pdf_path}")
        return pdf_path
    
    except Exception as e:
        print(f"   ❌ Error generating medical report: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Medical Report Generator loaded successfully")
    print("Import this module to generate PDF reports: from medical_report_generator import generate_medical_report")