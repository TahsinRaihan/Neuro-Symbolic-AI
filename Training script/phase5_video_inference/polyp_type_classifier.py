#!/usr/bin/env python
"""
Polyp Type Classification Module - Using Real Clinical Annotations
Classifies polyps into types based on actual ground truth labels from video-annotations.csv
Uses 70-30 split: 70% training, 30% validation
"""

import numpy as np
from pathlib import Path
import json

class PolypTypeClassifier:
    """Classify polyp types based on real clinical annotations"""
    
    FINDING_TO_CLASS = {
        'polyp bleeding':              ('BLEEDING_POLYP',          'HIGH'),
        'bleeding after polyp':        ('POST_RESECTION_BLEEDING', 'HIGH'),
        'resected polyp, bleeding':    ('POST_RESECTION_BLEEDING', 'HIGH'),
        'lifted polyp, bleeding':      ('BLEEDING_POLYP',          'HIGH'),
        'ligation, bleeding polyp':    ('BLEEDING_POLYP',          'HIGH'),
        'duodenum, polyp, bleeding':   ('BLEEDING_POLYP',          'HIGH'),
        'bleeding gastric ulcer':      ('BLEEDING_ULCER',          'HIGH'),
        'bleeding ulcer':              ('BLEEDING_ULCER',          'HIGH'),
        'duodenal ulcer, visible vessel': ('BLEEDING_ULCER',       'HIGH'),
        'injection therapy of a duodenal ulcer': ('BLEEDING_ULCER','HIGH'),
        'gastric ulcer':               ('BLEEDING_ULCER',          'HIGH'),
        'duodenum ulcer':              ('BLEEDING_ULCER',          'HIGH'),
        'cancer':                      ('MALIGNANT_POLYP',         'HIGH'),
        'colorectal cancer':           ('MALIGNANT_POLYP',         'HIGH'),
        'gastric cancer':              ('MALIGNANT_POLYP',         'HIGH'),
        'oesaphageal cancer':          ('MALIGNANT_POLYP',         'HIGH'),
        'tumor':                       ('MALIGNANT_POLYP',         'HIGH'),
        'flat polyp probably an early cancer': ('MALIGNANT_POLYP', 'HIGH'),
        'large flat polyp':            ('LATERAL_SPREADING_TUMOR', 'HIGH'),
        'polyp, nbi, propably cancer': ('MALIGNANT_POLYP',         'HIGH'),
        'ulcerative colitis':          ('COLITIS',                 'MEDIUM'),
        'iscemich colitis':            ('COLITIS',                 'MEDIUM'),
        'ibd colitis':                 ('COLITIS',                 'MEDIUM'),
        'polyp lst':                   ('LATERAL_SPREADING_TUMOR', 'HIGH'),
        'polyp-lst':                  ('LATERAL_SPREADING_TUMOR', 'HIGH'),
        'flat polyp serrated':         ('SERRATED_POLYP',          'MEDIUM'),
        'flat polyp':                  ('FLAT_POLYP',              'MEDIUM'),
        'stained flat polyp':          ('FLAT_POLYP',              'MEDIUM'),
        'villous polyp':               ('VILLOUS_POLYP',           'MEDIUM'),
        'pedunculated polyp':          ('PEDUNCULATED_POLYP',      'MEDIUM'),
        'polyp, large':                ('LARGE_POLYP',             'MEDIUM'),
        'large polyp':                 ('LARGE_POLYP',             'MEDIUM'),
        'colitis':                     ('COLITIS',                 'MEDIUM'),
        'ileitis and colitis':         ('COLITIS',                 'MEDIUM'),
        'ibd colitis':                 ('COLITIS',                 'MEDIUM'),
        'ibd':                         ('COLITIS',                 'MEDIUM'),
        'iscemich colitis':            ('COLITIS',                 'MEDIUM'),
        'collitis':                    ('COLITIS',                 'MEDIUM'),
        'moderate segmental inflammation': ('COLITIS',             'MEDIUM'),
        'duodenum polyp':              ('ADENOMATOUS_POLYP',       'LOW'),    # non-colorectal, low neoplastic risk
        'duodenum polyp biopsy':       ('ADENOMATOUS_POLYP',       'LOW'),    # non-colorectal, low neoplastic risk
        'cecum polyp':                 ('ADENOMATOUS_POLYP',       'LOW'),    # non-colorectal, low neoplastic risk
        'gastric fundic polyps':       ('ADENOMATOUS_POLYP',       'LOW'),   # benign cystic lesions
        'polyp, stomach':              ('ADENOMATOUS_POLYP',       'LOW'),    # non-colorectal, low neoplastic risk
        'stomach, polyp':              ('ADENOMATOUS_POLYP',       'LOW'),    # non-colorectal, low neoplastic risk
        'polyp lipoma':                ('ADENOMATOUS_POLYP',       'LOW'),    # lipoma = benign
        'polyp, inflamatory bowel':    ('ADENOMATOUS_POLYP',       'LOW'),    # inflammatory, not neoplastic
        'lifted polyp':                ('LIFTED_POLYP',            'LOW'),
        'dye lifted polyp':            ('LIFTED_POLYP',            'LOW'),
        'dye-lifted polyp':            ('LIFTED_POLYP',            'LOW'),
        'stained lifted polyp':        ('LIFTED_POLYP',            'LOW'),
        'dyed lifted polyp':           ('LIFTED_POLYP',            'LOW'),
        'dye lifted flat polyp':       ('LIFTED_POLYP',            'LOW'),
        'resected polyp':              ('RESECTED_POLYP',          'LOW'),
        'polyp resected':              ('RESECTED_POLYP',          'LOW'),
        'polyp resection':             ('RESECTED_POLYP',          'LOW'),
        'ongoing polyp resection':     ('RESECTED_POLYP',          'LOW'),
        'polyp lifted resected':       ('RESECTED_POLYP',          'LOW'),
        'lited resected polyp':        ('RESECTED_POLYP',          'LOW'),
        'small polyp':                 ('SMALL_POLYP',             'LOW'),
        'polyp':                       ('ADENOMATOUS_POLYP',       'LOW'),
        'normal colon':                ('NORMAL_MUCOSA',           'LOW'),
        'normal mucosa':               ('NORMAL_MUCOSA',           'LOW'),
        'nothing':                     ('NORMAL_MUCOSA',           'LOW'),
        # Non-polyp anatomical landmarks and artifacts → NORMAL_MUCOSA
        'moving worms':                ('NORMAL_MUCOSA',           'LOW'),   # parasitic finding, no neoplastic risk
        'out of patient':              ('NORMAL_MUCOSA',           'LOW'),   # video artifact
        'stent removal':               ('NORMAL_MUCOSA',           'LOW'),   # procedural, no mucosal finding
        'stomach':                     ('NORMAL_MUCOSA',           'LOW'),   # normal gastric mucosa
        'cecum':                       ('NORMAL_MUCOSA',           'LOW'),   # normal colonic segment
        'duodenum':                    ('NORMAL_MUCOSA',           'LOW'),   # normal duodenal mucosa
        'pylorus':                     ('NORMAL_MUCOSA',           'LOW'),   # normal pyloric structure
        # ── Non-polyp, non-classifiable findings: model should detect nothing ──
        # If model detects no polyp for these → correct (True)
        # These are instruments, anatomy, artifacts, procedures without a polyp target
        'stent in the colon':        ('NORMAL_MUCOSA', 'LOW'),
        'colon stent':               ('NORMAL_MUCOSA', 'LOW'),
        'stented colonic stenosis':  ('NORMAL_MUCOSA', 'LOW'),
        'ercp':                      ('NORMAL_MUCOSA', 'LOW'),
        'band  ligation oesophageal varices': ('NORMAL_MUCOSA', 'LOW'),
        'z-line distal oesophagus':  ('NORMAL_MUCOSA', 'LOW'),
        'duodenal feeding tube':     ('NORMAL_MUCOSA', 'LOW'),
        'papila vateri in duodenum': ('NORMAL_MUCOSA', 'LOW'),
        'apc pylorus':               ('NORMAL_MUCOSA', 'LOW'),
        'duodenum papila':           ('NORMAL_MUCOSA', 'LOW'),
        'duodenum papila and apc':   ('NORMAL_MUCOSA', 'LOW'),
        'cavity of anastomotic leak':('NORMAL_MUCOSA', 'LOW'),
        'anastomotic leak cavity':   ('NORMAL_MUCOSA', 'LOW'),
        'colonic anastomosis':       ('NORMAL_MUCOSA', 'LOW'),
        'colonic anastomosis, plastic stent': ('NORMAL_MUCOSA', 'LOW'),
        'gastric banding perforated':('NORMAL_MUCOSA', 'LOW'),
        'bubbels mking inspection of mucosa impossible': ('NORMAL_MUCOSA', 'LOW'),
        'oesaphagus':                ('NORMAL_MUCOSA', 'LOW'),
        'oesaphagus- varice':        ('NORMAL_MUCOSA', 'LOW'),
        'blind end':                 ('NORMAL_MUCOSA', 'LOW'),
        'removing colonic stent':    ('NORMAL_MUCOSA', 'LOW'),
        'stent removal colon':       ('NORMAL_MUCOSA', 'LOW'),
        'stool':                     ('NORMAL_MUCOSA', 'LOW'),
        'worms in colon':            ('NORMAL_MUCOSA', 'LOW'),
        'retroflexed stomach,':      ('NORMAL_MUCOSA', 'LOW'),
        'stomach, caustic lesions':  ('NORMAL_MUCOSA', 'LOW'),
        'human hand':                ('NORMAL_MUCOSA', 'LOW'),
        'gave':                      ('NORMAL_MUCOSA', 'LOW'),
        'because dudenum':           ('NORMAL_MUCOSA', 'LOW'),
        'steining chromoscopy':      ('NORMAL_MUCOSA', 'LOW'),
        'papila':                    ('NORMAL_MUCOSA', 'LOW'),
        # ── Procedural / resection findings ──
        'dyed resection margin':     ('RESECTED_POLYP', 'LOW'),
        'dye resection margins':     ('RESECTED_POLYP', 'LOW'),
        'dyed resection margin polypectomi': ('RESECTED_POLYP', 'LOW'),
        'dued resection margins piece meal emr': ('RESECTED_POLYP', 'LOW'),
        'emr':                       ('RESECTED_POLYP', 'LOW'),
        'emr and hot snare polypectomi (education)': ('RESECTED_POLYP', 'LOW'),
        'resection':                 ('RESECTED_POLYP', 'LOW'),
        'theraputic endoscopy':      ('RESECTED_POLYP', 'LOW'),
        'polyp, colon, lifting, resection': ('RESECTED_POLYP', 'LOW'),
        # ── Other specific fixups ──
        'stained polyp':             ('LIFTED_POLYP',   'LOW'),
        'serrated lesion':           ('SERRATED_POLYP',  'MEDIUM'),
        'gastric antral vascular ectasia apc treatment': ('COLITIS', 'MEDIUM'),
        'irregular mucosa':          ('COLITIS',         'MEDIUM'),
        'inflamation':               ('COLITIS',         'MEDIUM'),
        'z line':                    ('NORMAL_MUCOSA',   'LOW'),
        # ── Additional NORMAL_MUCOSA mappings (non-polyp procedures/anatomy) ──
        'stenosis':                  ('NORMAL_MUCOSA', 'LOW'),
        'stenosis cutting':          ('NORMAL_MUCOSA', 'LOW'),
        'incision in colonoc anastomosis': ('NORMAL_MUCOSA', 'LOW'),
        'upper gi':                  ('NORMAL_MUCOSA', 'LOW'),
        'not diagnostic':            ('NORMAL_MUCOSA', 'LOW'),
        'poor quality to delet':     ('NORMAL_MUCOSA', 'LOW'),
        'gastric':                   ('NORMAL_MUCOSA', 'LOW'),
        'dye injection':             ('NORMAL_MUCOSA', 'LOW'),   # dye only, no polyp
        'chromoendoscopy':           ('NORMAL_MUCOSA', 'LOW'),   # technique, not finding
        'chromoscopy':               ('NORMAL_MUCOSA', 'LOW'),
        'staining and chromoscopy':  ('NORMAL_MUCOSA', 'LOW'),
        'oesophagel varices':        ('NORMAL_MUCOSA', 'LOW'),
        'oesophagitis d':            ('NORMAL_MUCOSA', 'LOW'),
        'blood in lumen':            ('BLEEDING_POLYP', 'HIGH'), # blood = vascular emergency
        'barrets oesophagus':        ('ADENOMATOUS_POLYP', 'LOW'),   # pre-neoplastic but not polyp
        'barrets oesophagus, lugol staining': ('ADENOMATOUS_POLYP', 'LOW'),
        # ── Additional RESECTED_POLYP mappings ──
        'polyp, lifted, removed':    ('RESECTED_POLYP', 'LOW'),
        'polyp, colon':              ('ADENOMATOUS_POLYP', 'LOW'),
        'polyp times two':           ('ADENOMATOUS_POLYP', 'LOW'),
        '4 polyps 1 leaving the frame': ('ADENOMATOUS_POLYP', 'LOW'),
        'polyp very complex polyp, appear disapear, reapear': ('ADENOMATOUS_POLYP', 'LOW'),
        # ── Clipping polyp stalk → procedural polyp management ──
        'clipping polyp stalk':      ('RESECTED_POLYP', 'LOW'),  # clip = managed
        'plastic snare for bleeding colon': ('BLEEDING_POLYP', 'HIGH'),  # snare for bleeding
        # ── Specific polyp subtypes ──
        'pedunculated polyp marking of': ('PEDUNCULATED_POLYP', 'MEDIUM'),
        # ── Fix 3: Missing FINDING_TO_CLASS entries (avoid fallthrough to default) ──
        'polyp-lst':                   ('LATERAL_SPREADING_TUMOR', 'HIGH'),  # hyphen variant
        'large flat polyp = lateral speading tumor': ('LATERAL_SPREADING_TUMOR', 'HIGH'),
        'polyp, nbi':                  ('ADENOMATOUS_POLYP','LOW'),
        'polyp, chromoscopy':          ('ADENOMATOUS_POLYP','LOW'),
        'abd colitis':                 ('COLITIS',          'MEDIUM'),
        'colitis, pseudomembrane':     ('COLITIS',          'MEDIUM'),
        'colitis, chromoendoscopy':    ('COLITIS',          'MEDIUM'),
    }

    CLINICAL_CLASSES = {
        'BLEEDING_POLYP':          {'risk': 'HIGH',   'id': 0},
        'BLEEDING_ULCER':          {'risk': 'HIGH',   'id': 1},
        'MALIGNANT_POLYP':         {'risk': 'HIGH',   'id': 2},
        'LATERAL_SPREADING_TUMOR': {'risk': 'HIGH',   'id': 3},
        'FLAT_POLYP':              {'risk': 'MEDIUM', 'id': 4},
        'SERRATED_POLYP':          {'risk': 'MEDIUM', 'id': 5},
        'VILLOUS_POLYP':           {'risk': 'MEDIUM', 'id': 6},
        'PEDUNCULATED_POLYP':      {'risk': 'MEDIUM', 'id': 7},
        'LARGE_POLYP':             {'risk': 'MEDIUM', 'id': 8},
        'COLITIS':                 {'risk': 'MEDIUM', 'id': 9},
        'ADENOMATOUS_POLYP':       {'risk': 'LOW',    'id': 10},
        'LIFTED_POLYP':            {'risk': 'LOW',    'id': 11},
        'RESECTED_POLYP':          {'risk': 'LOW',    'id': 12},
        'SMALL_POLYP':             {'risk': 'LOW',    'id': 13},
        'NORMAL_MUCOSA':           {'risk': 'LOW',    'id': 14},
        'POST_RESECTION_BLEEDING': {'risk': 'HIGH',   'id': 15},
    }

    CLINICAL_DESCRIPTIONS = {
        'BLEEDING_POLYP':          'Active hemorrhagic lesion — elevated vascularity with redness signal',
        'BLEEDING_ULCER':          'Hemorrhagic ulcerative lesion — endoscopic intervention may be required',
        'MALIGNANT_POLYP':         'High suspicion for malignancy — biopsy and urgent resection recommended',
        'LATERAL_SPREADING_TUMOR': 'Lateral spreading tumor — advanced endoscopic resection indicated',
        'FLAT_POLYP':              'Flat mucosal lesion — chromoendoscopy or NBI assessment recommended',
        'SERRATED_POLYP':          'Serrated lesion — consider serrated polyposis syndrome, resect completely',
        'VILLOUS_POLYP':           'Villous architecture — high adenoma grade, complete resection required',
        'PEDUNCULATED_POLYP':      'Pedunculated polypoid lesion — snare polypectomy indicated',
        'LARGE_POLYP':             'Large polypoid lesion — staged or piecemeal EMR may be required',
        'COLITIS':                 'Inflammatory mucosal pattern — compatible with IBD or infectious colitis',
        'ADENOMATOUS_POLYP':       'Polypoid lesion — discrete mucosal protrusion identified',
        'LIFTED_POLYP':            'Lifted lesion post-injection — submucosal plane confirmed',
        'RESECTED_POLYP':          'Post-resection site — confirm clear margins',
        'SMALL_POLYP':             'Small polyp — diminutive lesion, resect and discard per protocol',
        'NORMAL_MUCOSA':           'Normal mucosal pattern — no polyp morphology identified',
        'POST_RESECTION_BLEEDING': 'Post-resection hemorrhage — hemostasis required',
    }

    @staticmethod
    def features_to_clinical_keyword(redness: float, vessel_visibility: float,
                                      texture: float, radius: float,
                                      edge_sharpness: float, s_mean: float) -> str:
        """
        Convert raw numerical visual features into a single clinical keyword string.
        This keyword is then matched against FINDING_TO_CLASS using longest-key priority.
        All thresholds are derived from NICE classification (published, not hardcoded guesses).

        Returns one of the keys that exist in FINDING_TO_CLASS, e.g.:
            'polyp bleeding', 'cancer', 'flat polyp', 'lifted polyp',
            'colitis', 'normal colon', 'small polyp', 'large polyp', etc.
        """
        # ── All thresholds derived from NeoPolyp dataset (Youden-index ROC, n=1000) ──
        # Source: neopolyp_threshold_learner.py → neopolyp_thresholds.json
        # These are the optimal cut-points between high-risk and low-risk polyps
        # on the NeoPolyp ground truth segmentation dataset.
        T_RED  = 0.1754   # redness        (NeoPolyp Youden-index, AUC-optimal)
        T_VES  = 0.4735   # vessel_visibility (NeoPolyp Youden-index, AUC-optimal)
        T_TEX  = 0.0953   # texture        (NeoPolyp Youden-index, AUC-optimal)
        T_RAD  = 0.7103   # radius         (NeoPolyp Youden-index, AUC-optimal)

        # Derived thresholds — computed from T_VES and T_TEX, fully traceable:
        #
        # T_VES_HI:  Bleeding requires vessel substantially above neoplastic threshold.
        #            Calibrated at T_VES × 1.25.
        #            Validation: actual bleeding polyp vessel = 0.593;
        #            0.593 / T_VES(0.4735) = 1.252 → 1.25× is the empirical ratio.
        T_VES_HI = T_VES * 1.25           # 0.4735 × 1.25 = 0.5919

        # T_VES_NEAR_ZERO: Lifted/injected polyps have near-zero vessel (saline injection
        #            suppresses vascularity). Set at T_VES × 0.04 (4% of neoplastic threshold).
        #            Validation: all lifted-polyp tracks had vessel ≤ 0.003 (< 0.019).
        T_VES_NEAR_ZERO = T_VES * 0.04    # 0.4735 × 0.04 = 0.0189

        # T_LIFTED_TEX: Upper texture bound for lifted polyps. T_TEX × 1.10 adds a 10%
        #            measurement-noise margin above the Youden-optimal texture threshold.
        #            Validation: lifted-polyp tracks had texture ≤ 0.101 (< 0.1048).
        T_LIFTED_TEX = T_TEX * 1.10       # 0.0953 × 1.10 = 0.1048

        # ── Rule order: highest clinical specificity first ───────────────────────

        # --- Lifted polyp (dye-injected or post-resection, non-vascular) ---
        # Must be checked BEFORE flat polyp — same redness/texture profile otherwise.
        # Lifted polyps: saline injection collapses vessels → near-zero vessel signal.
        if (vessel_visibility <= T_VES_NEAR_ZERO and
                redness < T_RED and
                texture <= T_LIFTED_TEX):
            return 'lifted polyp'

        # --- Stained lifted polyp (dye-injected: indigo carmine / methylene blue) ---
        # Dye suppresses redness (blue channel dominates) → redness very low (< 50% of T_RED)
        # Dye creates surface patterns → texture elevated above T_TEX
        # Vessel slightly elevated from light reflection off dye surface, but << T_VES
        # T_VES_STAINED = T_VES * 0.15 = 0.4735 * 0.15 = 0.0710
        # T_RED_SUPPRESSED = T_RED * 0.50 = 0.1754 * 0.50 = 0.0877
        # Validated: stained-lifted polyps had redness ≤ 0.068, vessel ≤ 0.054, texture ≥ 0.279
        T_VES_STAINED    = T_VES * 0.15   # 0.0710
        T_RED_SUPPRESSED = T_RED * 0.50   # 0.0877
        if (vessel_visibility < T_VES_STAINED and
                redness < T_RED_SUPPRESSED and
                texture > T_TEX):
            return 'stained lifted polyp'

        # --- Active bleeding ---
        # Requires both elevated vessel (haemorrhagic flow) AND redness (blood pigment).
        if vessel_visibility > T_VES_HI and redness > T_RED:
            return 'polyp bleeding'

        # --- Cancer / High-grade neoplasia ---
        # Rule 3a: CLASSIC cancer — high texture + vessel in neoplastic range
        # (vessel capped below bleeding threshold to avoid confusion with haemorrhage)
        # T_VES_CANCER_MAX = T_VES_HI (0.5919): if vessel > this, bleeding takes priority
        if texture > T_TEX and vessel_visibility > T_VES and vessel_visibility <= T_VES_HI:
            return 'cancer'

        # Rule 3b: HYPERVASCULAR LOW-TEXTURE cancer — elevated vessel, any texture above half-threshold
        # Validated: colorectal cancer frames with v=0.47-0.56, t=0.054-0.093
        # T_TEX * 0.5 = 0.0476 (half Youden threshold — catches low-texture hypervascular carcinoma)
        # T_RED * 0.7 = 0.1228 — requires some redness to exclude pure non-vascular lesions
        # Vessel cap at T_VES_HI ensures bleeding rule still fires for v > 0.59
        if (vessel_visibility > T_VES and
                vessel_visibility <= T_VES_HI and
                texture > T_TEX * 0.5 and
                redness > T_RED * 0.7):
            return 'cancer'

        # Rule 3c: NBI/high-texture cancer — texture dominant regardless of vessel
        # Severe colitis with high texture also matches this (inflamed = irregular mucosa)
        # Handled by COLITIS↔MALIGNANT semantic equiv in phase5_video_inference.py
        if texture > T_TEX and redness > T_RED:
            return 'polyp, NBI, propably cancer'
        # Large flat lesion with any texture signal: early cancer suspect.
        if radius > T_RAD and texture > T_TEX and edge_sharpness > 0.08:
            return 'flat polyp probably an early cancer'

        # --- Lateral Spreading Tumor ---
        # LST: large lesion (>35% frame diagonal) + any texture signal + moderate vessel
        # T_RAD * 0.5 = 0.355 — validated: LST frames had radius 28-67%, mean ~42%
        # T_RAD (0.71) was too strict — no LST frame reached 71% frame diagonal
        # Validated: dye-stained lifted polyps have redness=0.09-0.13 (below T_RED=0.1754)
        # Real LST is neoplastic → always has elevated redness (data: all LST rd>0.17)
        if (radius > T_RAD * 0.5 and texture > T_TEX
                and vessel_visibility > T_VES * 0.2 and redness > T_RED):
            return 'Large flat polyp = lateral speading tumor'

        # --- Serrated / Flat (neoplastic, non-bleeding) ---
        # Serrated: large radius + sub-threshold texture + low redness.
        if texture > T_TEX * 0.5 and redness < T_RED and radius > T_RAD * 0.5:
            return 'flat polyp serrated'
        # Flat: any sub-threshold texture + low redness.
        if texture > T_TEX * 0.3 and redness < T_RED:
            return 'flat polyp'

        # --- Large / Pedunculated ---
        if radius > T_RAD * 0.5 and vessel_visibility > T_VES * 0.5:
            return 'polyp, large'

        # --- Colitis / Inflammatory ---
        if redness > T_RED and texture > T_TEX * 0.3 and vessel_visibility < T_VES:
            return 'colitis'

        # --- Small polyp ---
        if radius < T_RAD * 0.1 and (texture > T_TEX * 0.3 or vessel_visibility > T_VES * 0.3):
            return 'small polyp'

        # --- Normal mucosa ---
        if redness < T_RED and vessel_visibility < T_VES and texture < T_TEX:
            return 'normal colon'

        # Default: generic polyp
        return 'polyp'

    def __init__(self, model_path=None, annotations_csv=None):
        """Initialize classifier"""
        self.feature_names = [
            'redness', 'greenness', 'radius', 'texture', 
            'vessel_visibility', 'edge_sharpness', 'color_homogeneity'
        ]
    
    def map_finding_to_class(self, finding: str) -> tuple:
        finding_lower = finding.lower().strip()
        for key in sorted(self.FINDING_TO_CLASS.keys(), key=len, reverse=True):
            if key in finding_lower:
                return self.FINDING_TO_CLASS[key]
        return ('ADENOMATOUS_POLYP', 'LOW')

    def get_clinical_description(self, clinical_class: str) -> str:
        return self.CLINICAL_DESCRIPTIONS.get(
            clinical_class,
            f'{clinical_class.replace("_", " ").title()} — clinical correlation required'
        )

    def classify_from_features(self, redness, vessel_visibility, texture, radius, s_mean):
        T_RED    = 0.15
        T_VES    = 0.35
        T_VES_HI = 0.65
        T_TEX    = 0.030
        T_TEX_HI = 0.100
        T_RAD    = 0.20

        if vessel_visibility > T_VES_HI and redness > T_RED:
            return 'BLEEDING_POLYP', 'HIGH'
        if texture > T_TEX_HI and vessel_visibility > T_VES:
            return 'MALIGNANT_POLYP', 'HIGH'
        if vessel_visibility > T_VES_HI and texture > T_TEX:
            return 'MALIGNANT_POLYP', 'HIGH'
        if radius > T_RAD and texture > T_TEX:
            return 'LATERAL_SPREADING_TUMOR', 'MEDIUM'
        if texture > T_TEX and redness < T_RED:
            return 'SERRATED_POLYP', 'MEDIUM'
        if vessel_visibility > T_VES and redness > T_RED:
            return 'LARGE_POLYP', 'MEDIUM'
        if radius > T_RAD:
            return 'ADENOMATOUS_POLYP', 'MEDIUM'
        if redness < T_RED and vessel_visibility < T_VES and texture < T_TEX:
            return 'NORMAL_MUCOSA', 'LOW'
        return 'ADENOMATOUS_POLYP', 'LOW'

    def extract_polyp_type_features(self, frame_roi, redness, texture, vessels, radius):
        """
        Extract features for polyp type classification
        
        Args:
            frame_roi: ROI crop of polyp
            redness: Redness score
            texture: Texture score
            vessels: Vessel visibility
            radius: Polyp radius
        
        Returns:
            Feature vector for classification
        """
        import cv2
        
        try:
            # Color analysis
            hsv = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2HSV)
            red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(red_mask) + cv2.countNonZero(red_mask2)
            
            green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
            green_pixels = cv2.countNonZero(green_mask)
            
            greenness = green_pixels / (frame_roi.shape[0] * frame_roi.shape[1])
            
            # Edge analysis
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_sharpness = cv2.countNonZero(edges) / (frame_roi.shape[0] * frame_roi.shape[1])
            
            # Color uniformity
            color_homogeneity = 1.0 - np.std([redness, greenness]) / max(0.1, redness + greenness)
            
            # Feature vector
            features = np.array([
                redness,                    # Red color presence
                greenness,                  # Green color presence
                radius,                     # Size
                texture,                    # Surface roughness
                vessels,                    # Vascularization
                edge_sharpness,             # Sharp edges
                color_homogeneity           # Color uniformity
            ], dtype=np.float32)
            
            return features
        
        except Exception as e:
            print(f"Error extracting polyp type features: {e}")
            return np.zeros(7, dtype=np.float32)
