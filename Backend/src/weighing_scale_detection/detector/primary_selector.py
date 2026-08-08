"""
Intelligent primary scale selection using multi-factor scoring.
Solves the business problem: "Which scale shows the bag's weight?"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

class PrimaryScaleSelector:
    """
    Selects the most relevant weighing scale from multiple detections.
    
    Business Logic:
    - In laundry operations, the active scale is typically:
        1. Largest in frame (being actively used)
        2. Most centered (operator focus)
        3. Highest contrast (readable display)
        4. Lower in frame (on counter/table)
    """
    
    def __init__(
        self,
        area_weight: float = 0.35,
        centrality_weight: float = 0.25,
        contrast_weight: float = 0.20,
        vertical_position_weight: float = 0.20
    ):
        """
        Initialize with configurable scoring weights.
        
        Args:
            area_weight: Importance of scale size (0-1)
            centrality_weight: Importance of center position (0-1)
            contrast_weight: Importance of display readability (0-1)
            vertical_position_weight: Importance of lower position (0-1)
        """
        # Validate weights sum to 1.0
        total = area_weight + centrality_weight + contrast_weight + vertical_position_weight
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"
        
        self.weights = {
            'area': area_weight,
            'centrality': centrality_weight,
            'contrast': contrast_weight,
            'vertical': vertical_position_weight
        }
    
    def resolve_primary_scale(
        self,
        detections: List[Dict],
        img_shape: Tuple[int, int, int],
        image: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Identify the primary (relevant) scale from multiple detections.
        
        Args:
            detections: List of detection dicts from ScaleDetector
            img_shape: Image dimensions (height, width, channels)
            image: Optional full image for contrast calculation
            
        Returns:
            Detection dict with highest composite score, or None if no detections
        """
        if not detections:
            return None
        
        if len(detections) == 1:
            detections[0]['primary_score'] = 1.0
            detections[0]['score_breakdown'] = {
                'area': 1.0, 'centrality': 1.0, 
                'contrast': 1.0, 'vertical': 1.0
            }
            detections[0]['is_primary'] = True
            return detections[0]
        
        img_h, img_w = img_shape[:2]
        img_center = np.array([img_w / 2, img_h / 2])
        
        scored_detections = []
        
        for det in detections:
            scores = {}
            
            # 1. AREA SCORE (Normalized by image size)
            bbox_area = det['area']
            normalized_area = bbox_area / (img_h * img_w)
            scores['area'] = min(normalized_area * 10, 1.0)  # Cap at 1.0
            
            # 2. CENTRALITY SCORE (Inverse distance from center)
            det_center = self._get_bbox_center(det['bbox'])
            max_dist = np.linalg.norm(img_center)  # Max possible distance
            actual_dist = np.linalg.norm(det_center - img_center)
            scores['centrality'] = 1 - (actual_dist / max_dist)
            
            # 3. CONTRAST SCORE (If image provided)
            if image is not None:
                scores['contrast'] = self._calculate_contrast_score(
                    image, det['bbox']
                )
            else:
                scores['contrast'] = 0.5  # Neutral score
            
            # 4. VERTICAL POSITION SCORE (Lower is better)
            bbox_vertical_center = det_center[1]  # Y coordinate
            scores['vertical'] = bbox_vertical_center / img_h
            
            # COMPOSITE SCORE
            composite_score = sum(
                scores[key] * self.weights[key]
                for key in scores
            )
            
            det['primary_score'] = composite_score
            det['score_breakdown'] = scores
            det['is_primary'] = False  # Will set True for winner
            
            scored_detections.append(det)
        
        # Select highest scoring detection
        primary_detection = max(scored_detections, key=lambda x: x['primary_score'])
        primary_detection['is_primary'] = True
        
        return primary_detection
    
    def _get_bbox_center(self, bbox: List[float]) -> np.ndarray:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def _calculate_contrast_score(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> float:
        """
        Calculate contrast/sharpness of the detected region.
        Higher contrast = more readable display.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop region
        roi = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Calculate standard deviation (measure of contrast)
        std_dev = np.std(gray)
        
        # Normalize (typical std_dev range is 0-60 for digital displays)
        normalized_contrast = min(std_dev / 60.0, 1.0)
        
        return normalized_contrast
    
    def explain_decision(self, primary_detection: Dict) -> str:
        """
        Generate human-readable explanation of why this scale was chosen.
        Useful for debugging and stakeholder trust.
        """
        if not primary_detection:
            return "No scales detected"
        
        breakdown = primary_detection.get('score_breakdown', {})
        score = primary_detection.get('primary_score', 0)
        
        explanation = f"""
PRIMARY SCALE SELECTION (Score: {score:.3f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        reasons = []
        if breakdown.get('area', 0) > 0.7:
            reasons.append("✓ Largest scale in frame")
        if breakdown.get('centrality', 0) > 0.7:
            reasons.append("✓ Centrally positioned")
        if breakdown.get('contrast', 0) > 0.6:
            reasons.append("✓ High contrast display")
        if breakdown.get('vertical', 0) > 0.6:
            reasons.append("✓ Lower position (typical counter height)")
        
        explanation += "\n".join(reasons)
        
        explanation += f"\n\nScore Breakdown:"
        for key, value in breakdown.items():
            bar = "█" * int(value * 20)
            explanation += f"\n  {key.capitalize():12s} [{bar:<20s}] {value:.2%}"
        
        return explanation