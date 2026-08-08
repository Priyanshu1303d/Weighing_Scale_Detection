import pytest
import numpy as np
from weighing_scale_detection.detector.primary_selector import PrimaryScaleSelector

def test_single_detection():
    """Single detection should always be primary"""
    selector = PrimaryScaleSelector()
    
    detections = [{
        'bbox': [100, 100, 200, 200],
        'area': 10000,
        'confidence': 0.9
    }]
    
    primary = selector.resolve_primary_scale(detections, (480, 640, 3))
    
    assert primary is not None
    assert primary['primary_score'] == 1.0
    assert primary['is_primary'] == True

def test_largest_scale_wins():
    """Larger scale should be prioritized"""
    selector = PrimaryScaleSelector()
    
    detections = [
        {'bbox': [50, 50, 100, 100], 'area': 2500, 'confidence': 0.8},  # Small
        {'bbox': [200, 200, 400, 400], 'area': 40000, 'confidence': 0.85}  # Large
    ]
    
    primary = selector.resolve_primary_scale(detections, (480, 640, 3))
    
    assert primary['area'] == 40000  # Largest should win

def test_explanation_generation():
    """Should generate human-readable explanation"""
    selector = PrimaryScaleSelector()
    
    detection = {
        'primary_score': 0.85,
        'score_breakdown': {
            'area': 0.9,
            'centrality': 0.8,
            'contrast': 0.7,
            'vertical': 0.9
        }
    }
    
    explanation = selector.explain_decision(detection)
    
    assert "PRIMARY SCALE SELECTION" in explanation
    assert "area" in explanation.lower()