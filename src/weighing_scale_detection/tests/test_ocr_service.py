"""
Unit tests for OCR service
"""

import pytest
import cv2
import numpy as np
from weighing_scale_detection.ocr.ocr_service import WeightOCRService


@pytest.fixture
def ocr_service():
    """Create OCR service instance (reused across tests)"""
    return WeightOCRService(languages=['en'], gpu=False)


def test_ocr_initialization(ocr_service):
    """Test that OCR service initializes correctly"""
    assert ocr_service.reader is not None


def test_number_extraction(ocr_service):
    """Test number and unit extraction"""
    test_cases = [
        ("88.25 KG", 88.25, 'kg'),
        ("125.5 LB", 125.5, 'lb'),
        ("O.75O KG", 0.750, 'kg'),  # OCR mistake: O→0
        ("8B.2S", 88.25, 'kg'),  # OCR mistakes: B→8, S→5
    ]
    
    for text, expected_value, expected_unit in test_cases:
        value, unit = ocr_service._extract_number_and_unit(text)
        assert value == expected_value, f"Failed for '{text}'"
        assert unit == expected_unit


def test_weight_validation(ocr_service):
    """Test weight validation logic"""
    assert ocr_service._validate_weight(5.5) == True  # Valid
    assert ocr_service._validate_weight(0.05) == False  # Too light
    assert ocr_service._validate_weight(150.0) == False  # Too heavy
    assert ocr_service._validate_weight(-5.0) == False  # Negative


def test_preprocessing(ocr_service):
    """Test image preprocessing"""
    # Create synthetic scale display image
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, "88.25", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    processed = ocr_service._preprocess_for_ocr(img)
    
    # Check that output is binary (only 0 and 255)
    unique_values = np.unique(processed)
    assert len(unique_values) <= 2
    assert processed.shape[0] >= 40  # Should be resized if too small