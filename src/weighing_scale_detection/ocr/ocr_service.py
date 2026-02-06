"""
OCR Service for extracting weight readings from scale displays.

This module handles:
1. Preprocessing images for optimal OCR accuracy
2. Extracting numerical readings from digital displays
3. Validating and formatting weight values
"""

import easyocr
import cv2
import numpy as np
import re
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightOCRService:
    """
    Service for extracting weight values from weighing scale displays.
    
    Handles the complete pipeline:
    - Image preprocessing (crop, denoise, threshold)
    - OCR text extraction
    - Post-processing (validation, formatting)
    """
    
    def __init__(
        self,
        languages: list = ['en'],
        gpu: bool = False,
        model_storage_directory: str = None
    ):
        """
        Initialize OCR service.
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'] for English + Hindi)
            gpu: Use GPU for faster processing (requires CUDA)
            model_storage_directory: Where to cache OCR models (default: ~/.EasyOCR)
        
        Note: First run downloads ~100MB model files
        """
        logger.info("Initializing EasyOCR Reader...")
        
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                model_storage_directory=model_storage_directory,
                verbose=False  # Suppress download progress for cleaner logs
            )
            logger.info("✅ OCR Reader initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR: {e}")
            raise
    
    def extract_weight(
        self,
        image: np.ndarray,
        bbox: list,
        debug: bool = False
    ) -> Dict:
        """
        Extract weight reading from a detected scale display.
        
        Args:
            image: Full image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2] of the scale display
            debug: If True, saves intermediate processing steps
            
        Returns:
            Dictionary containing:
            - 'raw_text': Original OCR output
            - 'weight_value': Extracted numerical value (float or None)
            - 'unit': Detected unit (kg, lb, oz, etc.)
            - 'confidence': OCR confidence score (0-1)
            - 'is_valid': Boolean indicating if reading is valid
        """
        # STEP 1: Crop the region of interest
        crop = self._crop_scale_region(image, bbox)
        
        if debug:
            cv2.imwrite("debug_1_crop.jpg", crop)
        
        # STEP 2: Preprocess for OCR
        processed = self._preprocess_for_ocr(crop)
        
        if debug:
            cv2.imwrite("debug_2_processed.jpg", processed)
        
        # STEP 3: Run OCR with character whitelist
        # Constrain to only digits, decimal point, spaces, and common weight units
        # This significantly improves accuracy by preventing digit/letter confusion
        allowlist = '0123456789. KGLBOZ'
        
        ocr_results = self.reader.readtext(
            processed,
            allowlist=allowlist,
            paragraph=False  # Read each text block separately
        )
        
        if debug:
            logger.info(f"Raw OCR Results: {ocr_results}")
        
        # STEP 4: Parse and validate results
        result = self._parse_ocr_output(ocr_results)
        
        return result
    
    def _crop_scale_region(
        self,
        image: np.ndarray,
        bbox: list,
        padding: int = 5
    ) -> np.ndarray:
        """
        Crop the scale display from the full image.
        
        Args:
            image: Full image
            bbox: [x1, y1, x2, y2]
            padding: Extra pixels around bbox (helps with edge digits)
        
        Returns:
            Cropped image region
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding (but stay within image bounds)
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = image[y1:y2, x1:x2]
        
        return crop
    
    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess cropped image for optimal OCR accuracy.
        
        Digital displays need specific preprocessing:
        1. Convert to grayscale (reduce color noise)
        2. Apply CLAHE for contrast enhancement
        3. Denoise (remove camera grain)
        4. Threshold (make digits pure black/white)
        5. Morphological operations (clean up artifacts)
        6. Upscale for better recognition
        
        Args:
            crop: Cropped scale display image
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # STEP 1: Convert to grayscale
        # Why? OCR works better on single-channel images
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # STEP 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Why? Enhances local contrast, making digits more distinct
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # STEP 3: Denoise
        # Why? Removes camera sensor noise that confuses OCR
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # STEP 4: Adaptive Thresholding
        # Why? Digital displays have varying brightness across the screen
        # Otsu's method automatically finds the best threshold value
        _, binary = cv2.threshold(
            denoised,
            0,  # Threshold value (0 = auto-calculate)
            255,  # Max value (white)
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # STEP 5: Invert if digits are darker than background
        # Why? EasyOCR expects dark text on light background
        # Check if more pixels are white than black
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # STEP 6: Morphological operations (helps with broken digits)
        # Why? Connects broken segments in seven-segment displays
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # STEP 7: Resize if too small (OCR works better on larger images)
        # Why? EasyOCR has minimum recommended size of 20px height
        # Increased from 40px to 60px for better digit recognition
        h, w = cleaned.shape
        if h < 60:
            scale_factor = 60 / h
            new_w = int(w * scale_factor)
            cleaned = cv2.resize(cleaned, (new_w, 60), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def _parse_ocr_output(self, ocr_results: list) -> Dict:
        """
        Parse and validate OCR results to extract weight value.
        
        OCR often returns messy text like:
        - "88.25 KG" ✓
        - "8B.2S" (mistakes 8 for B, 5 for S) ❌
        - "SCALE 88.25" (extra words) ⚠️
        
        This function cleans and validates the output.
        
        Args:
            ocr_results: List of (bbox, text, confidence) tuples from EasyOCR
            
        Returns:
            Parsed result dictionary
        """
        result = {
            'raw_text': '',
            'weight_value': None,
            'unit': 'kg',  # Default assumption
            'confidence': 0.0,
            'is_valid': False
        }
        
        if not ocr_results:
            return result
        
        # Combine all detected text (sometimes split across multiple boxes)
        full_text = ' '.join([text for (_, text, _) in ocr_results])
        avg_confidence = np.mean([conf for (_, _, conf) in ocr_results])
        
        result['raw_text'] = full_text
        result['confidence'] = float(avg_confidence)
        
        # Extract numerical value using regex
        weight_value, unit = self._extract_number_and_unit(full_text)
        
        if weight_value is not None:
            result['weight_value'] = weight_value
            result['unit'] = unit
            result['is_valid'] = self._validate_weight(weight_value)
        
        return result
    
    def _extract_number_and_unit(self, text: str) -> Tuple[Optional[float], str]:
        """
        Extract numerical value and unit from OCR text.
        
        Handles common OCR errors:
        - 'O' → '0'
        - 'B' → '8'
        - 'S' → '5'
        - 'I' → '1'
        
        Args:
            text: Raw OCR text
            
        Returns:
            (weight_value, unit) or (None, 'kg')
        """
        # Clean the text
        text = text.upper().strip()
        
        # Extract unit FIRST (before character corrections)
        # This prevents 'LB' from being corrupted to 'L8' by the 'B'→'8' correction
        unit = 'kg'  # Default
        for u in ['KG', 'LB', 'OZ', 'G']:
            if u in text:
                unit = u.lower()
                break
        
        # Common OCR character corrections for digits
        corrections = {
            'O': '0',
            'o': '0',
            'B': '8',
            'S': '5',
            'I': '1',
            'l': '1',
            'Z': '2',
            'G': '6'
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # Extract numerical value
        # Pattern: optional digits, decimal point, digits
        # Examples: "88.25", "125.5", "0.750"
        pattern = r'(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        
        if matches:
            try:
                # Take the first (usually largest) number found
                weight_value = float(matches[0])
                return weight_value, unit
            except ValueError:
                return None, unit
        
        return None, unit
    
    def _validate_weight(self, weight: float) -> bool:
        """
        Validate if extracted weight is reasonable.
        
        Typical laundry bag weights: 0.5 kg - 50 kg
        
        Args:
            weight: Extracted weight value
            
        Returns:
            True if valid, False otherwise
        """
        # Sanity checks
        if weight <= 0:
            return False
        
        if weight > 100:  # Unreasonably heavy for laundry
            logger.warning(f"Suspicious weight detected: {weight} kg")
            return False
        
        if weight < 0.1:  # Unreasonably light
            logger.warning(f"Suspicious weight detected: {weight} kg")
            return False
        
        return True
    
    def extract_weight_batch(
        self,
        images_and_bboxes: list,
        show_progress: bool = True
    ) -> list:
        """
        Process multiple images in batch.
        
        Args:
            images_and_bboxes: List of (image, bbox) tuples
            show_progress: Print progress
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, (image, bbox) in enumerate(images_and_bboxes):
            if show_progress:
                print(f"Processing {i+1}/{len(images_and_bboxes)}...", end='\r')
            
            result = self.extract_weight(image, bbox)
            results.append(result)
        
        if show_progress:
            print(f"Processed {len(images_and_bboxes)} images")
        
        return results