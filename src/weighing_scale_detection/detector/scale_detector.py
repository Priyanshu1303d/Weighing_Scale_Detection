"""
ScaleDetector - Main detection class for weighing scale displays

This class wraps YOLOv8 functionality and provides a clean interface
for detecting scale displays in images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from PIL import Image
from weighing_scale_detection.detector.primary_selector import PrimaryScaleSelector
from weighing_scale_detection.ocr.ocr_service import WeightOCRService

class ScaleDetector:
    """
    Detects weighing scale displays in images using fine-tuned YOLOv8.
    
    The detector can process single images or batches and returns
    structured detection results with bounding boxes and confidence scores.
    
    Example:
        >>> detector = ScaleDetector('models/best.pt')
        >>> detections = detector.detect('image.jpg')
        >>> print(f"Found {len(detections)} scales")
    """
    
    def __init__(
        self, 
        model_path: str, 
        conf_threshold: float = 0.25,
        device: str = 'cpu',
        enable_ocr: bool = True
    ):
        """
        Initialize the scale detector.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            conf_threshold: Minimum confidence score (0-1) for detections
            device: Device to run inference on ('cpu', 'cuda', or '0' for GPU)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.model = YOLO(str(model_path))
            self.conf_threshold = conf_threshold
            self.device = device

            self.primary_selector = PrimaryScaleSelector()
            self.enable_ocr = enable_ocr
            if enable_ocr:
                self.ocr_service = WeightOCRService(gpu=(device != 'cpu'))

            print(f"✅ Loaded model: {model_path.name}")
            print(f"🎯 Confidence threshold: {conf_threshold}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect(
        self, 
        image: Union[str, np.ndarray],
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect scale displays in an image.
        
        Args:
            image: Either path to image file (str) or numpy array (BGR format)
            conf_threshold: Override default confidence threshold for this detection
            
        Returns:
            List of detection dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] - Bounding box coordinates
                - 'confidence': float - Detection confidence score (0-1)
                - 'class_name': str - Detected class name
                - 'class_id': int - Class ID
                - 'area': int - Bounding box area in pixels
        
        Example:
            >>> detections = detector.detect('scale.jpg')
            >>> for det in detections:
            ...     print(f"Scale at {det['bbox']} with {det['confidence']:.2%} confidence")
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            area = int((x2 - x1) * (y2 - y1))
            
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(box.conf[0].cpu().numpy()),
                'class_name': result.names[int(box.cls[0])],
                'class_id': int(box.cls[0]),
                'area': area
            }
            detections.append(detection)
        
        return detections

    def detect_with_primary(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Enhanced detection that identifies the primary (most relevant) scale.
        
        Args:
            image: Either path to image file (str), numpy array (BGR format), or PIL Image
            
        Returns:
            Dictionary containing:
                - 'all_detections': List of all scale detections
                - 'primary_scale': The primary scale detection (or None)
                - 'num_scales': Total number of scales detected
        
        Example:
            >>> result = detector.detect_with_primary('scales.jpg')
            >>> print(f"Found {result['num_scales']} scales")
            >>> if result['primary_scale']:
            ...     print(f"Primary scale score: {result['primary_scale']['primary_score']:.2%}")
        """
        detections = self.detect(image)

        # Convert image to numpy array if needed
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array (RGB format)
            img = np.array(image)
            # Handle RGBA images (check shape after conversion to numpy)
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # Convert RGB to BGR for OpenCV
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image

        # Use PrimaryScaleSelector to identify the most relevant scale
        primary_scale = self.primary_selector.resolve_primary_scale(
            detections, 
            img.shape,
            image=img
        )

        return {
            'all_detections': detections,
            'primary_scale': primary_scale,
            'num_scales': len(detections)
        }

    def detect_with_weight(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Detect scale and extract weight value in one call.
        
        Args:
            image: Input image (file path, NumPy array, or PIL Image)
        
        Returns:
            Detection result dict with OCR results included in primary_scale
        """
        # Get primary scale
        result = self.detect_with_primary(image)
        primary_scale = result['primary_scale']
        
        # Extract weight if OCR enabled
        if self.enable_ocr and primary_scale:
            # Convert image to NumPy array (BGR) for OCR
            if isinstance(image, str):
                img = cv2.imread(image)
            elif isinstance(image, Image.Image):
                # Convert PIL Image to NumPy array
                img_array = np.array(image)
                # Handle RGBA images
                if img_array.shape[-1] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # Already a NumPy array
                img = image
            
            ocr_result = self.ocr_service.extract_weight(
                img,
                primary_scale['bbox']
            )
            
            primary_scale['ocr_result'] = ocr_result
        
        return result
    
    def detect_and_visualize(
        self,
        image: Union[str, np.ndarray],
        show_confidence: bool = True,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect scales and draw bounding boxes on the image.
        
        Args:
            image: Path to image or numpy array
            show_confidence: Whether to display confidence scores on boxes
            box_color: BGR color for bounding boxes (default: green)
            thickness: Line thickness for boxes
            
        Returns:
            Tuple of:
                - Annotated image (numpy array in BGR format)
                - List of detections
        
        Example:
            >>> annotated_img, detections = detector.detect_and_visualize('test.jpg')
            >>> cv2.imwrite('result.jpg', annotated_img)
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        detections = self.detect(img)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            label = det['class_name']
            
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
            
            if show_confidence:
                text = f"{label}: {conf:.2%}"
            else:
                text = label
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                box_color,
                cv2.FILLED
            )
            
            cv2.putText(
                img,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return img, detections
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        show_progress: bool = True
    ) -> List[List[Dict]]:
        """
        Detect scales in multiple images efficiently.
        
        Args:
            images: List of image paths or numpy arrays
            show_progress: Whether to print progress
            
        Returns:
            List of detection lists (one list per image)
        
        Example:
            >>> image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> all_detections = detector.detect_batch(image_paths)
            >>> print(f"Processed {len(all_detections)} images")
        """
        all_detections = []
        
        for i, image in enumerate(images):
            if show_progress:
                print(f"Processing {i+1}/{len(images)}...", end='\r')
            
            detections = self.detect(image)
            all_detections.append(detections)
        
        if show_progress:
            print(f"Processed {len(images)} images")
        
        return all_detections
    
    def get_largest_detection(
        self,
        image: Union[str, np.ndarray]
    ) -> Optional[Dict]:
        """
        Get the largest (by area) detected scale display.
        
        Useful when image contains multiple scales but you only
        want the most prominent one.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Detection dict with largest area, or None if no detections
        
        Example:
            >>> largest = detector.get_largest_detection('scales.jpg')
            >>> if largest:
            ...     print(f"Largest scale: {largest['area']} pixels")
        """
        detections = self.detect(image)
        
        if not detections:
            return None
        
        return max(detections, key=lambda d: d['area'])