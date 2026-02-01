"""
Create visual comparison grid of original vs detected images
Perfect for portfolio and presentations
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def create_comparison_grid(num_samples=6):
    """Create side-by-side comparison grid"""
    

    model_path = "models/scale_detection_v1/weights/best.pt"
    test_dir = Path("data/labeled/test/images")
    output_path = "results/comparison_grid.jpg"
    
    print("Loading model...")
    model = YOLO(model_path)
    
    test_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))[:num_samples]
    
    print(f"Creating comparison for {len(test_images)} images...")
    
    comparison_pairs = []
    
    for img_path in test_images:
        original = cv2.imread(str(img_path))
        
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        detected = results[0].plot()
        
        target_height = 400
        h, w = original.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        
        original_resized = cv2.resize(original, (new_w, target_height))
        detected_resized = cv2.resize(detected, (new_w, target_height))
        
        cv2.putText(original_resized, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(detected_resized, "Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        pair = np.hstack([original_resized, detected_resized])
        comparison_pairs.append(pair)
    
    grid = np.vstack(comparison_pairs)
    
    cv2.imwrite(output_path, grid)
    
    print(f"✅ Comparison grid saved: {output_path}")
    print(f"   Size: {grid.shape[1]}x{grid.shape[0]} pixels")

if __name__ == "__main__":
    create_comparison_grid()