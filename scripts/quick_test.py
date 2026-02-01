"""
Quick test script - Verify model works on test images
Run this immediately after training to check model performance
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

def quick_test():
    """Test trained model on a few images"""
    
    model_path = "models/scale_detection_v1/weights/best.pt"
    test_images_dir = "data/labeled/test/images"
    output_dir = "results/quick_test"
    
    print("="*70)
    print("QUICK MODEL TEST")
    print("="*70)
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                  list(Path(test_images_dir).glob("*.png"))
    
    print(f"Found {len(test_images)} test images")
    print("="*70)
    
    for i, img_path in enumerate(test_images[:5], 1):
        print(f"\nTest {i}/{min(5, len(test_images))}: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            save=False,
            verbose=False
        )
        
        result = results[0]
        
        num_detections = len(result.boxes)
        
        if num_detections > 0:
            confidences = result.boxes.conf.cpu().numpy()
            print(f"  ✓ Found {num_detections} scale(s)")
            print(f"  ✓ Confidence scores: {[f'{c:.2%}' for c in confidences]}")
            
            annotated = result.plot()
            output_path = Path(output_dir) / f"test_{i}_{img_path.name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"  ✓ Saved: {output_path}")
        else:
            print(f"  ⚠ No detection (might be a difficult image)")
    
    print("\n" + "="*70)
    print(f" Quick test complete!")
    print(f" Check results in: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    quick_test()