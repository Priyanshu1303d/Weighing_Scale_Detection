"""
OCR Debug Script - Understand what's happening step by step
"""

import cv2
import numpy as np
from pathlib import Path
from weighing_scale_detection.detector.scale_detector import ScaleDetector

def debug_single_image(image_path: str):
    """
    Debug OCR on a single image with FULL visibility
    """
    print("="*70)
    print(f"DEBUGGING OCR: {image_path}")
    print("="*70)
    
    # STEP 1: Load image
    img = cv2.imread(image_path)
    print(f"\n✅ Image loaded: {img.shape}")
    
    # STEP 2: Detect scale
    detector = ScaleDetector(
        "models/scale_detection_v1/weights/best.pt",
        enable_ocr=False  # Disable for now
    )
    
    result = detector.detect_with_primary(img)
    primary_scale = result['primary_scale']
    
    if not primary_scale:
        print("❌ No scale detected!")
        return
    
    print(f"\n✅ Primary scale detected:")
    print(f"   Bbox: {primary_scale['bbox']}")
    print(f"   Confidence: {primary_scale['confidence']:.2%}")
    print(f"   Area: {primary_scale['area']} pixels")
    
    # STEP 3: Crop the scale region (save it)
    x1, y1, x2, y2 = map(int, primary_scale['bbox'])
    crop = img[y1:y2, x1:x2]
    
    cv2.imwrite("debug_outputs/1_original_crop.jpg", crop)
    print(f"\n✅ Saved cropped region: debug_outputs/1_original_crop.jpg")
    print(f"   Crop size: {crop.shape}")
    
    # STEP 4: Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_outputs/2_grayscale.jpg", gray)
    print(f"\n✅ Converted to grayscale")
    
    # STEP 5: Apply CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite("debug_outputs/3_enhanced.jpg", enhanced)
    print(f"\n✅ Applied CLAHE (contrast enhancement)")
    
    # STEP 6: Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10)
    cv2.imwrite("debug_outputs/4_denoised.jpg", denoised)
    print(f"\n✅ Denoised image")
    
    # STEP 7: Binary thresholding (Otsu)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("debug_outputs/5_binary_otsu.jpg", binary)
    print(f"\n✅ Applied Otsu's threshold")
    
    # STEP 8: Check if we need to invert
    mean_val = np.mean(binary)
    print(f"\n📊 Binary image mean: {mean_val:.2f}")
    
    if mean_val > 127:
        print("   → Image is mostly white, inverting...")
        binary = cv2.bitwise_not(binary)
        cv2.imwrite("debug_outputs/6_inverted.jpg", binary)
    else:
        print("   → Image is mostly black, keeping as is")
    
    # STEP 9: Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("debug_outputs/7_morphology.jpg", cleaned)
    print(f"\n✅ Applied morphological operations")
    
    # STEP 10: Resize if needed
    h, w = cleaned.shape
    print(f"\n📐 Current size: {w}x{h}")
    
    if h < 60:
        scale_factor = 60 / h
        new_w = int(w * scale_factor)
        resized = cv2.resize(cleaned, (new_w, 60), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("debug_outputs/8_final_resized.jpg", resized)
        print(f"   → Resized to: {new_w}x60")
        final_img = resized
    else:
        final_img = cleaned
        print(f"   → No resize needed")
    
    # STEP 11: Now try OCR on the final processed image
    print("\n" + "="*70)
    print("TESTING OCR ON PROCESSED IMAGE")
    print("="*70)
    
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    # Try WITHOUT allowlist first
    print("\n🔍 Attempt 1: No character restrictions")
    results_no_filter = reader.readtext(final_img)
    print(f"   Results: {results_no_filter}")
    
    # Try WITH allowlist
    print("\n🔍 Attempt 2: Only digits and weight units")
    allowlist = '0123456789. KGLBOZ'
    results_filtered = reader.readtext(final_img, allowlist=allowlist)
    print(f"   Results: {results_filtered}")
    
    # Try with paragraph mode
    print("\n🔍 Attempt 3: Paragraph mode")
    results_paragraph = reader.readtext(final_img, paragraph=True)
    print(f"   Results: {results_paragraph}")
    
    print("\n" + "="*70)
    print("CAN YOU READ THE DIGITS IN THE FINAL IMAGE?")
    print("Open: debug_outputs/8_final_resized.jpg")
    print("If YOU can't read it, OCR won't either.")
    print("="*70)


if __name__ == "__main__":
    # Create output directory
    Path("debug_outputs").mkdir(exist_ok=True)
    
    # Test on one of your test images
    test_image = "data/labeled/test/images/12_jpeg.rf.3066208cf2f15ec6ab94734e75779fcf.jpg"
    
    debug_single_image(test_image)