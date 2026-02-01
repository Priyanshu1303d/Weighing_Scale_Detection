"""
Production inference script
Detect weighing scale displays in images with flexible options
"""

import argparse
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import sys

def load_model(model_path):
    """Load YOLO model"""
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
    return model

def get_image_files(input_path):
    """Get list of image files from path"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            return [input_path]
        else:
            print(f"Error: {input_path} is not a valid image file")
            sys.exit(1)
    
    elif input_path.is_dir():
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(ext))
        
        if not image_files:
            print(f"Error: No images found in {input_path}")
            sys.exit(1)
        
        return sorted(image_files)
    
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

def run_inference(
    model_path,
    input_path,
    output_dir,
    conf_threshold=0.25,
    save_txt=False,
    save_json=False,
    show_labels=True,
    show_conf=True
):
    """
    Run inference on images
    
    Args:
        model_path: Path to trained model
        input_path: Path to image or directory
        output_dir: Where to save results
        conf_threshold: Confidence threshold (0-1)
        save_txt: Save results as txt files
        save_json: Save results as JSON
        show_labels: Show class labels on images
        show_conf: Show confidence scores
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_model(model_path)
    
    image_files = get_image_files(input_path)
    
    print("\n" + "="*70)
    print(f"Running inference on {len(image_files)} image(s)")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")
    
    all_results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        result = results[0]
        num_detections = len(result.boxes)
        
        img_results = {
            'filename': img_path.name,
            'image_path': str(img_path),
            'num_detections': num_detections,
            'detections': []
        }
        
        if num_detections > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                img_results['detections'].append(detection)
            
            confidences = [d['confidence'] for d in img_results['detections']]
            print(f"  ✓ Found {num_detections} scale(s)")
            print(f"    Confidence: {', '.join([f'{c:.2%}' for c in confidences])}")
            
            annotated = result.plot(
                labels=show_labels,
                conf=show_conf,
                line_width=2
            )
            output_img_path = output_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_img_path), annotated)
            print(f"  💾 Saved: {output_img_path.name}")
            
            if save_txt:
                txt_path = output_dir / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for det in img_results['detections']:
                        x1, y1, x2, y2 = det['bbox']
                        img_h, img_w = result.orig_shape
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        f.write(f"{det['class_id']} {x_center} {y_center} {width} {height} {det['confidence']}\n")
                print(f" Saved: {txt_path.name}")
        else:
            print(f"  ⚠  No detections")
        
        all_results.append(img_results)
        print()
    
    if save_json:
        json_path = output_dir / "detection_results.json"
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': str(model_path),
            'confidence_threshold': conf_threshold,
            'total_images': len(image_files),
            'total_detections': sum(r['num_detections'] for r in all_results),
            'results': all_results
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"JSON summary saved: {json_path}")
    
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {sum(r['num_detections'] for r in all_results)}")
    print(f"Images with detections: {sum(1 for r in all_results if r['num_detections'] > 0)}")
    print(f"Images without detections: {sum(1 for r in all_results if r['num_detections'] == 0)}")
    
    avg_conf = []
    for r in all_results:
        avg_conf.extend([d['confidence'] for d in r['detections']])
    
    if avg_conf:
        print(f"Average confidence: {sum(avg_conf)/len(avg_conf):.2%}")
        print(f"Min confidence: {min(avg_conf):.2%}")
        print(f"Max confidence: {max(avg_conf):.2%}")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Detect weighing scale displays in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/inference.py --input test.jpg
  
  # Directory of images
  python scripts/inference.py --input data/test/images/ --output results/detections/
  
  # Custom confidence threshold
  python scripts/inference.py --input test.jpg --conf 0.5
  
  # Save results as JSON
  python scripts/inference.py --input images/ --save-json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to image file or directory'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/scale_detection_v1/weights/best.pt',
        help='Path to trained model (default: models/scale_detection_v1/weights/best.pt)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/predictions',
        help='Output directory (default: results/predictions)'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold 0-1 (default: 0.25)'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results in YOLO format .txt files'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results as JSON file'
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide class labels on output images'
    )
    
    parser.add_argument(
        '--no-conf',
        action='store_true',
        help='Hide confidence scores on output images'
    )
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        input_path=args.input,
        output_dir=args.output,
        conf_threshold=args.conf,
        save_txt=args.save_txt,
        save_json=args.save_json,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf
    )

if __name__ == "__main__":
    main()