"""
Comprehensive model evaluation script
Calculates detailed metrics on test set and generates evaluation report
"""

from ultralytics import YOLO
import json
from pathlib import Path
from datetime import datetime

def evaluate_model():
    """
    Run comprehensive evaluation on test set
    """
    
    model_path = "models/scale_detection_v1/weights/best.pt"
    data_yaml = "data/labeled/data.yaml"
    results_dir = Path("results/metrics")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print("="*70)
    
    model = YOLO(model_path)
    
    print("\n🔍 Running evaluation on test set...")
    metrics = model.val(
        data=data_yaml,
        split='test',      
        conf=0.25,         
        iou=0.50,          
        plots=True,        
        save_json=True,    
        project=str(results_dir),
        name='evaluation'
    )
    
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': str(model_path),
        'dataset': str(data_yaml),
        'metrics': {
            'mAP50': float(metrics.box.map50),         # mAP at IoU=0.50
            'mAP50-95': float(metrics.box.map),        # mAP at IoU=0.50:0.95
            'precision': float(metrics.box.mp),        # Precision
            'recall': float(metrics.box.mr),           # Recall
            'f1_score': float(2 * (metrics.box.mp * metrics.box.mr) / 
                            (metrics.box.mp + metrics.box.mr + 1e-6))  # F1 Score
        },
        'performance': {
            'inference_speed_ms': float(metrics.speed['inference']),
            'preprocess_speed_ms': float(metrics.speed['preprocess']),
            'postprocess_speed_ms': float(metrics.speed['postprocess'])
        }
    }
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print("\nDetection Metrics:")
    print(f"  mAP@0.50      : {results['metrics']['mAP50']:.4f} ({results['metrics']['mAP50']*100:.2f}%)")
    print(f"  mAP@0.50:0.95 : {results['metrics']['mAP50-95']:.4f} ({results['metrics']['mAP50-95']*100:.2f}%)")
    print(f"  Precision     : {results['metrics']['precision']:.4f} ({results['metrics']['precision']*100:.2f}%)")
    print(f"  Recall        : {results['metrics']['recall']:.4f} ({results['metrics']['recall']*100:.2f}%)")
    print(f"  F1 Score      : {results['metrics']['f1_score']:.4f}")
    
    print("\n Speed Metrics:")
    print(f"  Inference     : {results['performance']['inference_speed_ms']:.2f} ms/image")
    print(f"  Preprocess    : {results['performance']['preprocess_speed_ms']:.2f} ms/image")
    print(f"  Postprocess   : {results['performance']['postprocess_speed_ms']:.2f} ms/image")
    total_time = sum(results['performance'].values())
    print(f"  Total         : {total_time:.2f} ms/image ({1000/total_time:.1f} FPS)")
    
    print("\n Interpretation:")
    if results['metrics']['mAP50'] >= 0.90:
        print("   Excellent detection accuracy!")
    elif results['metrics']['mAP50'] >= 0.75:
        print("  ✓ Good detection accuracy")
    else:
        print("   May need more training or data")
    
    if results['metrics']['recall'] >= 0.90:
        print("   Model finds almost all scales!")
    
    if results['metrics']['precision'] >= 0.90:
        print("  Model makes very few false detections!")
    
    print("="*70)

    json_path = results_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {json_path}")
    print(f" Plots saved to: {results_dir / 'evaluation'}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    evaluate_model()