"""
Training script for weighing scale detection model
"""
import os
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

def train_model():
    """
    Train YOLOv8 model on weighing scale dataset
    """
    
    BASE_DIR = Path(__file__).parent.parent
    data_yaml = BASE_DIR / "data" / "labeled" / "data.yaml"
    model_save_dir = BASE_DIR / "models"
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 16
    
    print("="*70)
    print("WEIGHING SCALE DETECTION - TRAINING")
    print("="*70)
    print(f" Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Dataset: {data_yaml}")
    print(f" Base Model: YOLOv8n (Nano - fastest)")
    print(f" Epochs: {EPOCHS}")
    print(f" Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f" Batch Size: {BATCH_SIZE}")
    print(f" Output: {model_save_dir}")
    print("="*70)
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=str(data_yaml),
        
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        
        name='scale_detection_v1',
        project=str(model_save_dir),
        exist_ok=True,
        
        patience=15,
        save=True,
        save_period=10,
        
        val=True,
        plots=True,
        
        device=0,
        workers=8,
        
        # optimizer='AdamW',
        # lr0=0.001,
        # lrf=0.01,
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    best_model_path = model_save_dir / "scale_detection_v1" / "weights" / "best.pt"
    print(f"Best model: {best_model_path}")
    print(f"Training results: {model_save_dir / 'scale_detection_v1'}")
    
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n Final Metrics:")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    
    print("="*70)
    print(f" End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()