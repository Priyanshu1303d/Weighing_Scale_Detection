# 📊 Model Training Report

**Project:** Weighing Scale Display Detection  
**Date:** January 31, 2026 | **Author:** Priyanshu Kumar Singh

---

## Executive Summary

Fine-tuned **YOLOv8n** for weighing scale display detection. Achieved **99.5% mAP@0.50**, **100% recall**, and **99.17% precision** on test set.

---

## 1. Model Architecture

**Model:** YOLOv8n (Nano)

| Specification | Value |
|--------------|-------|
| Backbone | CSPDarknet53 + C2f |
| Neck | PANet |
| Parameters | 3.2M |
| Size | 6.5 MB |
| Input | 640×640 px |

**Why YOLOv8n:** Real-time speed (~64 FPS), lightweight, suitable for edge deployment.

---

## 2. Dataset

**Source:** NoScrubs assignment (~150 images) + Roboflow augmentations  
**Format:** YOLO TXT | **Class:** `scale_display` (single)

| Split | Images | Use |
|-------|--------|-----|
| Train | 390 | Training with augmentations |
| Val | 49 | Early stopping |
| Test | 49 | Final evaluation |

**Augmentations:** Mosaic, MixUp, HSV shifts, flip, rotation (±10°), scale (0.5×-1.5×)

---

## 3. Training Configuration

```yaml
Model: yolov8n.pt (COCO pre-trained)
Epochs: 100 (early stopped at 37)
Image Size: 640×640
Batch: 16
Device: GPU (CUDA)
Optimizer: SGD
LR: 0.01 → 0.0001 (cosine decay)
Patience: 15 epochs
```

**Compute:** NVIDIA GPU, 8 workers, ~8.5 min training time

---

## 4. Training Results

### Loss Convergence

| Component | Epoch 1 | Epoch 37 | Reduction |
|-----------|---------|----------|-----------|
| Box Loss | 1.071 | 0.704 | -34.3% |
| Class Loss | 3.057 | 0.517 | -83.1% |

### Performance Over Time

| Epoch  | mAP@0.50 | Precision | Recall | Notes |
|--------|----------|-----------|--------|-------|
| 1      | 24.4%    | 0.03%     | 100%   | Initial |
| 8      | 99.5%    | 95.3%     | 100%   | Converged |
| 22     | **99.5%** | 99.1%    | 100%   | **Best** |
| 37     | 99.5%    | 99.3%     | 100%   | Final |

---

## 5. Test Set Evaluation

### Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **mAP@0.50** | **99.50%** | ⭐ Near perfect |
| mAP@0.50:0.95 | 79.78% | ✅ Robust |
| **Precision** | **99.17%** | ⭐ Minimal FP |
| **Recall** | **100%** | ⭐ Perfect |
| F1 Score | 0.9958 | ⭐ Balanced |

### Speed

| Phase     | Time      | FPS |
|-----------|-----------|-----|
| Inference | 10.82 ms  | ~64 |
| Total     | 15.64 ms  |  -  |

**Optimal Threshold:** 0.25 (default)

---

## 6. Strengths & Limitations

### ✅ Strengths
- Perfect recall (100%) - finds all scales
- Near-perfect precision (99.17%) - minimal false positives  
- Real-time speed (64 FPS)
- Lightweight (6.5 MB) - edge deployable

### ⚠️ Limitations
- Single class only
- Small dataset (~150 original images)
- Multi-scale scenarios need post-processing

---

## 7. Deployment

### Export Formats

```bash
# ONNX (cross-platform)
yolo export model=models/scale_detection_v1/weights/best.pt format=onnx

# TensorRT (NVIDIA)
yolo export model=models/scale_detection_v1/weights/best.pt format=engine

# TFLite (mobile)
yolo export model=models/scale_detection_v1/weights/best.pt format=tflite
```

### Options
- Python API (Streamlit, CLI)
- REST API (FastAPI + Docker)
- Edge (Jetson, Raspberry Pi, iOS/Android)

---

## 8. Reproducibility

```bash
pip install ultralytics opencv-python numpy
python scripts/train.py
python scripts/evaluate.py
```

**Output:** `models/scale_detection_v1/weights/best.pt`

---

## Conclusion

YOLOv8n achieves exceptional performance for scale detection with 99.5% mAP, 100% recall, real-time speed (64 FPS), and production-ready deployment options.

**Business Impact:** No missed detections, minimal false alarms, real-time video processing capable.

---

**References:**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Dataset](https://universe.roboflow.com/playfieldvision/weighing-scale-detection/dataset/1)
