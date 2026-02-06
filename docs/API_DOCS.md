# 📚 API Documentation

**Weighing Scale Detection** | v1.0.0 | Priyanshu Kumar Singh

---

## Installation

```bash
git clone https://github.com/Priyanshu1303d/weighing-scale-detection.git
cd weighing-scale-detection
python -m venv .venv
.venv\Scripts\activate  # Windows | source .venv/bin/activate (Linux/macOS)
pip install -r requirements.txt
pip install -e .
```

---

## Python API

### ScaleDetector Class

```python
from weighing_scale_detection.detector.scale_detector import ScaleDetector

# Initialize
detector = ScaleDetector(
    model_path="models/scale_detection_v1/weights/best.pt",
    conf_threshold=0.25,
    device='cpu'  # 'cuda' for GPU
)

# Single image detection
result = detector.detect("test.jpg")
print(f"Found {result['num_detections']} scale(s)")

# Primary scale detection (multi-scale images)
from PIL import Image
img = Image.open("image.jpg")
result = detector.detect_with_primary(img)
if result['primary_scale']:
    print(f"Primary: {result['primary_scale']['bbox']}")
    print(f"Score: {result['primary_scale']['primary_score']:.2%}")

# Batch processing
results = detector.detect_batch(
    image_paths=["img1.jpg", "img2.jpg"],
    save_dir="results/output"
)
```

### Detection Format

```python
{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.95,
    "class_name": "scale_display",
    "area": 12500,
    "center": [320.5, 240.5],
    "is_primary": True,        # Optional
    "primary_score": 0.87      # Optional
}
```

---

## Command-Line Interface

### Inference

```bash
# Single image
python scripts/inference.py --input test.jpg

# Batch directory
python scripts/inference.py --input data/images --save-json

# Custom options
python scripts/inference.py --input data/images --conf 0.5 --save-json --save-txt
```

**Options:**
- `--input` - Image/directory (required)
- `--conf` - Threshold 0-1 (default: 0.25)
- `--output` - Output dir (default: results/predictions)
- `--save-json` - Save JSON summary
- `--save-txt` - Save YOLO format files

### Training

```bash
python scripts/train.py
# Output: models/scale_detection_v1/weights/best.pt
```

### Evaluation

```bash
python scripts/evaluate.py
# Output: results/metrics/evaluation_results.json
```

---

## Web Application

```bash
streamlit run app/streamlit_app.py
# Open: http://localhost:8501
```

**Features:**
- Drag-and-drop image upload
- Confidence threshold slider
- Primary scale highlighting (RED=primary, GREEN=other)
- Download annotated image + JSON

---

## Response Formats

### Single Detection

```json
{
  "image_path": "test.jpg",
  "num_detections": 1,
  "detections": [{
    "bbox": [450, 320, 680, 520],
    "confidence": 0.97,
    "area": 46012
  }]
}
```

### Primary Detection

```json
{
  "num_scales": 2,
  "primary_scale": {
    "bbox": [450, 320, 680, 520],
    "confidence": 0.97,
    "primary_score": 0.87,
    "is_primary": true
  },
  "all_detections": [...]
}
```

---

## Examples

### Basic Usage

```python
from weighing_scale_detection.detector.scale_detector import ScaleDetector

detector = ScaleDetector(conf_threshold=0.25)
result = detector.detect("warehouse.jpg")

for det in result['detections']:
    print(f"Scale at {det['bbox']}, conf: {det['confidence']:.1%}")
```

### Batch Processing

```python
from pathlib import Path

detector = ScaleDetector()
images = [str(f) for f in Path("data/images").glob("*.jpg")]

results = detector.detect_batch(images, save_dir="results/batch")
print(f"Total detections: {sum(r['num_detections'] for r in results)}")
```

### Custom Visualization

```python
import cv2

detector = ScaleDetector()
result = detector.detect("image.jpg")
img = cv2.imread("image.jpg")

for det in result['detections']:
    x1, y1, x2, y2 = map(int, det['bbox'])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
cv2.imwrite("output.jpg", img)
```

---

## Performance

| Device | Speed | FPS |
|--------|-------|-----|
| GPU | 10.8 ms | 64 |
| CPU (8-core) | 45-60 ms | 16-22 |

**Optimization:**
```python
# Use GPU
detector = ScaleDetector(device='cuda')

# Higher threshold (faster NMS)
detector = ScaleDetector(conf_threshold=0.4)
```

---


**GitHub:** [Priyanshu1303d/weighing-scale-detection](https://github.com/Priyanshu1303d/weighing-scale-detection)  

