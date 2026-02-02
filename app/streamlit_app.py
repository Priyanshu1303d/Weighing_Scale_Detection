"""
Professional Streamlit Web Application for Weighing Scale Detection

Features:
- Drag-and-drop image upload with example images
- Real-time detection visualization
- Adjustable confidence threshold
- Detection statistics and metrics
- Download results (image + JSON)
- Uniform, responsive layout

"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import sys
from datetime import datetime
from weighing_scale_detection.detector.scale_detector import ScaleDetector

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Scale Display Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Uniform container styling */
    .stContainer {
        min-height: 400px;
    }
    
    /* Image container - fixed height */
    .image-container {
        height: 450px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Results container - fixed height */
    .results-container {
        height: 450px;
        overflow-y: auto;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        background-color: #ffffff;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Stats section */
    .stats-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Detection card */
    .detection-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        width: 100%;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea15 0%, #764ba215 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached for performance)"""
    model_path = "models/scale_detection_v1/weights/best.pt"
    
    if not Path(model_path).exists():
        st.error(f"❌ Model not found at {model_path}")
        st.info("💡 Make sure you've trained the model first: `python scripts/train.py`")
        st.stop()
    
    return YOLO(model_path)

def load_example_image(example_num):
    """Load example image"""
    example_path = Path(f"app/examples/example{example_num}.jpg")
    
    if example_path.exists():
        return Image.open(example_path)
    else:
        st.warning(f"Example image {example_num} not found at {example_path}")
        return None

def detect_scales(image, model, conf_threshold):
    """
    Run detection on image
    
    Args:
        image: PIL Image
        model: YOLO model
        conf_threshold: Confidence threshold (0-1)
        
    Returns:
        tuple: (annotated_image_array, detections_list)
    """
    # Convert PIL to OpenCV format (BGR)
    img_array = np.array(image)
    
    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run detection
    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        verbose=False,
        device='cpu'  # Ensure compatibility
    )
    
    result = results[0]
    
    # Get annotated image
    annotated = result.plot(
        line_width=3,
        font_size=16
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Extract detections
    detections = []
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = result.names[cls_id]
        
        detections.append({
            'id': idx + 1,
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': conf,
            'class_name': cls_name,
            'box_area': int((x2 - x1) * (y2 - y1)),
            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
        })
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    return annotated_rgb, detections

def image_to_bytes(image_array):
    """Convert numpy array to bytes for download"""
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=95)
    return buf.getvalue()

# ==================== SESSION STATE INITIALIZATION ====================

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_source' not in st.session_state:
    st.session_state.image_source = None

# ==================== SIDEBAR ====================

with st.sidebar:
    # Logo/Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>⚖️ Scale Detector</h2>
        <p style='color: white; margin: 0; font-size: 0.9rem;'>YOLOv8 Powered</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Detection Settings")
    
    # Confidence threshold slider
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects (but may include false positives)"
    )
    
    # Visual feedback for threshold
    if conf_threshold < 0.3:
        st.info("🔍 Low threshold - More detections, higher false positives")
    elif conf_threshold > 0.6:
        st.info("🎯 High threshold - Fewer detections, higher accuracy")
    else:
        st.success("✅ Balanced threshold - Recommended")
    
    st.markdown("---")
    
    # OCR Settings
    st.markdown("### 🔤 OCR Settings")
    enable_ocr = st.checkbox(
        "Enable Weight Extraction",
        value=True,
        help="Extract weight values from detected scale displays using OCR"
    )
    
    st.markdown("---")
    
    # Model information
    st.markdown("## 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("mAP@0.50", "99.50%", delta="Excellent")
        st.metric("Precision", "99.17%")
    with col2:
        st.metric("Recall", "100%", delta="Perfect")
        st.metric("Speed", "64 FPS")
    
    st.markdown("---")
    
    # Quick facts
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Architecture:** YOLOv8n (Nano)
        
        **Training Details:**
        - Dataset: 390 training images
        - Epochs: 37 (early stopped)
        - Augmentations: Flip, rotation, brightness
        
        **Performance:**
        - Inference: 10.8ms per image
        - F1 Score: 0.9959
        - Perfect recall (finds all scales!)
        
        **Use Case:**
        Detects weighing scale displays in images,
        ideal for automated weight logging in
        laundry services.
        """)
    
    with st.expander("🎯 How to Use"):
        st.markdown("""
        1. **Upload** an image using the file uploader
        2. **Or** click an example button to test
        3. **Adjust** confidence threshold if needed
        4. **View** detection results on the right
        5. **Download** annotated image and data
        """)
    
    st.markdown("---")
    
    # Author info
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: white; border-radius: 8px;'>
        <p style='margin: 0; font-weight: 600;'>Created by</p>
        <p style='margin: 0; color: #667eea;'>Priyanshu Kumar Singh</p>
        <p style='margin: 0; font-size: 0.8rem;'>NoScrubs Internship 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

# Header
st.markdown('<h1 class="main-title">⚖️ Weighing Scale Display Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Scale Detection | Built with YOLOv8 & Streamlit</p>', unsafe_allow_html=True)

# Create two main columns with equal height
col_left, col_right = st.columns(2, gap="large")

# ==================== LEFT COLUMN: IMAGE UPLOAD ====================

with col_left:
    st.markdown("### 📤 Input Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP",
        label_visibility="collapsed"
    )
    
    # Example buttons
    st.markdown("**Or try an example:**")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    
    with ex_col1:
        if st.button("📷 Example 1", use_container_width=True):
            st.session_state.uploaded_image = load_example_image(1)
            st.session_state.image_source = "example1.jpg"
            st.rerun()
    
    with ex_col2:
        if st.button("📷 Example 2", use_container_width=True):
            st.session_state.uploaded_image = load_example_image(2)
            st.session_state.image_source = "example2.jpg"
            st.rerun()
    
    with ex_col3:
        if st.button("📷 Example 3", use_container_width=True):
            st.session_state.uploaded_image = load_example_image(3)
            st.session_state.image_source = "example3.jpg"
            st.rerun()
    
    st.markdown("---")
    
    # Handle uploaded file
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.session_state.image_source = uploaded_file.name
    
    # Display image
    if st.session_state.uploaded_image is not None:
        image = st.session_state.uploaded_image
        
        # Create container with fixed height
        image_container = st.container()
        with image_container:
            st.image(
                image,
                caption=f"📁 {st.session_state.image_source}",
                use_container_width=True
            )
        
        # Image info
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;'>
            <strong>Image Info:</strong><br>
            📐 Size: {image.size[0]} × {image.size[1]} pixels<br>
            📊 Format: {image.format}<br>
            🎨 Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Placeholder when no image
        st.markdown("""
        <div class='image-container'>
            <div style='text-align: center; color: #999;'>
                <h3 style='margin: 0;'>📤</h3>
                <p style='margin: 0.5rem 0 0 0;'>Upload an image or try an example</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== RIGHT COLUMN: DETECTION RESULTS ====================

with col_right:
    st.markdown("### 🎯 Detection Results")
    
    if st.session_state.uploaded_image is not None:
        try:
            # Load model
            with st.spinner("🔄 Loading model..."):
                model = load_model()
            
            # NEW: Use enhanced detection with primary scale identification
            with st.spinner("🔍 Detecting scales and identifying primary..."):
                detector = ScaleDetector(
                    "models/scale_detection_v1/weights/best.pt",
                    conf_threshold=conf_threshold,
                    enable_ocr=enable_ocr
                )
                
                # Use OCR-enabled detection if enabled, otherwise use standard detection
                if enable_ocr:
                    result = detector.detect_with_weight(st.session_state.uploaded_image)
                else:
                    result = detector.detect_with_primary(st.session_state.uploaded_image)
                
                all_detections = result['all_detections']
                primary_scale = result['primary_scale']
                num_scales = result['num_scales']
            
            # Display results
            if num_scales > 0:
                # Visualize with primary highlighted
                img_array = np.array(st.session_state.uploaded_image)
                
                # Handle RGBA images
                if img_array.shape[-1] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                annotated = img_array.copy()
                
                # Draw all detections
                for det in all_detections:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    
                    # Primary scale: RED with thick border
                    # Other scales: GREEN with normal border
                    is_primary = det.get('is_primary', False)
                    color = (255, 0, 0) if is_primary else (0, 255, 0)  # RGB format
                    thickness = 4 if is_primary else 2
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label
                    label = "PRIMARY" if is_primary else f"{det['confidence']:.1%}"
                    label_bg_color = color
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        label_bg_color,
                        cv2.FILLED
                    )
                    
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2
                    )
                
                # Show annotated image
                results_container = st.container()
                with results_container:
                    st.image(
                        annotated,
                        caption="🔴 RED = Primary Scale | 🟢 GREEN = Other Scales",
                        use_container_width=True
                    )
                
                # Show primary scale info
                if primary_scale:
                    st.success(f"✅ PRIMARY SCALE IDENTIFIED (Score: {primary_scale['primary_score']:.2%})")
                    
                    with st.expander("🧠 Why was this chosen?", expanded=True):
                        explanation = detector.primary_selector.explain_decision(primary_scale)
                        st.code(explanation, language=None)
                    
                    st.markdown(f"**Total scales detected:** {num_scales}")
                    
                    # Show detailed breakdown for primary scale
                    st.markdown("#### � Primary Scale Metrics")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Confidence", 
                            f"{primary_scale['confidence']:.2%}",
                            delta="Detection Score"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Primary Score",
                            f"{primary_scale['primary_score']:.2%}",
                            delta="Selection Score"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Area",
                            f"{primary_scale['area']:,} px²",
                            delta="Bbox Size"
                        )
                    
                    # OCR Weight Reading Section
                    if enable_ocr and 'ocr_result' in primary_scale:
                        st.markdown("---")
                        st.markdown("### 📖 Weight Reading")
                        
                        ocr_data = primary_scale['ocr_result']
                        
                        ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
                        
                        with ocr_col1:
                            if ocr_data['is_valid']:
                                st.metric(
                                    "Weight",
                                    f"{ocr_data['weight_value']:.2f} {ocr_data['unit']}",
                                    delta="✓ Valid"
                                )
                            else:
                                st.metric(
                                    "Weight",
                                    "N/A",
                                    delta="✗ Invalid"
                                )
                        
                        with ocr_col2:
                            st.metric(
                                "OCR Confidence",
                                f"{ocr_data['confidence']:.1%}"
                            )
                        
                        with ocr_col3:
                            st.metric(
                                "Raw Text",
                                ocr_data['raw_text'] if ocr_data['raw_text'] else "N/A"
                            )
                    
                    # Show all detections in expandable section
                    if num_scales > 1:
                        with st.expander(f"📋 All {num_scales} Detections", expanded=False):
                            for idx, det in enumerate(all_detections, 1):
                                is_primary = det.get('is_primary', False)
                                prefix = "🔴 PRIMARY - " if is_primary else "🟢 "
                                
                                st.markdown(f"**{prefix}Detection #{idx}**")
                                
                                detail_col1, detail_col2 = st.columns(2)
                                
                                with detail_col1:
                                    st.caption(f"Confidence: {det['confidence']:.2%}")
                                    st.caption(f"Area: {det['area']:,} px²")
                                
                                with detail_col2:
                                    if 'primary_score' in det:
                                        st.caption(f"Primary Score: {det['primary_score']:.2%}")
                                    bbox = det['bbox']
                                    st.caption(f"Position: ({int(bbox[0])}, {int(bbox[1])})")
                                
                                st.markdown("---")
                
                else:
                    st.warning("⚠️ Scales detected but could not determine primary scale")
                
            else:
                # No detections
                st.warning("⚠️ No scale displays detected in this image")
                st.info("💡 **Try these suggestions:**\n- Lower the confidence threshold in the sidebar\n- Ensure the image contains a weighing scale display\n- Try a different image")
                
                # Still show the original image
                st.image(
                    st.session_state.uploaded_image,
                    caption="Original Image (No Detections)",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")
            st.exception(e)
    
    else:
        # Placeholder when no image
        st.markdown("""
        <div class='results-container' style='display: flex; align-items: center; justify-content: center;'>
            <div style='text-align: center; color: #999;'>
                <h3 style='margin: 0;'>🎯</h3>
                <p style='margin: 0.5rem 0;'>Detection results will appear here</p>
                <p style='margin: 0; font-size: 0.9rem;'>Upload an image to get started</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== STATISTICS & DOWNLOAD SECTION ====================

if st.session_state.uploaded_image is not None and 'all_detections' in locals() and len(all_detections) > 0:
    
    st.markdown("---")
    
    st.markdown("### 📊 Detection Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric(
            "Total Detections",
            num_scales,
            delta="Scales Found"
        )
    
    with stat_col2:
        avg_conf = np.mean([d['confidence'] for d in all_detections])
        st.metric(
            "Average Confidence",
            f"{avg_conf:.1%}",
            delta=f"{avg_conf*100:.1f}%"
        )
    
    with stat_col3:
        max_conf = max([d['confidence'] for d in all_detections])
        st.metric(
            "Highest Confidence",
            f"{max_conf:.1%}",
            delta="Best Detection"
        )
    
    with stat_col4:
        total_area = sum([d['area'] for d in all_detections])
        st.metric(
            "Total Coverage",
            f"{total_area:,} px²",
            delta="Area"
        )
    
    st.markdown("---")
    
    st.markdown("### 💾 Download Results")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        img_bytes = image_to_bytes(annotated)
        
        st.download_button(
            label="📥 Download Annotated Image",
            data=img_bytes,
            file_name=f"detected_{st.session_state.image_source}",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with download_col2:
        detection_data = {
            'metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source_image': st.session_state.image_source,
                'model': 'YOLOv8n',
                'confidence_threshold': conf_threshold,
                'total_detections': num_scales,
                'primary_scale_detected': primary_scale is not None
            },
            'all_detections': all_detections,
            'primary_scale': primary_scale
        }
        
        json_data = json.dumps(detection_data, indent=2)
        
        st.download_button(
            label="📄 Download Detection Data (JSON)",
            data=json_data,
            file_name=f"detections_{Path(st.session_state.image_source).stem}.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")
st.markdown("""
<div class="footer">
    <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
        ⚖️ Weighing Scale Detection System
    </p>
    <p style='margin: 0.3rem 0;'>
        Built with <span style='color: #e25555;'>❤️</span> using Streamlit & YOLOv8
    </p>
    <p style='margin: 0.3rem 0;'>
        <strong>NoScrubs Internship Assignment 2025</strong>
    </p>
    <p style='margin: 0.5rem 0;'>
        Created by <a href='https://github.com/Priyanshu1303d' target='_blank' style='color: #667eea; text-decoration: none; font-weight: 600;'>
        Priyanshu Kumar Singh</a>
    </p>
    <p style='margin: 0.5rem 0; font-size: 0.85rem; color: #999;'>
        🔗 <a href='https://github.com/Priyanshu1303d/weighing-scale-detection' target='_blank' style='color: #999;'>View on GitHub</a> | 
        📧 <a href='mailto:priyanshu1303d@gmail.com' style='color: #999;'>Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)