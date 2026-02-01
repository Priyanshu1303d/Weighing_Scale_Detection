"""
Visual Results Page - Shows comparison grid and sample detections
"""

import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Visual Results", page_icon="🖼️", layout="wide")

st.title("🖼️ Visual Detection Results")

# Show comparison grid
st.markdown("## 📊 Before & After Comparison")

comparison_path = Path("results/comparison_grid.jpg")

if comparison_path.exists():
    img = Image.open(comparison_path)
    st.image(img, caption="Original vs Detected - Comparison Grid", use_container_width=True)
else:
    st.warning("Comparison grid not found. Run: `python scripts/create_comparison.py`")

# Show individual results
st.markdown("## 🎯 Sample Detections")

results_dir = Path("results/quick_test")

if results_dir.exists():
    result_images = sorted(list(results_dir.glob("*.jpg")))
    
    if result_images:
        cols = st.columns(3)
        
        for i, img_path in enumerate(result_images[:6]):
            with cols[i % 3]:
                img = Image.open(img_path)
                st.image(img, caption=img_path.stem, use_container_width=True)
    else:
        st.info("No sample images found")
else:
    st.warning("Results directory not found. Run: `python scripts/quick_test.py`")
