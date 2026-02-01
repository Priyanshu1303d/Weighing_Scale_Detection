"""
Model Metrics Dashboard - Shows training and evaluation metrics
"""

import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")

st.title("📊 Model Performance Metrics")

metrics_file = Path("results/metrics/evaluation_results.json")

if metrics_file.exists():
    with open(metrics_file) as f:
        data = json.load(f)
    
    st.markdown("## 🎯 Detection Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "mAP@0.50",
            f"{data['metrics']['mAP50']:.4f}",
            f"{data['metrics']['mAP50']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{data['metrics']['precision']:.4f}",
            f"{data['metrics']['precision']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{data['metrics']['recall']:.4f}",
            f"{data['metrics']['recall']*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "F1 Score",
            f"{data['metrics']['f1_score']:.4f}"
        )
    
    st.markdown("## ⚡ Speed Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inference", f"{data['performance']['inference_speed_ms']:.2f} ms")
    
    with col2:
        st.metric("Preprocess", f"{data['performance']['preprocess_speed_ms']:.2f} ms")
    
    with col3:
        st.metric("Postprocess", f"{data['performance']['postprocess_speed_ms']:.2f} ms")
    
    with col4:
        total = sum(data['performance'].values())
        fps = 1000 / total
        st.metric("FPS", f"{fps:.1f}")
    
    st.markdown("## 📈 Metrics Visualization")
    
    fig = go.Figure()
    
    metrics_to_plot = [
        ('mAP@0.50', data['metrics']['mAP50']),
        ('Precision', data['metrics']['precision']),
        ('Recall', data['metrics']['recall']),
        ('F1 Score', data['metrics']['f1_score'])
    ]
    
    for i, (name, value) in enumerate(metrics_to_plot):
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            title={'text': name},
            domain={'row': i // 2, 'column': i % 2},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 90], 'color': "lightgreen"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
    
    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 📝 Training Information")
    
    st.json({
        "Model Path": data['model_path'],
        "Dataset": data['dataset'],
        "Evaluation Date": data['evaluation_date']
    })

else:
    st.error("❌ Metrics file not found. Please run evaluation first!")
    st.code("python scripts/evaluate.py")