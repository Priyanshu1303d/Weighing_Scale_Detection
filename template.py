import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(message)s")

project_name = "weighing_scale_detection"

list_of_files = [
    ".env",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "setup.py",
    
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/detector/__init__.py",
    f"src/{project_name}/detector/scale_detector.py",
    f"src/{project_name}/detector/primary_selector.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/image_utils.py",
    f"src/{project_name}/utils/visualization.py",
    f"src/{project_name}/detector/tests/__init__.py",
    f"src/{project_name}/tests/__init__.py",
    f"src/{project_name}/tests/test_primary_selector.py",
    
    "scripts/train.py",
    "scripts/inference.py",
    "scripts/evaluate.py",
    "scripts/quick_test.py",
    "scripts/create_comparison.py"
    
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_model_training.ipynb",
    "notebooks/03_results_analysis.ipynb",
    
    "data/raw/.gitkeep",
    "data/labeled/.gitkeep",
    
    "models/.gitkeep",
    
    "results/predictions/.gitkeep",
    "results/metrics/.gitkeep",
    "results/visualizations/.gitkeep",
    
    "app/streamlit_app.py",
    "app/pages/1_Model_Metrics.py",
    "app/pages/2_Predictions.py",
    "app/examples/.gitkeep",
    
    "docs/TRAINING_REPORT.md",
    "docs/API_DOCS.md",
    "README.md"
]

def create_project_structure():
    """Create complete project structure"""
    
    created_folders = set()
    created_files = []
    
    for file_path_str in list_of_files:
        file_path = Path(file_path_str)
        folder = file_path.parent
        
        if folder != Path(".") and str(folder) not in created_folders:
            folder.mkdir(parents=True, exist_ok=True)
            created_folders.add(str(folder))
            logging.info(f"📁 Created folder: {folder}")
        
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as f:
                if file_path.name == ".gitkeep":
                    f.write("# Keeps directory in git\n")
            created_files.append(str(file_path))
            logging.info(f"✅ Created file: {file_path}")
    
    print("\n" + "======================================================================")
    print(f"📁 Folders created: {len(created_folders)}")
    print(f"✅ Files created: {len(created_files)}")
    print("======================================================================")

if __name__ == '__main__':
    create_project_structure()