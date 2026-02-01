from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.1.0"

PROJECT_NAME = "weighing_scale_detection"
REPO_NAME = "weighing-scale-detection"
AUTHOR_USER_NAME = "Priyanshu1303d"
AUTHOR_EMAIL = "priyanshu1303@gmail.com"

setup(
    name=PROJECT_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Fine-tuned YOLOv8 model for detecting weighing scale displays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
)