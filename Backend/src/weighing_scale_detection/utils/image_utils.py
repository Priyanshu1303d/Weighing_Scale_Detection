"""
Image utility functions for loading, saving, and processing images.

These utilities handle common image operations needed throughout
the detection pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (BGR format)
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be loaded
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save image to file.
    
    Args:
        image: Numpy array (BGR format)
        output_path: Where to save the image
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(output_path), image)
    
    if not success:
        raise IOError(f"Failed to save image: {output_path}")

def resize_image(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 640,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Either int (for longest side) or (width, height) tuple
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if isinstance(target_size, int):
        if maintain_aspect:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = target_size, target_size
    else:
        new_w, new_h = target_size
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def get_image_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to directory
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff',
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    
    image_files = []
    
    for ext in extensions:
        if recursive:
            image_files.extend(dir_path.rglob(ext))
        else:
            image_files.extend(dir_path.glob(ext))
    
    return [str(f) for f in sorted(image_files)]

def draw_grid(
    images: List[np.ndarray],
    titles: List[str] = None,
    grid_cols: int = None
) -> np.ndarray:
    """
    Arrange multiple images in a grid for comparison.
    
    Args:
        images: List of images (all should have same size)
        titles: Optional titles for each image
        grid_cols: Number of columns (auto-calculated if None)
        
    Returns:
        Grid image
    """
    n = len(images)
    
    if n == 0:
        raise ValueError("No images provided")
    
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))
    h, w = images[0].shape[:2]
    
    grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * h
        x_start = col * w
        
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        grid[y_start:y_start+h, x_start:x_start+w] = img
        
        if titles and idx < len(titles):
            cv2.putText(
                grid,
                titles[idx],
                (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
    
    return grid