"""
scripts/p8/corruption_utils.py
Helper functions for applying synthetic corruptions to images.
"""
import random
from typing import Optional
from PIL import Image, ImageDraw

def apply_occlusion(image: Image.Image, scale_ratio: float = 0.3) -> Image.Image:
    """
    Applies random block occlusion (Cutout).
    scale_ratio: Size of the occlusion block relative to the image size.
    """
    img = image.copy()
    w, h = img.size
    
    # Calculate block size
    block_w = min(w, int(w * scale_ratio))
    block_h = min(h, int(h * scale_ratio))
    
    # Random position
    x = random.randint(0, w - block_w)
    y = random.randint(0, h - block_h)
    
    draw = ImageDraw.Draw(img)
    # Draw black rectangle
    draw.rectangle([x, y, x + block_w, y + block_h], fill="black")
    
    return img

def apply_low_resolution(image: Image.Image, scale_factor: float = 0.1) -> Image.Image:
    """
    Simulates sensor degradation by downsampling and then upsampling.
    scale_factor: The ratio to downsample to (e.g., 0.1 means 10% of original size).
    """
    w, h = image.size
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    
    # Downsample
    img_small = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    
    # Upsample back to original size (nearest neighbor to keep pixelation effect, or bilinear for blur)
    # Using NEAREST makes the pixelation very obvious, which is good for this test.
    img_restored = img_small.resize((w, h), resample=Image.Resampling.NEAREST)
    
    return img_restored

def apply_grayscale(image: Image.Image) -> Image.Image:
    """
    Simulates IR camera or color loss.
    """
    # Convert to grayscale (L) then back to RGB so it stays compatible with models expecting 3 channels
    return image.convert("L").convert("RGB")

def apply_corruption(image_path: str, corruption_type: str, severity: int = 1) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    if corruption_type == "occlusion":
        # Severity 1: 20%, Severity 2: 30%, Severity 3: 40%
        ratio = 0.1 + (severity * 0.1)
        return apply_occlusion(img, scale_ratio=ratio)
    
    elif corruption_type == "resolution":
        # Severity 1: 20%, Severity 2: 10%, Severity 3: 5%
        factors = {1: 0.2, 2: 0.1, 3: 0.05}
        factor = factors.get(severity, 0.1)
        return apply_low_resolution(img, scale_factor=factor)
    
    elif corruption_type == "grayscale":
        return apply_grayscale(img)
    
    else:
        return img
