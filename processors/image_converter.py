"""
PDF to high-quality image converter using PyMuPDF
"""

import fitz  # PyMuPDF
from typing import List, Optional
from pathlib import Path
import io
from PIL import Image

from utils.logger import logger


class ImageConverter:
    """Convert PDF pages to high-quality images for OCR and LLM processing"""
    
    def __init__(self, dpi: int = 600):
        """
        Initialize image converter
        
        Args:
            dpi: Resolution for image conversion (default: 600)
        """
        self.dpi = dpi
        
    def optimize_for_ocr(self, image_bytes: bytes) -> bytes:
        """
        Optimize image for OCR processing
        
        Args:
            image_bytes: Original image bytes
            
        Returns:
            Optimized image bytes
        """
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale for better OCR
            if img.mode != 'L':
                img = img.convert('L')
            
            # Apply slight sharpening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            # Convert back to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Failed to optimize image for OCR: {e}")
            return image_bytes  # Return original on failure
    
    def _convert_with_pil(self, pixmap, output_format: str) -> bytes:
        """
        Convert pixmap to specified format using PIL
        
        Args:
            pixmap: PyMuPDF pixmap object
            output_format: Target image format
            
        Returns:
            Image bytes in specified format
        """
        # Get pixmap as PIL Image
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to target format
        buffer = io.BytesIO()
        img.save(buffer, format=output_format.upper())
        return buffer.getvalue()
    
