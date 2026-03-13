"""
PDF to high-quality image converter using PyMuPDF
"""

import fitz  # PyMuPDF
from typing import List, Optional
from pathlib import Path
import io
from PIL import Image, ImageEnhance

# Increase PIL safety limit for high-resolution PDFs (default is ~89M pixels)
Image.MAX_IMAGE_PIXELS = 200000000  # 200M pixels

from utils.logger import logger


class ImageConverter:
    """Convert PDF pages to high-quality images for OCR and LLM processing"""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize image converter
        
        Args:
            dpi: Resolution for image conversion (default: 300)
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

    def convert_page_to_image(self, page_pdf_bytes: bytes) -> bytes:
        """
        Convert single page PDF bytes to PNG image bytes

        Args:
            page_pdf_bytes: PDF bytes containing single page

        Returns:
            PNG image bytes
        """
        try:
            pdf_stream = io.BytesIO(page_pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            if doc.page_count > 0:
                page = doc[0]
                mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pix = None
                doc.close()
                return img_data

            doc.close()
            raise ValueError("No pages in PDF")

        except Exception as e:
            logger.error(f"Failed to convert page to image: {e}")
            raise
