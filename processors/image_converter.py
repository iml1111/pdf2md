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
            dpi: Resolution for image conversion (default: 300)
        """
        self.dpi = dpi
        
    def pdf_to_images(self, pdf_path: str, output_format: str = "png") -> List[bytes]:
        """
        Convert all PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            output_format: Image format (png, jpg, etc.)
            
        Returns:
            List of image bytes for each page
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Converting {doc.page_count} pages to images at {self.dpi} DPI")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Calculate zoom factor for desired DPI
                mat = fitz.Matrix(self.dpi/72.0, self.dpi/72.0)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to bytes
                if output_format.lower() == "png":
                    img_data = pix.tobytes("png")
                elif output_format.lower() in ["jpg", "jpeg"]:
                    img_data = pix.tobytes("jpeg")
                else:
                    # For other formats, use PIL
                    img_data = self._convert_with_pil(pix, output_format)
                
                images.append(img_data)
                pix = None  # Free memory
                
                logger.debug(f"Converted page {page_num + 1}/{doc.page_count}")
            
            doc.close()
            logger.info(f"Successfully converted {len(images)} pages to {output_format} images")
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
        
        return images
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int, output_format: str = "png") -> Optional[bytes]:
        """
        Convert a specific PDF page to image
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            output_format: Image format
            
        Returns:
            Image bytes or None if failed
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_num < 1 or page_num > doc.page_count:
                logger.error(f"Page {page_num} out of range (1-{doc.page_count})")
                return None
            
            page = doc[page_num - 1]  # Convert to 0-indexed
            
            # Calculate zoom factor for desired DPI
            mat = fitz.Matrix(self.dpi/72.0, self.dpi/72.0)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to bytes
            if output_format.lower() == "png":
                img_data = pix.tobytes("png")
            elif output_format.lower() in ["jpg", "jpeg"]:
                img_data = pix.tobytes("jpeg")
            else:
                img_data = self._convert_with_pil(pix, output_format)
            
            pix = None
            doc.close()
            
            return img_data
            
        except Exception as e:
            logger.error(f"Failed to convert page {page_num} to image: {e}")
            return None
    
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
    
    def adjust_dpi(self, new_dpi: int):
        """
        Adjust DPI setting for future conversions
        
        Args:
            new_dpi: New DPI value (typically 150-600)
        """
        if new_dpi < 72:
            logger.warning(f"DPI {new_dpi} is very low, using minimum 72")
            new_dpi = 72
        elif new_dpi > 600:
            logger.warning(f"DPI {new_dpi} is very high, using maximum 600")
            new_dpi = 600
        
        self.dpi = new_dpi
        logger.info(f"DPI adjusted to {self.dpi}")