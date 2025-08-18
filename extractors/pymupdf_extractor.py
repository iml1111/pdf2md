"""
PyMuPDF (fitz) based PDF text and structure extractor
"""

import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64
from io import BytesIO
from utils.logger import logger, log_extraction_result


class PyMuPDFExtractor:
    """Extract text and structure using PyMuPDF"""
    
    def __init__(self):
        self.name = "PyMuPDF"
    
    def extract_text(self, pdf_bytes: bytes, page_number: int) -> str:
        """
        Extract only text from a single PDF page
        
        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference
            
        Returns:
            Extracted text as string
        """
        import io
        
        try:
            # Open PDF from bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            if doc.page_count == 0:
                logger.error(f"No pages in PDF for page {page_number}")
                return ''
            
            # Extract text from first (and only) page
            page = doc[0]
            page_text = page.get_text("text")
            
            doc.close()
            
            return page_text
            
        except Exception as e:
            logger.error(f"PyMuPDF text extraction failed for page {page_number}: {e}")
            return ''
    
    def extract_structure(self, pdf_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """
        Extract structural metadata from a single PDF page
        
        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference
            
        Returns:
            Dictionary containing blocks, dimensions, images
        """
        import io
        
        try:
            # Open PDF from bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            if doc.page_count == 0:
                return {}
            
            page = doc[0]
            
            # Extract text blocks with position info
            blocks = page.get_text("dict")
            
            structure = {
                'blocks': self._process_blocks(blocks),
                'dimensions': {
                    'width': page.rect.width,
                    'height': page.rect.height
                },
                'images': self._extract_images(doc, page)
            }
            
            doc.close()
            return structure
            
        except Exception as e:
            logger.error(f"PyMuPDF structure extraction failed for page {page_number}: {e}")
            return {}
    
    def _process_blocks(self, blocks_dict: Dict) -> List[Dict]:
        """Process text blocks to extract structure"""
        processed_blocks = []
        
        for block in blocks_dict.get('blocks', []):
            if block.get('type') == 0:  # Text block
                block_info = {
                    'bbox': block.get('bbox'),
                    'lines': []
                }
                
                for line in block.get('lines', []):
                    line_text = ''
                    for span in line.get('spans', []):
                        line_text += span.get('text', '')
                    
                    if line_text.strip():
                        block_info['lines'].append({
                            'text': line_text,
                            'bbox': line.get('bbox'),
                            'font_size': line.get('spans', [{}])[0].get('size', 0) if line.get('spans') else 0
                        })
                
                if block_info['lines']:
                    processed_blocks.append(block_info)
        
        return processed_blocks
    
    def _extract_images(self, doc, page) -> List[Dict]:
        """Extract images from a page"""
        images = []
        
        try:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        images.append({
                            'index': img_index,
                            'data': img_base64
                        })
                    pix = None
                except Exception as e:
                    logger.warning(f"Failed to extract image: {e}")
        except Exception as e:
            logger.warning(f"Failed to get image list: {e}")
        
        return images