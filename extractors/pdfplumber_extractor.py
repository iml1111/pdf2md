"""
pdfplumber based PDF text and table extractor
"""

import pdfplumber
from typing import Dict, List, Any
from utils.logger import logger, log_extraction_result


class PDFPlumberExtractor:
    """Extract text, tables, and structure using pdfplumber"""
    
    def __init__(self):
        self.name = "pdfplumber"
    
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
            
            with pdfplumber.open(pdf_stream) as pdf:
                if len(pdf.pages) == 0:
                    logger.error(f"No pages in PDF for page {page_number}")
                    return ''
                
                # Extract text from first (and only) page
                page = pdf.pages[0]
                page_text = page.extract_text() or ''
                
                logger.debug(f"pdfplumber extracted {len(page_text)} chars from page {page_number}")
                return page_text
            
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed for page {page_number}: {e}")
            return ''
    
    def extract_tables(self, pdf_bytes: bytes, page_number: int) -> List[Dict]:
        """
        Extract tables from a single PDF page
        
        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference
            
        Returns:
            List of table data dictionaries
        """
        import io
        
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            
            with pdfplumber.open(pdf_stream) as pdf:
                if len(pdf.pages) == 0:
                    return []
                
                page = pdf.pages[0]
                tables = page.extract_tables()
                
                table_results = []
                for table_idx, table in enumerate(tables):
                    if table:
                        table_data = {
                            'index': table_idx,
                            'data': table,
                            'markdown': self._table_to_markdown(table)
                        }
                        table_results.append(table_data)
                
                return table_results
            
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed for page {page_number}: {e}")
            return []
    
    def extract_metadata(self, pdf_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """
        Extract metadata from a single PDF page
        
        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference
            
        Returns:
            Dictionary containing page dimensions and character samples
        """
        import io
        
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            
            with pdfplumber.open(pdf_stream) as pdf:
                if len(pdf.pages) == 0:
                    return {}
                
                page = pdf.pages[0]
                
                metadata = {
                    'dimensions': {
                        'width': page.width,
                        'height': page.height
                    }
                }
                
                # Extract character information (limited for performance)
                try:
                    chars = page.chars[:100]  # Limit to first 100 chars
                    metadata['chars_sample'] = [{
                        'text': char.get('text', ''),
                        'fontname': char.get('fontname', ''),
                        'size': char.get('size', 0)
                    } for char in chars]
                except:
                    pass
                
                return metadata
            
        except Exception as e:
            logger.error(f"pdfplumber metadata extraction failed for page {page_number}: {e}")
            return {}
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert table to markdown format"""
        if not table or not table[0]:
            return ""
        
        markdown_lines = []
        
        # Header row
        header = table[0]
        markdown_lines.append("| " + " | ".join(str(cell) if cell else "" for cell in header) + " |")
        
        # Separator
        markdown_lines.append("|" + "|".join([" --- " for _ in header]) + "|")
        
        # Data rows
        for row in table[1:]:
            markdown_lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
        
        return "\n".join(markdown_lines)