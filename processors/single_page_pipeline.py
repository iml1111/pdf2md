"""
Single Page Pipeline for processing individual PDF pages
"""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import io

from utils.config import get_config, Config
from utils.logger import logger
from utils.rate_limiter import APIRateLimiters

# Import extractors with type hints
from extractors.pymupdf_extractor import PyMuPDFExtractor
from extractors.pdfplumber_extractor import PDFPlumberExtractor
from extractors.tesseract_extractor import TesseractExtractor
from extractors.llm_extractor import LLMExtractor

# Import processors with type hints
from processors.image_converter import ImageConverter
from processors.llm_merger import LLMMerger


class SinglePagePipeline:
    """Pipeline for processing a single PDF page"""
    
    def __init__(self, page_number: int, total_pages: int):
        """
        Initialize single page pipeline
        
        Args:
            page_number: Current page number (1-indexed)
            total_pages: Total number of pages in document
        """
        self.page_number: int = page_number
        self.total_pages: int = total_pages
        self.config: Config = get_config()
                
        # Initialize extractors as individual variables with type hints
        self.pymupdf_extractor: PyMuPDFExtractor = PyMuPDFExtractor()
        self.pdfplumber_extractor: PDFPlumberExtractor = PDFPlumberExtractor()
        self.tesseract_extractor: TesseractExtractor = TesseractExtractor()
        self.llm_extractor: LLMExtractor = LLMExtractor()
        
        # Initialize processors as individual variables with type hints
        self.image_converter: ImageConverter = ImageConverter(dpi=self.config.image_dpi)
        self.llm_merger: LLMMerger = LLMMerger()
        
        # Initialize rate limiters (instance-specific, not singleton)
        self.rate_limiters: APIRateLimiters = APIRateLimiters()
        
    async def process_page(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Process a single PDF page through all extractors
        
        Args: page_pdf_bytes: PDF bytes for single page
        Returns: Integrated page results without markdown formatting
        """
        start_time = time.time()
        logger.info(f"📄 Processing page {self.page_number}/{self.total_pages}")
        
        try:
            # Step 1: Convert page to image
            logger.debug(f"Converting page {self.page_number} to image")
            page_image_bytes = await self._convert_page_to_image(page_pdf_bytes)
            optimized_image = self.image_converter.optimize_for_ocr(page_image_bytes)
            
            # Step 2: Run extractors in two phases for optimized parallel processing
            # Phase 1: Fast extractors (non-LLM) - run fully in parallel
            fast_extraction_tasks = [
                ('pymupdf', self._run_pymupdf(page_pdf_bytes)),
                ('pdfplumber', self._run_pdfplumber(page_pdf_bytes)),
                ('tesseract', self._run_tesseract(optimized_image))
            ]
            # Phase 2: LLM extractors - will be rate-limited
            llm_extraction_tasks = [
                ('llm_pdf', self._run_llm_pdf(page_pdf_bytes)),
                ('llm_img', self._run_llm_image(optimized_image))
            ]
            
            # Combine all extraction tasks
            extraction_results = {}
            all_extraction_tasks = fast_extraction_tasks + llm_extraction_tasks
            
            # Execute all extractors in parallel            
            all_results = await asyncio.gather(
                *[task for _, task in all_extraction_tasks],
                return_exceptions=True
            )
            
            # Process all results with single loop
            for (name, _), result in zip(all_extraction_tasks, all_results):
                if isinstance(result, Exception):
                    raise Exception(f"❌ Page {self.page_number} - {name}: {str(result)}")
                elif self._validate_result(result):
                    extraction_results[name] = result
                else:
                    logger.warning(f"⚠️ Page {self.page_number} - {name}: Invalid result format {result}")
            
            # Step 3: Merge text using LLM
            logger.debug(f"Merging text for page {self.page_number} using LLM")
            merged_text = await self.llm_merger.merge_text(extraction_results)
            
            # Step 4: Extract metadata and sources
            sources = self.llm_merger.get_valid_sources(extraction_results)
            metadata = self.llm_merger.extract_metadata(extraction_results)
            
            # Step 5: Build result dictionary
            processing_time = time.time() - start_time
            result = {
                'text': merged_text,
                'sources': sources,
                'metadata': metadata,
                'page_number': self.page_number,
                'total_pages': self.total_pages,
                'processing_time': processing_time,
                'successful_extractors': sum(
                    1 for r in extraction_results.values() 
                    if not r.get('error')
                )
            }
            logger.info(f"✅ Page {self.page_number} processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process page {self.page_number}: {e}")
            raise e
    
    async def _convert_page_to_image(self, page_pdf_bytes: bytes) -> bytes:
        """Convert single page PDF to image"""
        try:
            import fitz  # PyMuPDF
            
            # Open PDF from bytes
            pdf_stream = io.BytesIO(page_pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            if doc.page_count > 0:
                page = doc[0]  # Single page PDF
                mat = fitz.Matrix(self.config.image_dpi/72.0, self.config.image_dpi/72.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pix = None
                doc.close()
                return img_data
            
            doc.close()
            raise ValueError("No pages in PDF")
            
        except Exception as e:
            logger.error(f"Failed to convert page {self.page_number} to image: {e}")
            raise
    
    async def _run_pymupdf(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Run PyMuPDF extractor on single page"""
        try:
            # Extract text and structure separately
            text = await asyncio.to_thread(
                self.pymupdf_extractor.extract_text,
                page_pdf_bytes,
                self.page_number
            )
            
            structure = await asyncio.to_thread(
                self.pymupdf_extractor.extract_structure,
                page_pdf_bytes,
                self.page_number
            )
            
            # Combine results
            result = {
                'text': text,
                'page_number': self.page_number,
                **structure  # Unpack structure data
            }
            
            # Add error if text extraction failed
            if not text:
                result['error'] = 'Text extraction failed'
            
            return result
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}
    
    async def _run_pdfplumber(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Run pdfplumber extractor on single page"""
        try:
            # Extract text, tables, and metadata separately
            text = await asyncio.to_thread(
                self.pdfplumber_extractor.extract_text,
                page_pdf_bytes,
                self.page_number
            )
            
            tables = await asyncio.to_thread(
                self.pdfplumber_extractor.extract_tables,
                page_pdf_bytes,
                self.page_number
            )
            
            metadata = await asyncio.to_thread(
                self.pdfplumber_extractor.extract_metadata,
                page_pdf_bytes,
                self.page_number
            )
            
            # Combine results
            result = {
                'text': text,
                'page_number': self.page_number,
                'tables': tables,
                **metadata  # Unpack metadata
            }
            
            # Add error if text extraction failed
            if not text:
                result['error'] = 'Text extraction failed'
            
            return result
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}
    
    async def _run_tesseract(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run Tesseract OCR on single page image"""
        try:
            # Tesseract expects image bytes, not PDF bytes
            # Use extract_from_images which is already designed for this
            result = await asyncio.to_thread(
                self.tesseract_extractor.extract_from_images,
                [image_bytes]
            )
            
            # Add page number to result
            if result and 'page_number' not in result:
                result['page_number'] = self.page_number
            
            return result
        except Exception as e:
            logger.error(f"Tesseract extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}
    
    async def _run_llm_pdf(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Run LLM PDF extractor on single page with rate limiting"""
        try:
            # Get appropriate rate limiter based on LLM provider
            provider = self.config.llm.provider
            limiter = self.rate_limiters.get_limiter(provider)
            
            # Acquire rate limit before making LLM call
            await limiter.acquire()
            
            # Use the single page method directly
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm_extractor.extract_single_page_pdf,
                    page_pdf_bytes,
                    self.page_number,
                    self.total_pages
                ),
                timeout=600.0  # 10 minutes timeout for single page
            )
                
        except asyncio.TimeoutError:
            logger.error(f"LLM PDF extraction timed out for page {self.page_number}")
            return {'text': '', 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"LLM PDF extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}
    
    async def _run_llm_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run LLM image extractor on single page with rate limiting"""
        try:
            # Get appropriate rate limiter based on LLM provider
            provider = self.config.llm.provider
            limiter = self.rate_limiters.get_limiter(provider)
            
            # Acquire rate limit before making LLM call
            await limiter.acquire()
            
            # Use the more efficient single page method directly
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm_extractor.extract_single_page_image,
                    image_bytes,
                    self.page_number,
                    self.total_pages
                ),
                timeout=600.0  # 10 minutes timeout for single page
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM image extraction timed out for page {self.page_number}")
            return {'text': '', 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"LLM image extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}
    
    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate extraction result"""
        return bool(result and not result.get('error'))
