"""
Single Page Pipeline for processing individual PDF pages
"""

import asyncio
import io
import time
from typing import Any, Dict

from loguru import logger

from extractors.clova_ocr_extractor import ClovaOCRExtractor
from extractors.llm_extractor import LLMExtractor
from extractors.pdfplumber_extractor import PDFPlumberExtractor
from extractors.pymupdf_extractor import PyMuPDFExtractor
from processors.image_converter import ImageConverter
from processors.llm_merger import LLMMerger
from utils.config import Config
from utils.rate_limiter import APIRateLimiters


class SinglePagePipeline:
    """Pipeline for processing a single PDF page"""

    def __init__(self, page_number: int, total_pages: int, config: Config):
        self.page_number: int = page_number
        self.total_pages: int = total_pages
        self.config: Config = config

        self.pymupdf_extractor: PyMuPDFExtractor = PyMuPDFExtractor()
        self.pdfplumber_extractor: PDFPlumberExtractor = PDFPlumberExtractor()
        self.clova_ocr_extractor: ClovaOCRExtractor = ClovaOCRExtractor(self.config.clova_ocr)
        self.llm_extractor: LLMExtractor = LLMExtractor(self.config)

        self.image_converter: ImageConverter = ImageConverter(dpi=self.config.image_dpi)
        self.llm_merger: LLMMerger = LLMMerger(self.config)

        self.rate_limiters: APIRateLimiters = APIRateLimiters()

    async def process_page(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Process a single PDF page through all extractors"""
        start_time = time.time()
        logger.info(f"📄 Processing page {self.page_number}/{self.total_pages}")

        try:
            # Step 1: Convert page to image
            page_image_bytes = await self._convert_page_to_image(page_pdf_bytes)
            optimized_image = self.image_converter.optimize_for_ocr(page_image_bytes)

            # Step 2: Run all extractors in parallel
            # Fast extractors (non-LLM)
            fast_extraction_tasks = [
                ('pymupdf', self._run_pymupdf(page_pdf_bytes)),
                ('pdfplumber', self._run_pdfplumber(page_pdf_bytes)),
                ('clova_ocr', self._run_clova_ocr(page_pdf_bytes))
            ]
            # LLM image extractor (rate-limited internally)
            llm_extraction_tasks = [
                ('llm_img', self._run_llm_image(optimized_image))
            ]

            extraction_results = {}
            all_extraction_tasks = fast_extraction_tasks + llm_extraction_tasks

            all_results = await asyncio.gather(
                *[task for _, task in all_extraction_tasks],
                return_exceptions=True
            )

            for (name, _), result in zip(all_extraction_tasks, all_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Page {self.page_number} - {name}: {str(result)}")
                elif self._validate_result(result):
                    extraction_results[name] = result
                else:
                    result_str = str(result)
                    if len(result_str) > 1000:
                        result_str = result_str[:1000] + '... [truncated]'
                    logger.warning(f"⚠️ Page {self.page_number} - {name}: Invalid result format {result_str}")

            # Step 3: Merge text using LLM
            merged_text = await self.llm_merger.merge_text(extraction_results)

            # Step 4: Extract metadata and sources
            sources = self.llm_merger.get_valid_sources(extraction_results)
            metadata = self.llm_merger.extract_metadata(extraction_results)

            # Step 5: Build result
            processing_time = time.time() - start_time
            result = {
                'content': merged_text,
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
            import fitz

            pdf_stream = io.BytesIO(page_pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            if doc.page_count > 0:
                page = doc[0]
                mat = fitz.Matrix(self.config.image_dpi / 72.0, self.config.image_dpi / 72.0)
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
        """Run PyMuPDF extractor for hyperlinks only"""
        try:
            result = await asyncio.to_thread(
                self.pymupdf_extractor.extract_hyperlinks,
                page_pdf_bytes,
                self.page_number
            )
            return result
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for page {self.page_number}: {e}")
            return {'hyperlinks': [], 'error': str(e)}

    async def _run_pdfplumber(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Run pdfplumber extractor on single page"""
        try:
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

            result = {
                'text': text,
                'page_number': self.page_number,
                'tables': tables,
                **metadata
            }

            if not text:
                result['error'] = 'Text extraction failed'

            return result
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}

    async def _run_clova_ocr(self, page_pdf_bytes: bytes) -> Dict[str, Any]:
        """Run CLOVA OCR on single page PDF"""
        try:
            text = await self.clova_ocr_extractor.extract_text(
                page_pdf_bytes,
                self.page_number
            )

            if text:
                return {
                    'text': text,
                    'page_number': self.page_number
                }
            else:
                return {'text': '', 'error': 'CLOVA OCR returned no text'}
        except Exception as e:
            logger.error(f"CLOVA OCR extraction failed for page {self.page_number}: {e}")
            return {'text': '', 'error': str(e)}

    async def _run_llm_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run LLM image extractor on single page with rate limiting"""
        try:
            provider = self.config.llm.provider
            limiter = self.rate_limiters.get_limiter(provider)
            await limiter.acquire()

            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm_extractor.extract_single_page_image,
                    image_bytes,
                    self.page_number,
                    self.total_pages
                ),
                timeout=600.0
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
