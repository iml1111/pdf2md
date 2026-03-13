# usecases/extraction.py
"""Extraction usecase functions"""

import asyncio

from loguru import logger

from extractors.clova_ocr_extractor import ClovaOCRExtractor
from extractors.llm_extractor import LLMExtractor
from extractors.pdfplumber_extractor import PDFPlumberExtractor
from extractors.pymupdf_extractor import PyMuPDFExtractor
from processors.image_converter import ImageConverter
from usecases.models import ExtractionResult, PageInput
from utils.config import Config
from utils.rate_limiter import APIRateLimiters


async def extract_pdfplumber(page_input: PageInput) -> ExtractionResult:
    """PDFPlumber 텍스트/테이블/메타데이터 추출"""
    try:
        extractor = PDFPlumberExtractor()

        text = await asyncio.to_thread(
            extractor.extract_text,
            page_input.page_bytes,
            page_input.page_number,
        )
        tables = await asyncio.to_thread(
            extractor.extract_tables,
            page_input.page_bytes,
            page_input.page_number,
        )
        metadata = await asyncio.to_thread(
            extractor.extract_metadata,
            page_input.page_bytes,
            page_input.page_number,
        )

        error = None
        if not text:
            error = "Text extraction failed"

        return ExtractionResult(
            extractor_name="pdfplumber",
            text=text,
            tables=tables,
            metadata=metadata,
            error=error,
        )
    except Exception as e:
        logger.error(
            f"pdfplumber extraction failed for page {page_input.page_number}: {e}"
        )
        return ExtractionResult(
            extractor_name="pdfplumber",
            text="",
            error=str(e),
        )


async def extract_clova_ocr(
    page_input: PageInput,
    config: Config,
) -> ExtractionResult:
    """CLOVA OCR API 호출 (네이티브 async)"""
    try:
        extractor = ClovaOCRExtractor(config.clova_ocr)

        text = await extractor.extract_text(
            page_input.page_bytes,
            page_input.page_number,
        )

        if text:
            return ExtractionResult(
                extractor_name="clova_ocr",
                text=text,
            )
        else:
            return ExtractionResult(
                extractor_name="clova_ocr",
                text="",
                error="CLOVA OCR returned no text",
            )
    except Exception as e:
        logger.error(
            f"CLOVA OCR extraction failed for page {page_input.page_number}: {e}"
        )
        return ExtractionResult(
            extractor_name="clova_ocr",
            text="",
            error=str(e),
        )


async def extract_llm_image(
    page_input: PageInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> ExtractionResult:
    """LLM 멀티모달 비전 추출"""
    try:
        # Convert PDF bytes to image
        image_converter = ImageConverter(dpi=config.image_dpi)
        page_image_bytes = image_converter.convert_page_to_image(
            page_input.page_bytes,
        )
        optimized_image = image_converter.optimize_for_ocr(page_image_bytes)

        # Rate limit and call LLM
        extractor = LLMExtractor(config)
        limiter = rate_limiters.get_limiter(config.llm.provider)
        await limiter.acquire()

        result = await asyncio.wait_for(
            asyncio.to_thread(
                extractor.extract_single_page_image,
                optimized_image,
                page_input.page_number,
                page_input.total_pages,
            ),
            timeout=600.0,
        )

        return ExtractionResult(
            extractor_name="llm_img",
            text=result.get('text', ''),
            metadata={
                k: v for k, v in result.items()
                if k not in ('text', 'error', 'page_number')
            },
            error=result.get('error'),
        )
    except asyncio.TimeoutError:
        logger.error(
            f"LLM image extraction timed out for page {page_input.page_number}"
        )
        return ExtractionResult(
            extractor_name="llm_img",
            text="",
            error="Timeout",
        )
    except Exception as e:
        logger.error(
            f"LLM image extraction failed for page {page_input.page_number}: {e}"
        )
        return ExtractionResult(
            extractor_name="llm_img",
            text="",
            error=str(e),
        )


async def extract_hyperlinks(page_input: PageInput) -> ExtractionResult:
    """PyMuPDF 하이퍼링크 추출"""
    try:
        extractor = PyMuPDFExtractor()

        result = await asyncio.to_thread(
            extractor.extract_hyperlinks,
            page_input.page_bytes,
            page_input.page_number,
        )

        return ExtractionResult(
            extractor_name="pymupdf",
            text="",
            hyperlinks=result.get('hyperlinks', []),
            error=result.get('error'),
        )
    except Exception as e:
        logger.error(
            f"PyMuPDF extraction failed for page {page_input.page_number}: {e}"
        )
        return ExtractionResult(
            extractor_name="pymupdf",
            text="",
            hyperlinks=[],
            error=str(e),
        )
