#!/usr/bin/env python3
"""
PDF to Markdown Conversion Pipeline
Main CLI interface with page-by-page processing
"""

import argparse
import asyncio
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import fitz

# pdfminer FontBBox 경고 스팸 억제 (pdfplumber 내부 의존성)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from usecases.extraction import (
    extract_clova_ocr,
    extract_hyperlinks,
    extract_llm_image,
    extract_pdfplumber,
)
from usecases.finalizing import finalize_document
from usecases.merging import merge_page
from usecases.models import (
    ExtractionResult,
    FinalizeInput,
    MergeInput,
    PageInput,
)
from utils.config import Config, get_config
from utils.logger import logger, setup_logger
from utils.rate_limiter import APIRateLimiters
from utils.validators import validate_pdf_file


def batched(iterable, n):
    """itertools.batched polyfill for Python 3.11"""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


def split_pdf(pdf_path: str) -> List[bytes]:
    """Split PDF into individual page bytes"""
    page_pdfs = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            page_pdfs.append(single_page_doc.tobytes())
            single_page_doc.close()
        doc.close()
        logger.info(f"✅ Successfully split PDF into {len(page_pdfs)} pages")
    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        raise
    return page_pdfs


async def extract_all_for_page(
    page: PageInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> List[ExtractionResult]:
    """Run all 4 extractors in parallel for a single page"""
    results = await asyncio.gather(
        extract_pdfplumber(page),
        extract_clova_ocr(page, config),
        extract_llm_image(page, config, rate_limiters),
        extract_hyperlinks(page),
        return_exceptions=True,
    )

    # Convert exceptions to error ExtractionResults + diagnostic logging
    extraction_results = []
    extractor_names = ["pdfplumber", "clova_ocr", "llm_img", "pymupdf"]
    for name, result in zip(extractor_names, results):
        if isinstance(result, Exception):
            logger.error(f"❌ Page {page.page_number} - {name}: {result}")
            extraction_results.append(
                ExtractionResult(extractor_name=name, text="", error=str(result))
            )
        else:
            if result.error:
                logger.warning(
                    f"⚠️ Page {page.page_number} - {result.extractor_name}: {result.error}"
                )
            extraction_results.append(result)

    return extraction_results


async def run_pipeline(
    pdf_path: Path,
    config: Config,
) -> Dict[str, Any]:
    """Main pipeline logic: extract → merge → finalize"""
    start_time = time.time()
    rate_limiters = APIRateLimiters()

    # --- Step 0: Split PDF ---
    page_bytes_list = split_pdf(str(pdf_path))
    total_pages = len(page_bytes_list)

    if total_pages == 0:
        raise ValueError("No pages found in PDF")

    pages = [
        PageInput(page_bytes=b, page_number=i + 1, total_pages=total_pages)
        for i, b in enumerate(page_bytes_list)
    ]

    logger.info(f"🚀 Processing {total_pages} pages with PDF to Markdown pipeline")

    # --- Step 1: Extract (per-page parallel, batched) ---
    all_extractions: List[List[ExtractionResult]] = []
    batch_size = total_pages if total_pages <= 10 else 10

    for batch in batched(pages, batch_size):
        logger.info(
            f"📦 Processing batch: pages {batch[0].page_number} to {batch[-1].page_number}"
        )
        batch_results = await asyncio.gather(*[
            extract_all_for_page(page, config, rate_limiters)
            for page in batch
        ])
        all_extractions.extend(batch_results)
        logger.info(f"✅ Batch complete: {len(batch_results)} pages processed")

    # --- Step 2: Merge (per-page) ---
    merge_results = []
    for page_input, extractions in zip(pages, all_extractions):
        result = await merge_page(
            MergeInput(
                page_number=page_input.page_number,
                extraction_results=extractions,
            ),
            config,
            rate_limiters,
        )
        merge_results.append(result)

    # --- Step 3: Finalize ---
    final = await finalize_document(
        FinalizeInput(
            merge_results=merge_results,
            total_pages=total_pages,
            source_file=pdf_path.name,
        ),
        config,
    )

    processing_time = time.time() - start_time

    return {
        'final': final,
        'merge_results': merge_results,
        'total_pages': total_pages,
        'processing_time': processing_time,
    }


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="PDF to Markdown Conversion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples: python main.py --in document.pdf --out output.md",
    )
    parser.add_argument(
        '--in', '-i', dest='input_pdf', type=str, required=True,
        help='Input PDF file path',
    )
    parser.add_argument(
        '--out', '-o', dest='output_path', type=str,
        help='Output markdown file path (default: same as input with .md extension)',
    )
    parser.add_argument(
        '--llm', type=str, choices=['openai', 'anthropic'], default='anthropic',
        help='LLM provider to use',
    )
    return parser.parse_args()


def main():
    """CLI entry point"""
    args = parse_args()
    setup_logger(level="INFO")
    config = get_config()
    config.llm.provider = args.llm
    logger.info(f"✅ Using {config.llm.provider} as LLM provider")

    # Validate credentials
    try:
        config.llm.validate_credentials()
    except ValueError as e:
        logger.error(f"LLM 설정 오류: {e}")
        print(f"\n❌ {e}", file=sys.stderr)
        return 1

    try:
        config.clova_ocr.validate_credentials()
    except ValueError as e:
        logger.error(f"CLOVA OCR 설정 오류: {e}")
        print(f"\n❌ {e}", file=sys.stderr)
        return 1

    try:
        # Validate PDF
        pdf_path = Path(args.input_pdf)
        is_valid = validate_pdf_file(str(pdf_path))
        if not is_valid:
            raise ValueError("PDF validation failed")

        logger.info("🚀 Initializing PDF to Markdown Pipeline...")

        # Run pipeline
        pipeline_result = asyncio.run(run_pipeline(pdf_path, config))

        final = pipeline_result['final']
        merge_results = pipeline_result['merge_results']
        total_pages = pipeline_result['total_pages']
        processing_time = pipeline_result['processing_time']

        # Write output
        if args.output_path:
            output_file = Path(args.output_path)
        else:
            output_file = pdf_path.with_suffix('.md')

        output_file.write_text(final.markdown, encoding='utf-8')

        # Calculate stats
        successful_pages = sum(
            1 for mr in merge_results
            if not mr.error and mr.merged_text.strip()
        )

        # Log summary
        logger.info("✨ Processing Complete!")
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")

        # JSON report to stdout
        report = {
            'pdf_path': str(pdf_path),
            'output_path': str(output_file),
            'processing_time': processing_time,
            'total_pages': total_pages,
            'successful_pages': successful_pages,
            'page_results': [
                {
                    'page_number': mr.page_number,
                    'has_error': bool(mr.error),
                    'error': mr.error,
                    'content_length': len(mr.merged_text),
                }
                for mr in merge_results
            ],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))

        total_content_length = sum(len(mr.merged_text) for mr in merge_results)
        if total_content_length == 0:
            logger.warning(
                "⚠️ All pages produced empty content — output is fallback only"
            )

        # Exit codes
        if successful_pages == 0:
            print(
                f"\n❌ Failed: No pages produced content. Output saved to: {output_file}",
                file=sys.stderr,
            )
            return 1
        elif successful_pages < total_pages:
            print(
                f"\n⚠️ Partial success: {successful_pages}/{total_pages} pages extracted."
                f" Markdown saved to: {output_file}"
            )
            return 0
        else:
            print(f"\n✅ Success! Markdown saved to: {output_file}")
            return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
