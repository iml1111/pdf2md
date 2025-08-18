#!/usr/bin/env python3
"""
PDF to Markdown Conversion Pipeline
Main CLI interface with page-by-page processing
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
import fitz  # For PDF analysis and page splitting
import io

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config, Config
from utils.logger import setup_logger, logger
from utils.validators import validate_pdf_file

# Import PDF processing modules
from processors.single_page_pipeline import SinglePagePipeline
from processors.page_orchestrator import PageOrchestrator
from processors.final_orchestrator import FinalOrchestrator


class PDF2MDPipeline:
    """PDF to Markdown conversion pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PDF to Markdown pipeline
        
        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.page_orchestrator = PageOrchestrator(self.config)
        self.final_orchestrator = FinalOrchestrator(self.config)

        logger.info("✅ PDF to Markdown pipeline initialized")
    
    def _split_pdf_into_pages(self, pdf_path: str) -> List[bytes]:
        """
        Split PDF into individual page PDFs
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PDF bytes, one for each page
        """
        page_pdfs = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            
            for page_num in range(total_pages):
                single_page_doc = fitz.open()
                single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # Convert to bytes
                pdf_bytes = single_page_doc.tobytes()
                page_pdfs.append(pdf_bytes)
                
                single_page_doc.close()
            
            doc.close()
            logger.info(f"✅ Successfully split PDF into {len(page_pdfs)} pages")
            
        except Exception as e:
            logger.error(f"Failed to split PDF: {e}")
            raise
        
        return page_pdfs
    
    async def process_single_page(
        self,
        page_pdf_bytes: bytes,
        page_number: int,
        total_pages: int
    ) -> Dict[str, Any]:
        """
        Process a single PDF page through the pipeline
        
        Args:
            page_pdf_bytes: PDF bytes for single page
            page_number: Current page number (1-indexed)
            total_pages: Total number of pages
            
        Returns:
            Integrated page result
        """
        try:
            # Step 1: Process page through SinglePagePipeline
            single_page_pipeline = SinglePagePipeline(page_number, total_pages)
            merged_result = await single_page_pipeline.process_page(page_pdf_bytes)
            
            # Step 2: Integrate with PageOrchestrator (no formatting)
            integrated_result = self.page_orchestrator.integrate_page_results(
                merged_result, page_number, total_pages
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number}: {e}")
            return {
                'page_number': page_number,
                'content': '',
                'error': str(e)
            }
    
    async def process_pdf_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF pages in parallel batches
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of integrated page results
        """
        # Split PDF into pages
        page_pdfs = self._split_pdf_into_pages(pdf_path)
        total_pages = len(page_pdfs)
        
        if total_pages == 0:
            logger.error("No pages found in PDF")
            return []
        
        logger.info(f"🚀 Processing {total_pages} pages with PDF to Markdown pipeline")
        
        # Dynamic batch size based on total pages and memory considerations
        if total_pages <= 10:
            batch_size = total_pages  # Process all at once for small PDFs
        else:
            batch_size = 10  # Process 10 pages at a time for large PDFs
        
        page_results = []
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_tasks = []
            
            logger.info(f"📦 Processing batch: pages {batch_start + 1} to {batch_end}")
            
            for i in range(batch_start, batch_end):
                page_number = i + 1  # 1-indexed
                page_pdf_bytes = page_pdfs[i]
                
                # Create task for page processing
                task = self.process_single_page(page_pdf_bytes, page_number, total_pages)
                batch_tasks.append((page_number, task))
            
            # Process batch in parallel
            if batch_tasks:
                task_results = await asyncio.gather(*[task for _, task in batch_tasks])
                
                for (page_num, _), result in zip(batch_tasks, task_results):
                    page_results.append(result)
                
                logger.info(f"✅ Batch complete: {len(task_results)} pages processed")
        
        # Sort results by page number to ensure correct order
        page_results.sort(key=lambda x: x.get('page_number', 0))
        
        return page_results
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Main entry point for PDF to Markdown conversion
        
        Args:
            pdf_path: Path to input PDF
            output_path: Optional output path for markdown
            
        Returns:
            Path to generated markdown file
        """
        # Validate PDF
        pdf_path = Path(pdf_path)
        is_valid = validate_pdf_file(str(pdf_path))
        if not is_valid:
            raise ValueError("PDF validation failed")
        
        start_time = time.time()
        
        # Process all pages
        try:
            loop = asyncio.get_running_loop()
            page_results = loop.run_until_complete(self.process_pdf_pages(str(pdf_path)))
        except RuntimeError:
            page_results = asyncio.run(self.process_pdf_pages(str(pdf_path)))
        
        if not page_results:
            raise ValueError("Failed to process any pages from PDF")
        
        successful_pages = sum(1 for r in page_results if not r.get('error'))
        
        final_markdown = self.final_orchestrator.generate_final_document(
            page_results, str(pdf_path)
        )
        
        # Determine output path
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = pdf_path.with_suffix('.md')
        
        # Save markdown
        output_file.write_text(final_markdown, encoding='utf-8')
        
        # Calculate processing time & log summary
        processing_time = time.time() - start_time
        logger.info("✨ Processing Complete!")
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        # Output processing report to stdout instead of saving to file
        report = {
            'pdf_path': str(pdf_path),
            'output_path': str(output_file),
            'processing_time': processing_time,
            'total_pages': len(page_results),
            'successful_pages': successful_pages,
            'page_results': [
                {
                    'page_number': r.get('page_number'),
                    'has_error': bool(r.get('error')),
                    'error': r.get('error'),
                    'content_length': len(r.get('content', ''))
                }
                for r in page_results
            ]
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return str(output_file)


def main():
    """Main CLI entry point for PDF to Markdown pipeline"""
    parser = argparse.ArgumentParser(
        description="PDF to Markdown Conversion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples: python main.py --in document.pdf --out output.md
        """
    )
    
    parser.add_argument(
        '--in', '-i',
        dest='input_pdf',
        type=str,
        required=True,
        help='Input PDF file path'
    )
    
    parser.add_argument(
        '--out', '-o',
        dest='output_path',
        type=str,
        help='Output markdown file path (default: same as input with .md extension)'
    )
    
    parser.add_argument(
        '--llm',
        type=str,
        choices=['openai', 'anthropic'],
        default='anthropic',
        help='LLM provider to use '
    )
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    config = get_config()
    
    config.llm.provider = args.llm  
    logger.info(f"✅ Using {config.llm.provider} as LLM provider")
    
    try:
        logger.info("🚀 Initializing PDF to Markdown Pipeline...")
        pipeline = PDF2MDPipeline(config)
        output_file = pipeline.process_pdf(args.input_pdf, args.output_path)
        print(f"\n✅ Success! Markdown saved to: {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())