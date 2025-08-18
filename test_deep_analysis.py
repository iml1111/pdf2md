#!/usr/bin/env python3
"""
Deep Analysis Pipeline for PDF to Markdown
Saves all intermediate results for detailed analysis
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
import fitz
import io
from datetime import datetime
import traceback

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config, Config
from utils.logger import setup_logger, logger
from utils.validators import validate_pdf_file

# Import PDF processing modules
from processors.single_page_pipeline import SinglePagePipeline
from processors.page_orchestrator import PageOrchestrator
from processors.final_orchestrator import FinalOrchestrator


class DeepAnalysisPipeline:
    """Deep analysis PDF to Markdown pipeline with comprehensive logging"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the deep analysis pipeline
        
        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.page_orchestrator = PageOrchestrator(self.config)
        self.final_orchestrator = FinalOrchestrator(self.config)
        
        # Create test_deep directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("test_deep") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Deep analysis output directory: {self.output_dir}")
        logger.info("✅ Deep Analysis Pipeline initialized")
    
    def _save_json(self, data: Any, filename: str, subdir: str = ""):
        """Save data to JSON file"""
        if subdir:
            filepath = self.output_dir / subdir / filename
            filepath.parent.mkdir(exist_ok=True)
        else:
            filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 Saved: {filepath.relative_to(self.output_dir)}")
    
    def _save_text(self, text: str, filename: str, subdir: str = ""):
        """Save text to file"""
        if subdir:
            filepath = self.output_dir / subdir / filename
            filepath.parent.mkdir(exist_ok=True)
        else:
            filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"💾 Saved: {filepath.relative_to(self.output_dir)}")
    
    def _save_bytes(self, data: bytes, filename: str, subdir: str = ""):
        """Save bytes to file"""
        if subdir:
            filepath = self.output_dir / subdir / filename
            filepath.parent.mkdir(exist_ok=True)
        else:
            filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(data)
        logger.info(f"💾 Saved: {filepath.relative_to(self.output_dir)}")
    
    def _analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF structure and metadata"""
        try:
            doc = fitz.open(pdf_path)
            
            structure = {
                "page_count": doc.page_count,
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted,
                "is_form_pdf": doc.is_form_pdf,
                "pages": []
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                
                page_info = {
                    "page_number": page_num + 1,
                    "dimensions": {
                        "width": page.rect.width,
                        "height": page.rect.height
                    },
                    "text_info": {
                        "has_text": len(text.strip()) > 0,
                        "text_length": len(text),
                        "text_preview": text[:500] if text else "NO TEXT FOUND"
                    },
                    "images_info": {
                        "has_images": len(page.get_images()) > 0,
                        "image_count": len(page.get_images()),
                        "image_details": []
                    },
                    "fonts": page.get_fonts(),
                    "links": page.get_links()
                }
                
                # Get image details
                for img_index, img in enumerate(page.get_images()):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    page_info["images_info"]["image_details"].append({
                        "index": img_index,
                        "xref": xref,
                        "width": pix.width,
                        "height": pix.height,
                        "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                        "bytes": pix.size
                    })
                    pix = None
                
                structure["pages"].append(page_info)
            
            doc.close()
            return structure
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF structure: {e}")
            return {"error": str(e)}
    
    def _split_pdf_into_pages(self, pdf_path: str) -> List[bytes]:
        """Split PDF into individual page PDFs"""
        page_pdfs = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            
            logger.info(f"📄 Splitting PDF into {total_pages} pages")
            
            for page_num in range(total_pages):
                # Create new PDF with single page
                single_page_doc = fitz.open()
                single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # Convert to bytes
                pdf_bytes = single_page_doc.tobytes()
                page_pdfs.append(pdf_bytes)
                
                # Save individual page PDF
                self._save_bytes(
                    pdf_bytes, 
                    f"page_{page_num + 1}.pdf", 
                    "01_split_pages"
                )
                
                single_page_doc.close()
            
            doc.close()
            logger.info(f"✅ Successfully split into {len(page_pdfs)} pages")
            
            return page_pdfs
            
        except Exception as e:
            logger.error(f"Failed to split PDF: {e}")
            raise
    
    async def _process_single_page(self, page_pdf: bytes, page_number: int, total_pages: int) -> Dict[str, Any]:
        """Process single page with detailed extraction tracking"""
        logger.info(f"🔄 Processing page {page_number}/{total_pages}")
        
        page_dir = f"02_page_{page_number:02d}_processing"
        
        try:
            # Initialize pipeline
            pipeline = SinglePagePipeline(
                page_number=page_number,
                total_pages=total_pages
            )
            
            # Track individual extractor results
            extractor_results = {}
            
            # 1. Convert page to image
            logger.info(f"  📸 Converting page {page_number} to image...")
            start_time = time.time()
            
            page_image_bytes = await pipeline._convert_page_to_image(page_pdf)
            optimized_image = pipeline.image_converter.optimize_for_ocr(page_image_bytes)
            
            # Save images
            self._save_bytes(page_image_bytes, "original_image.png", page_dir)
            self._save_bytes(optimized_image, "optimized_image.png", page_dir)
            
            # Check image dimensions
            from PIL import Image
            img = Image.open(io.BytesIO(page_image_bytes))
            img_info = {
                "original_size": img.size,
                "original_mode": img.mode,
                "original_format": img.format
            }
            opt_img = Image.open(io.BytesIO(optimized_image))
            img_info["optimized_size"] = opt_img.size
            img_info["optimized_mode"] = opt_img.mode
            self._save_json(img_info, "image_info.json", page_dir)
            
            conversion_time = time.time() - start_time
            logger.info(f"  ✅ Image conversion took {conversion_time:.2f}s")
            logger.info(f"     Original: {img.size}, Optimized: {opt_img.size}")
            
            # 2. Run each extractor individually and save results
            
            # PyMuPDF
            logger.info(f"  📖 Running PyMuPDF extractor...")
            start_time = time.time()
            pymupdf_result = await pipeline._run_pymupdf(page_pdf)
            pymupdf_time = time.time() - start_time
            extractor_results['pymupdf'] = {
                'result': pymupdf_result,
                'time': pymupdf_time,
                'text_length': len(pymupdf_result.get('text', ''))
            }
            self._save_json(pymupdf_result, "pymupdf_result.json", page_dir)
            if pymupdf_result.get('text'):
                self._save_text(pymupdf_result['text'], "pymupdf_text.txt", page_dir)
            logger.info(f"  ✅ PyMuPDF: {len(pymupdf_result.get('text', ''))} chars in {pymupdf_time:.2f}s")
            
            # PDFPlumber
            logger.info(f"  📖 Running PDFPlumber extractor...")
            start_time = time.time()
            pdfplumber_result = await pipeline._run_pdfplumber(page_pdf)
            pdfplumber_time = time.time() - start_time
            extractor_results['pdfplumber'] = {
                'result': pdfplumber_result,
                'time': pdfplumber_time,
                'text_length': len(pdfplumber_result.get('text', ''))
            }
            self._save_json(pdfplumber_result, "pdfplumber_result.json", page_dir)
            if pdfplumber_result.get('text'):
                self._save_text(pdfplumber_result['text'], "pdfplumber_text.txt", page_dir)
            logger.info(f"  ✅ PDFPlumber: {len(pdfplumber_result.get('text', ''))} chars in {pdfplumber_time:.2f}s")
            
            # Tesseract OCR
            logger.info(f"  🔍 Running Tesseract OCR...")
            start_time = time.time()
            tesseract_result = await pipeline._run_tesseract(optimized_image)
            tesseract_time = time.time() - start_time
            extractor_results['tesseract'] = {
                'result': tesseract_result,
                'time': tesseract_time,
                'text_length': len(tesseract_result.get('text', ''))
            }
            self._save_json(tesseract_result, "tesseract_result.json", page_dir)
            if tesseract_result.get('text'):
                self._save_text(tesseract_result['text'], "tesseract_text.txt", page_dir)
            logger.info(f"  ✅ Tesseract: {len(tesseract_result.get('text', ''))} chars in {tesseract_time:.2f}s")
            
            # LLM PDF (Claude only)
            if self.config.llm.provider == "anthropic":
                logger.info(f"  🤖 Running LLM PDF extractor...")
                start_time = time.time()
                llm_pdf_result = await pipeline._run_llm_pdf(page_pdf)
                llm_pdf_time = time.time() - start_time
                extractor_results['llm_pdf'] = {
                    'result': llm_pdf_result,
                    'time': llm_pdf_time,
                    'text_length': len(llm_pdf_result.get('text', ''))
                }
                self._save_json(llm_pdf_result, "llm_pdf_result.json", page_dir)
                if llm_pdf_result.get('text'):
                    self._save_text(llm_pdf_result['text'], "llm_pdf_text.txt", page_dir)
                logger.info(f"  ✅ LLM PDF: {len(llm_pdf_result.get('text', ''))} chars in {llm_pdf_time:.2f}s")
            
            # LLM Image
            logger.info(f"  🤖 Running LLM Image extractor...")
            start_time = time.time()
            llm_img_result = await pipeline._run_llm_image(optimized_image)
            llm_img_time = time.time() - start_time
            extractor_results['llm_img'] = {
                'result': llm_img_result,
                'time': llm_img_time,
                'text_length': len(llm_img_result.get('text', ''))
            }
            self._save_json(llm_img_result, "llm_img_result.json", page_dir)
            if llm_img_result.get('text'):
                self._save_text(llm_img_result['text'], "llm_img_text.txt", page_dir)
            logger.info(f"  ✅ LLM Image: {len(llm_img_result.get('text', ''))} chars in {llm_img_time:.2f}s")
            
            # 3. Merge results using LLM
            logger.info(f"  🔀 Merging extraction results...")
            start_time = time.time()
            
            # Prepare extraction results for merger
            merger_input = {}
            for name, data in extractor_results.items():
                if data['result'] and not data['result'].get('error'):
                    merger_input[name] = data['result']
            
            merged_text = await pipeline.llm_merger.merge_text(merger_input)
            merge_time = time.time() - start_time
            
            self._save_text(merged_text, "merged_text.txt", page_dir)
            self._save_json({"merger_input": list(merger_input.keys()), "merged_length": len(merged_text)}, "merge_info.json", page_dir)
            logger.info(f"  ✅ Merged: {len(merged_text)} chars in {merge_time:.2f}s")
            
            # 4. Page orchestration
            logger.info(f"  🎯 Running page orchestration...")
            start_time = time.time()
            
            # Build result for orchestration
            page_result = {
                'text': merged_text,
                'sources': pipeline.llm_merger.get_valid_sources(merger_input),
                'metadata': pipeline.llm_merger.extract_metadata(merger_input),
                'page_number': page_number,
                'total_pages': total_pages
            }
            
            orchestrated = self.page_orchestrator.integrate_page_results(
                page_result, page_number, total_pages
            )
            orchestration_time = time.time() - start_time
            
            self._save_json(orchestrated, "orchestrated_result.json", page_dir)
            if orchestrated.get('content'):
                self._save_text(orchestrated['content'], "orchestrated_text.txt", page_dir)
            logger.info(f"  ✅ Orchestrated in {orchestration_time:.2f}s")
            
            # Save summary
            summary = {
                'page_number': page_number,
                'total_pages': total_pages,
                'image_dimensions': img_info,
                'extractors': {
                    name: {
                        'success': not data['result'].get('error'),
                        'text_length': data['text_length'],
                        'processing_time': data['time'],
                        'error': data['result'].get('error') if data['result'].get('error') else None
                    }
                    for name, data in extractor_results.items()
                },
                'merged_text_length': len(merged_text),
                'orchestrated_text_length': len(orchestrated.get('content', '')),
                'merge_time': merge_time,
                'orchestration_time': orchestration_time,
                'total_time': sum(data['time'] for data in extractor_results.values()) + merge_time + orchestration_time
            }
            self._save_json(summary, "page_summary.json", page_dir)
            
            logger.info(f"✅ Page {page_number} completed in {summary['total_time']:.2f}s")
            
            return orchestrated
            
        except Exception as e:
            logger.error(f"❌ Page {page_number} processing failed: {e}")
            error_info = {
                'page_number': page_number,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self._save_json(error_info, "error.json", page_dir)
            return {'page_number': page_number, 'content': '', 'error': str(e)}
    
    async def process_pdf(self, pdf_path: str) -> None:
        """Process PDF with deep analysis"""
        start_time = time.time()
        
        # Validate PDF
        if not validate_pdf_file(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        
        pdf_path = Path(pdf_path)
        logger.info(f"📚 Starting deep analysis of: {pdf_path.name}")
        
        # 1. Analyze PDF structure
        logger.info("📊 Analyzing PDF structure...")
        pdf_structure = self._analyze_pdf_structure(str(pdf_path))
        self._save_json(pdf_structure, "00_pdf_structure.json")
        
        # 2. Split PDF into pages
        page_pdfs = self._split_pdf_into_pages(str(pdf_path))
        total_pages = len(page_pdfs)
        
        # 3. Process each page
        page_results = []
        for i, page_pdf in enumerate(page_pdfs, 1):
            result = await self._process_single_page(page_pdf, i, total_pages)
            page_results.append(result)
        
        # 4. Generate final markdown
        logger.info("📝 Generating final markdown document...")
        final_start = time.time()
        
        final_markdown = self.final_orchestrator.generate_final_document(
            page_results, str(pdf_path)
        )
        
        final_time = time.time() - final_start
        
        # Save final outputs
        self._save_text(final_markdown, "03_final_output.md")
        
        # Save all page results
        self._save_json(page_results, "03_all_pages_results.json")
        
        # Processing summary
        processing_time = time.time() - start_time
        summary = {
            "input_file": str(pdf_path),
            "output_directory": str(self.output_dir),
            "total_pages": total_pages,
            "processing_time": processing_time,
            "final_generation_time": final_time,
            "llm_provider": self.config.llm.provider,
            "successful_pages": sum(1 for r in page_results if not r.get('error')),
            "failed_pages": sum(1 for r in page_results if r.get('error')),
            "final_markdown_length": len(final_markdown),
            "timestamp": datetime.now().isoformat()
        }
        self._save_json(summary, "00_processing_summary.json")
        
        logger.info("✨ Deep analysis complete!")
        logger.info(f"📊 Summary:")
        logger.info(f"  - Total processing time: {processing_time:.2f} seconds")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - Final markdown: {len(final_markdown)} characters")
        
        print(f"\n{'='*60}")
        print(f"🔍 DEEP ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Results saved to: {self.output_dir.absolute()}")
        print(f"\n📂 Directory structure:")
        print(f"  00_pdf_structure.json - PDF metadata and structure")
        print(f"  00_processing_summary.json - Overall processing statistics")
        print(f"  01_split_pages/ - Individual PDF pages")
        print(f"  02_page_XX_processing/ - Detailed results for each page:")
        print(f"    - original_image.png - Page as image")
        print(f"    - optimized_image.png - OCR-optimized image")
        print(f"    - image_info.json - Image dimensions and properties")
        print(f"    - pymupdf_result.json/txt - PyMuPDF extraction")
        print(f"    - pdfplumber_result.json/txt - PDFPlumber extraction")
        print(f"    - tesseract_result.json/txt - Tesseract OCR extraction")
        print(f"    - llm_pdf_result.json/txt - LLM PDF extraction (Claude only)")
        print(f"    - llm_img_result.json/txt - LLM Image extraction")
        print(f"    - merged_text.txt - LLM-merged text")
        print(f"    - merge_info.json - Merge process details")
        print(f"    - orchestrated_result.json/txt - Page orchestration result")
        print(f"    - page_summary.json - Page processing statistics")
        print(f"  03_final_output.md - Final markdown document")
        print(f"  03_all_pages_results.json - All page results combined")
        print(f"\n⏱️  Total time: {processing_time:.2f} seconds")
        print(f"📄 Pages processed: {summary['successful_pages']}/{total_pages}")
        print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description='Deep analysis PDF to Markdown converter')
    parser.add_argument(
        '-i', '--in',
        dest='input_pdf',
        type=str,
        required=True,
        help='Input PDF file path'
    )
    parser.add_argument(
        '--llm',
        type=str,
        choices=['openai', 'anthropic'],
        default='anthropic',
        help='LLM provider to use'
    )
    
    args = parser.parse_args()
    
    # Setup detailed logging
    setup_logger(level="DEBUG")
    
    # Configure LLM provider
    config = get_config()
    config.llm.provider = args.llm
    logger.info(f"🤖 Using {config.llm.provider} as LLM provider")
    
    try:
        logger.info("🚀 Starting Deep Analysis Pipeline...")
        pipeline = DeepAnalysisPipeline(config)
        
        await pipeline.process_pdf(args.input_pdf)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))