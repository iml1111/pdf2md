"""
LLM System Prompts for PDF to Markdown Pipeline

This module contains all system prompts used by LLM modules in the pipeline.
All prompts are centrally managed here for consistency and maintainability.
"""

from typing import Dict, Any


# ===== EXTRACTION PROMPTS (Used by LLMExtractor) =====

# PDF Extraction Prompt - Used for full PDF extraction
PDF_EXTRACTION_PROMPT = """Extract ALL text from this PDF document COMPLETELY.

CRITICAL REQUIREMENTS:
1. PRESERVE ALL CONTENT - Do not summarize or abbreviate ANYTHING
2. Extract text in reading order
3. Include all headers, footers, page numbers
4. Extract all tables, maintaining structure
5. Include all lists and bullet points
6. Preserve all technical terms and codes exactly
7. Extract all metadata (titles, authors, dates, etc.)
8. Include figure/image captions and labels
9. DO NOT omit any paragraphs or sections
10. DO NOT use phrases like "continues with..." or "and so on"

전문 출력 필수 - 절대 요약하지 마세요!
모든 내용을 빠짐없이 포함해야 합니다.

Provide the COMPLETE text extraction."""

# Image Extraction Prompt - Used for image-based extraction
IMAGE_EXTRACTION_PROMPT = """Extract ALL text from these document images COMPLETELY.

CRITICAL INSTRUCTIONS:
1. Extract EVERY piece of text visible in the images
2. DO NOT summarize or skip any content
3. Process in page order
4. Include headers, footers, page numbers
5. Extract tables with proper structure
6. Include all captions and labels
7. Preserve technical terms exactly
8. Extract handwritten text if present
9. Include marginalia and annotations
10. Never abbreviate or use ellipsis

전체 텍스트를 반드시 추출하세요.
요약이나 생략은 절대 금지입니다.

Output the COMPLETE text from all pages."""

# Single Page PDF Extraction Prompt - For page-specific PDF extraction
SINGLE_PAGE_PDF_PROMPT = """Extract ALL text from this single PDF page COMPLETELY.

This is page {page_number} of {total_pages}.

REQUIREMENTS:
1. Extract EVERY word and character from this specific page
2. DO NOT summarize or abbreviate anything
3. Include all headers, footers, and margin content
4. Preserve the exact content of this page
5. Include any tables, lists, or special formatting as text

Output the complete text from this page only."""

# Single Page Image Extraction Prompt - For page-specific image extraction  
SINGLE_PAGE_IMAGE_PROMPT = """Extract ALL text from this page image with maximum detail.

This is page {page_number} of {total_pages}.

CRITICAL EXTRACTION REQUIREMENTS:
1. Extract EVERY piece of text, no matter how small or faint
2. Include text from:
   - Main content area
   - Headers and footers  
   - Page numbers
   - Margins and annotations
   - Watermarks or background text
   - Tables, charts, and diagrams
   - Image captions and labels
   - Any handwritten notes
3. If the page appears blank or has minimal text:
   - Still report any visible marks, numbers, or symbols
   - Note if it's truly blank with "[Blank page]"
   - Include any subtle text like "This page intentionally left blank"
4. For images with embedded text:
   - Extract ALL text within images
   - Include chart labels, legends, axis titles
   - Extract text from logos or branding
5. DO NOT skip anything - even single characters matter

IMPORTANT: Even if the page seems to have little content, extract whatever is visible. 
Never return empty results unless the page is completely blank.

Output ALL extracted text from this page."""


# ===== INTEGRATION PROMPTS (Used by LLMMerger and PageOrchestrator) =====

# LLM Merge Prompt Template - Used by LLMMerger for combining extraction results
def get_llm_merge_prompt(extraction_data: str) -> str:
    """
    Create prompt for LLM to merge extractions intelligently
    
    Args:
        extraction_data: Formatted extraction results from multiple extractors
        
    Returns:
        Complete prompt for merging
    """
    return f"""You are merging text extraction results from multiple PDF extractors.
Each extractor has different strengths and reliability levels based on the PDF type.

EXTRACTOR RELIABILITY HIERARCHY:
1. PyMuPDF & PDFPlumber (HIGHEST PRIORITY):
   - Most accurate for native PDF text (not scanned)
   - When these have text, prefer their content over others
   - Especially trust for: dates, numbers, emails, URLs, proper formatting

2. Tesseract OCR (MEDIUM PRIORITY):
   - Primary source for scanned PDFs when PyMuPDF/PDFPlumber have no text
   - May have OCR errors: spaces in words, wrong characters (0/O, 1/l)
   - Good for: overall content structure, main text body
   - Needs correction for: OCR artifacts, broken words, spacing issues

3. LLM-based extractors (LOWER PRIORITY):
   - LLM PDF & LLM Image have similar reliability
   - May hallucinate or modify content
   - Use for: filling gaps, understanding context
   - Be cautious with: specific numbers, dates, contact information

MERGING STRATEGY:
1. If PyMuPDF or PDFPlumber have text → use as primary source
2. If they're empty (scanned PDF) → use Tesseract as base, correct OCR errors
3. Use LLM extractors to fill missing information and verify context
4. For conflicts in specific data (numbers, emails, dates):
   - Trust PyMuPDF/PDFPlumber first
   - Then Tesseract (with OCR correction)
   - LLM extractors last

COMMON OCR PATTERNS TO FIX:
- Broken Korean: "개 발 자" → "개발자", "니 에서" → "트에서", "호 Contacts" → "Contacts"
- Wrong symbols: "호" at line start → remove, "4" in phone → check context
- Spacing: Remove unnecessary spaces within Korean words
- Email/URL: Prefer versions without spaces or unusual characters

VALIDATION HINTS:
- Korean phone: Should match 010-XXXX-XXXX pattern
- Email: Should have @ and valid domain
- Dates: Prefer consistent format (YYYY.MM or YYYY-MM)
- Korean text: Should not have spaces within syllables

EXTRACTION RESULTS:
{extraction_data}

CRITICAL RULES:
- When PyMuPDF/PDFPlumber have content, use their structure and formatting
- Fix Tesseract OCR errors but preserve its unique content
- Don't trust LLM extractors for exact numbers/dates if other sources exist
- Include ALL unique information from all sources
- For contact info (phone, email), prefer the most common/repeated version
- Maintain original document structure and order
- Do NOT add information not present in any extraction
- Return ONLY the merged text without commentary

Merged text:"""


# Page Integration Prompt Template - Used by PageOrchestrator
def get_page_integration_prompt(
    page_number: int,
    total_pages: int,
    text: str,
) -> str:
    """
    Create prompt for integrating single page extraction results
    
    Args:
        page_number: Current page number
        total_pages: Total number of pages
        text: Merged text from extractors
        
    Returns:
        Complete prompt for page integration
    """
    return f"""You are integrating extraction results from page {page_number} of {total_pages}.

Multiple extractors have processed this page and their results have been merged.
Your task is to create a clean, integrated version of the page content.

IMPORTANT INSTRUCTIONS:
1. DO NOT add any markdown formatting (no #, *, **, etc.)
2. Preserve ALL content - do not summarize or omit anything
3. Maintain the natural reading order of the page
4. Fix any obvious OCR errors or inconsistencies
5. Preserve tables, lists, and structural elements as plain text
6. Return ONLY the integrated text content

Page content to integrate:
---
{text}
---

Please provide the integrated page content as plain text:"""


# ===== GENERATION PROMPTS (Used by FinalOrchestrator) =====

# Final Document Generation Prompt Template
def get_final_document_prompt(
    metadata: Dict[str, Any],
    combined_content: str
) -> str:
    """
    Create prompt for final markdown document generation
    
    Args:
        metadata: Document metadata (source file, pages, confidence, etc.)
        combined_content: Combined content from all pages
        
    Returns:
        Complete prompt for final document generation
    """
    return f"""Convert PDF content to clean markdown.

Source: {metadata['source_file']} ({metadata['total_pages']} pages)

Rules:
- Use proper markdown formatting (#, ##, ###, -, *, **, tables)
- PRESERVE ALL CONTENT COMPLETELY - no summarization or omission  
- Fix OCR errors naturally
- Remove [PAGE X] markers
- Maintain original structure and order

Content:
{combined_content}

Generate the complete markdown document:"""


# ===== UTILITY FUNCTIONS =====

def format_extraction_data(results: Dict[str, Dict]) -> str:
    """
    Format extraction results for LLM processing with reliability indicators
    
    Args:
        results: Dictionary mapping extractor name to results
        
    Returns:
        Formatted string with all extraction results and reliability notes
    """
    extraction_texts = []
    
    for name, result in results.items():
        text = result.get('text', '').strip()
        if text:
            # Add reliability indicators for each extractor
            if name == 'pymupdf':
                header = f"=== PYMUPDF EXTRACTION (Native PDF Reader - HIGH RELIABILITY) ==="
            elif name == 'pdfplumber':
                header = f"=== PDFPLUMBER EXTRACTION (Native PDF Reader - HIGH RELIABILITY) ==="
            elif name == 'tesseract':
                header = f"=== TESSERACT EXTRACTION (OCR - MEDIUM RELIABILITY, MAY HAVE ERRORS) ==="
            elif name == 'llm_pdf':
                header = f"=== LLM_PDF EXTRACTION (AI-based - LOWER RELIABILITY FOR SPECIFICS) ==="
            elif name == 'llm_img':
                header = f"=== LLM_IMG EXTRACTION (AI-based - LOWER RELIABILITY FOR SPECIFICS) ==="
            else:
                header = f"=== {name.upper()} EXTRACTION ==="
            
            extraction_texts.append(f"{header}\n{text}\n")
    
    return "\n".join(extraction_texts)