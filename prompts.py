"""
LLM System Prompts for PDF to Markdown Pipeline

This module contains all system prompts used by LLM modules in the pipeline.
All prompts are centrally managed here for consistency and maintainability.
"""

from typing import Dict, Any


# ===== EXTRACTION PROMPTS (Used by LLMExtractor) =====



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


# ===== INTEGRATION PROMPTS (Used by LLMMerger) =====

# LLM Merge Prompt Template - Used by LLMMerger for combining extraction results
def get_llm_merge_prompt(extraction_data: str) -> str:
    """
    Create prompt for LLM to merge extractions intelligently

    Args:
        extraction_data: Formatted extraction results from multiple extractors

    Returns:
        Complete prompt for merging
    """
    return f"""You are merging text extraction results from four complementary extractors, each with distinct strengths.

EXTRACTOR CHARACTERISTICS:

PDFPlumber (정적 텍스트/레이아웃 추출):
- Specializes in native PDF text extraction and structural analysis
- Excels at: table structures, precise positioning, formatting preservation
- Provides: clean text from native PDF elements, table data, layout measurements
- Strength: Static document structure and embedded text elements
- Limitation: May produce empty or fragmented results for image-heavy or scanned PDFs

LLM Image Extractor (텍스트/레이아웃 시각적 관점 추출):
- Analyzes document from visual perspective and contextual understanding
- Excels at: visual layout interpretation, document flow, contextual relationships
- Provides: visual structure insights, reading order, document organization
- Strength: Visual comprehension and contextual document understanding

CLOVA OCR (텍스트 정밀 추출):
- Performs precise optical character recognition on all text elements
- Excels at: comprehensive text recognition, Korean characters, fine detail extraction
- Provides: detailed text content, character-level precision, multilingual support
- Strength: Precise text recognition and character accuracy

PyMuPDF (하이퍼링크 감지):
- Specializes in detecting hyperlinks in PDF documents
- Excels at: identifying text with clickable links
- Provides: information about which text had links (URL not preserved)
- Strength: Accurate link detection from PDF annotations
- IMPORTANT: When links are found, mark them as [text](#) to indicate link presence

MERGING APPROACH:
1. Combine all four perspectives to create comprehensive extraction
2. Assess PDFPlumber output quality - if empty/fragmented, rely more on CLOVA OCR and LLM Image Extractor
3. Use PDFPlumber for structural foundation when it provides meaningful content
4. Apply CLOVA OCR for precise text content, especially crucial for image-heavy documents
5. Integrate LLM Image Extractor for visual context and reading flow
6. For image-dominant PDFs: prioritize CLOVA OCR and LLM Image Extractor over sparse PDFPlumber results
7. Cross-reference all sources to fill gaps and validate content
8. Preserve the strongest aspects from each extractor type based on document characteristics
9. Integrate PyMuPDF hyperlinks naturally into the text where relevant
10. When text had links, use markdown format [text](#) to mark link presence
11. Do NOT create separate sections for hyperlinks - integrate them inline with the content

EXTRACTION RESULTS:
{extraction_data}

HYPERLINK INTEGRATION INSTRUCTIONS:
- If HYPERLINK METADATA section shows "Text with link", format it as [text](#)
- The # symbol indicates a hyperlink was present, but URL is not preserved
- Example: "Text with link: 'GitHub'" → [GitHub](#)
- Do NOT include actual URLs
- Only mark text-based links, ignore image or other types

CRITICAL RULES:
- Leverage each extractor's unique strengths while considering document type
- Evaluate PDFPlumber output quality: if minimal/empty, focus on CLOVA OCR and LLM Image Extractor
- For text-rich native PDFs: utilize PDFPlumber's structural insights fully
- For image-heavy/scanned PDFs: rely primarily on CLOVA OCR and LLM Image Extractor
- Apply CLOVA OCR's precision for all text content, especially when PDFPlumber fails
- Incorporate LLM Image Extractor's visual understanding for context and flow
- Cross-reference all sources, giving more weight to sources with substantial content
- When PDFPlumber provides good results: preserve its table structures and formatting
- When PDFPlumber is sparse: don't force its limited data, focus on other extractors
- Maintain text accuracy and character precision from CLOVA OCR
- Follow visual layout and reading order insights from LLM Image Extractor
- Include hyperlinks from PyMuPDF naturally in the text using markdown link format [text](#) to mark link presence
- Do NOT include actual URLs, only mark that links existed
- Only mark text-based links, ignore image or area links
- Do NOT create separate sections for hyperlinks - integrate them inline with the content
- Include ALL relevant information from sources that provide meaningful content
- Adapt merging strategy based on the quality and completeness of each source
- Do NOT add information not present in any extraction source
- Return ONLY the merged text without commentary

Merged text:"""

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
    hyperlink_info = []

    for name, result in results.items():
        text = result.get('text', '').strip()

        # PyMuPDF 하이퍼링크 처리
        if name == 'pymupdf':
            hyperlinks = result.get('hyperlinks', [])
            if hyperlinks:
                # 하이퍼링크 정보를 별도로 수집하여 메타데이터로 전달
                for link in hyperlinks:
                    link_type = link.get('link_type', 'text')
                    if link.get('url'):
                        link_text = link.get('text', '').strip()
                        # 텍스트가 있는 경우만 처리 (텍스트 링크만)
                        if link_text and link_type == 'text':
                            hyperlink_info.append(f"Text with link: '{link_text}'")
                        # 텍스트가 없거나 다른 타입은 무시
                    # Internal page links are excluded for resume extraction
                    # elif link.get('page') is not None:  # Internal page link - SKIPPED
                text = ""  # 별도 섹션으로 표시하지 않음

        if text:
            # Add characteristic indicators for each extractor
            if name == 'clova_ocr':
                header = f"=== CLOVA OCR EXTRACTION (텍스트 정밀 추출) ==="
            elif name == 'pdfplumber':
                header = f"=== PDFPLUMBER EXTRACTION (정적 텍스트/레이아웃 추출) ==="
            elif name == 'llm_img':
                header = f"=== LLM_IMG EXTRACTION (텍스트/레이아웃 시각적 관점 추출) ==="
            else:
                header = f"=== {name.upper()} EXTRACTION ==="

            extraction_texts.append(f"{header}\n{text}\n")

    # 하이퍼링크 정보가 있으면 메타데이터로 추가
    if hyperlink_info:
        extraction_texts.append(f"=== HYPERLINK METADATA ===\n" + "\n".join(hyperlink_info) + "\n")

    return "\n".join(extraction_texts)
