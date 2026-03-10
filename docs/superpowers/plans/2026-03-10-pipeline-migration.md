# Pipeline Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate production-verified pipeline architecture from leslie-queue-worker to pdf2md CLI.

**Architecture:** Replace 5-extractor pipeline (PyMuPDF text + PDFPlumber + Tesseract + LLM PDF + LLM Image → PageOrchestrator → Final) with production 4-extractor pipeline (PDFPlumber + PyMuPDF hyperlinks + CLOVA OCR + LLM Image → LLMMerger → Final). Add Google Gemini as 3rd LLM provider.

**Tech Stack:** Python 3.11+, PyMuPDF (fitz), pdfplumber, aiohttp (CLOVA OCR), anthropic, google-genai, openai, pydantic, loguru

**Spec:** `docs/superpowers/specs/2026-03-10-pipeline-migration-design.md`

**Production reference:** `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `utils/config.py` | Add Google + CLOVA OCR config, 3-provider support |
| Modify | `utils/rate_limiter.py` | Add Google rate limiter (3 req/s) |
| Modify | `requirements.txt` | Remove tesseract deps, add google-genai + aiohttp |
| Delete | `extractors/tesseract_extractor.py` | Replaced by CLOVA OCR |
| Create | `extractors/clova_ocr_extractor.py` | CLOVA OCR API integration |
| Rewrite | `extractors/pymupdf_extractor.py` | Hyperlink-only extraction (remove text/structure) |
| Keep | `extractors/pdfplumber_extractor.py` | Already matches production |
| Rewrite | `extractors/llm_extractor.py` | 3-provider support, remove PDF extraction |
| Rewrite | `prompts.py` | Production prompts (remove unused, update merge/format) |
| Rewrite | `processors/llm_merger.py` | 3-provider merge with production strategy |
| Rewrite | `processors/final_orchestrator.py` | 3-provider final generation |
| Rewrite | `processors/single_page_pipeline.py` | 4-extractor pipeline, remove Tesseract |
| Delete | `processors/page_orchestrator.py` | Removed (unnecessary LLM call) |
| Modify | `processors/__init__.py` | Remove PageOrchestrator export |
| Modify | `main.py` | Remove PageOrchestrator, add `--llm google` |

---

## Chunk 1: Foundation (Config, Dependencies, Rate Limiter)

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Remove Tesseract/OpenCV/numpy/pandas dependencies, add google-genai and aiohttp:

```txt
# PDF Processing Libraries
PyMuPDF>=1.24.0
pdfplumber>=0.11.0
pypdfium2>=4.30.0
pdf2image>=1.17.0

# Image Processing
Pillow>=10.3.0

# AI/LLM Integration
anthropic>=0.34.0
google-genai>=1.0.0
openai>=1.40.0

# HTTP Client (for CLOVA OCR)
aiohttp>=3.9.0

# Utilities
click>=8.1.0
tqdm>=4.66.0
python-dotenv>=1.0.0
pydantic>=2.0.0
aiofiles>=23.0.0
asyncio>=3.4.3

# Logging and Monitoring
loguru>=0.7.0
rich>=13.0.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: update dependencies for production pipeline migration"
```

---

### Task 2: Update config.py

**Files:**
- Modify: `utils/config.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/model/appmodel/pdf2md.py`

- [ ] **Step 1: Rewrite config.py with 3-provider + CLOVA OCR support**

```python
"""
Configuration management for hybrid PDF to Markdown pipeline
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: str = Field(default="anthropic", pattern="^(anthropic|google|openai)$")
    anthropic_api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    google_model: str = Field(default="gemini-2.5-flash")
    openai_model: str = Field(default="gpt-5-2025-08-07")
    max_tokens: int = Field(default=16384)
    max_tokens_limit: int = Field(default=128000)
    dynamic_token_adjustment: bool = Field(default=True)
    temperature: float = Field(default=0.1)


class ClovaOCRConfig(BaseModel):
    """CLOVA OCR configuration"""
    url: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_URL", ""))
    secret_key: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_SECRET", ""))


class Config(BaseModel):
    """Main pipeline configuration"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    clova_ocr: ClovaOCRConfig = Field(default_factory=ClovaOCRConfig)
    image_dpi: int = Field(default=300, ge=150, le=900)
    output_dir: str = Field(default="output")


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get current configuration"""
    global _config
    if _config is None:
        _config = Config()
    return _config
```

- [ ] **Step 2: Commit**

```bash
git add utils/config.py
git commit -m "feat: add Google Gemini and CLOVA OCR config support"
```

---

### Task 3: Update rate_limiter.py

**Files:**
- Modify: `utils/rate_limiter.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/utils/rate_limiter.py`

- [ ] **Step 1: Add Google rate limiter routing**

The production rate_limiter.py already has Google routing via the `general` limiter (3 req/s) in the `else` branch of `get_limiter()`. The current pdf2md rate_limiter.py is identical to production. Add explicit `google` matching for clarity:

In `get_limiter()`, add Google routing before the `else`:

```python
elif 'google' in provider_lower or 'gemini' in provider_lower:
    return self.general
```

- [ ] **Step 2: Commit**

```bash
git add utils/rate_limiter.py
git commit -m "feat: add explicit Google Gemini rate limiter routing"
```

---

## Chunk 2: Extractors

### Task 4: Create CLOVA OCR extractor

**Files:**
- Create: `extractors/clova_ocr_extractor.py`

Production reference: CLOVA OCR logic from `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/job/resume_extract/resume_quick_extract.py` lines 132-164

- [ ] **Step 1: Create clova_ocr_extractor.py**

```python
"""
CLOVA OCR based text extractor
"""

import json
import time
from typing import Optional

import aiohttp
from loguru import logger

from utils.config import ClovaOCRConfig


class ClovaOCRExtractor:
    """Extract text using CLOVA OCR API"""

    def __init__(self, config: ClovaOCRConfig):
        self.name = "CLOVA OCR"
        self.config = config

    async def extract_text(self, page_pdf_bytes: bytes, page_number: int) -> Optional[str]:
        """
        Extract text from a single PDF page using CLOVA OCR API

        Args:
            page_pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference

        Returns:
            Extracted text string, or None on failure
        """
        if not self.config.url or not self.config.secret_key:
            logger.warning("CLOVA OCR credentials not configured")
            return None

        try:
            clova_message = {
                "version": "V2",
                "requestId": f"pdf2md_page_{page_number}",
                "timestamp": int(time.time()),
                "images": [{
                    "format": "pdf",
                    "name": f"page_{page_number}"
                }]
            }

            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                page_pdf_bytes,
                filename=f'page_{page_number}.pdf',
                content_type='application/pdf'
            )
            form_data.add_field(
                'message',
                json.dumps(clova_message),
                content_type='application/json'
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=self.config.url,
                    data=form_data,
                    headers={'X-OCR-SECRET': self.config.secret_key}
                ) as response:
                    if response.status == 200:
                        clova_result = await response.json()
                        logger.info(f"✅ CLOVA OCR completed for page {page_number}")
                    else:
                        logger.error(f"❌ CLOVA OCR failed for page {page_number}: {response.status}")
                        return None

            # Parse CLOVA OCR response
            clova_text = ""
            for field in clova_result['images'][0]['fields']:
                clova_text += field['inferText'] + ('\n' if field['lineBreak'] else ' ')

            return clova_text.strip()

        except Exception as e:
            logger.error(f"CLOVA OCR extraction failed for page {page_number}: {e}")
            return None
```

- [ ] **Step 2: Commit**

```bash
git add extractors/clova_ocr_extractor.py
git commit -m "feat: add CLOVA OCR extractor"
```

---

### Task 5: Rewrite PyMuPDF extractor (hyperlink-only)

**Files:**
- Rewrite: `extractors/pymupdf_extractor.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/extractors/pymupdf_extractor.py` — `extract_hyperlinks()` method (lines 147-261)

- [ ] **Step 1: Rewrite pymupdf_extractor.py**

Replace the entire file with the production version's hyperlink-only extractor. Remove `extract_text()`, `extract_structure()`, `_process_blocks()`, `_extract_images()`. Keep only `extract_hyperlinks()` with the 6-step fallback link type detection:

```python
"""
PyMuPDF (fitz) based PDF hyperlink extractor
"""

import io
from typing import Any, Dict

import fitz  # PyMuPDF
from loguru import logger


class PyMuPDFExtractor:
    """Extract hyperlinks using PyMuPDF"""

    def __init__(self):
        self.name = "PyMuPDF"

    def extract_hyperlinks(self, pdf_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """
        Extract hyperlinks from a single PDF page

        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference

        Returns:
            Dictionary containing list of hyperlinks with URLs and positions
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            if doc.page_count == 0:
                return {'hyperlinks': []}

            page = doc[0]
            links = page.get_links()

            hyperlinks = []
            for link in links:
                link_rect = link.get('from')
                link_text = ""
                link_type = "text"

                if link_rect:
                    # 1. 링크 영역 내 텍스트 검색
                    text_dict = page.get_text("dict")
                    text_found = False

                    for block in text_dict.get('blocks', []):
                        if block.get('type') == 0:
                            for line in block.get('lines', []):
                                for span in line.get('spans', []):
                                    span_rect = fitz.Rect(span.get('bbox', []))
                                    if span_rect.intersects(link_rect):
                                        link_text += span.get('text', '')
                                        text_found = True

                    if not text_found:
                        # 2. 이미지 확인
                        image_list = page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                                if img_rect and img_rect.intersects(link_rect):
                                    link_type = "image"
                                    link_text = f"[Image Link {img_index + 1}]"
                                    break
                            except:
                                continue

                        # 3. 폼 필드 확인
                        if link_type == "text":
                            widgets = page.widgets()
                            for widget in widgets:
                                widget_rect = widget.rect
                                if widget_rect and widget_rect.intersects(link_rect):
                                    field_type = widget.field_type_string
                                    field_name = widget.field_name or f"Field_{len(widgets)}"
                                    link_type = "form_field"
                                    link_text = f"[{field_type} Field: {field_name}]"
                                    break

                        # 4. 주석 확인
                        if link_type == "text":
                            annotations = page.annots()
                            for annot in annotations:
                                annot_rect = annot.rect
                                if annot_rect and annot_rect.intersects(link_rect):
                                    annot_type = annot.type[1] if annot.type else "Unknown"
                                    link_type = "annotation"
                                    link_text = f"[{annot_type} Annotation]"
                                    break

                        # 5. 도형 확인
                        if link_type == "text":
                            drawings = page.get_drawings()
                            for i, drawing in enumerate(drawings):
                                if hasattr(drawing, 'rect') and drawing.rect and drawing.rect.intersects(link_rect):
                                    link_type = "drawing"
                                    link_text = f"[Drawing Element {i + 1}]"
                                    break

                        # 6. 클릭 가능 영역
                        if link_type == "text":
                            link_type = "area"
                            link_text = "[Clickable Area]"

                if link.get('uri'):  # External URL only
                    hyperlinks.append({
                        'url': link['uri'],
                        'text': link_text.strip(),
                        'link_type': link_type,
                        'rect': list(link.get('from', [])),
                        'page_number': page_number,
                        'type': 'external'
                    })

            doc.close()
            return {'hyperlinks': hyperlinks}

        except Exception as e:
            logger.error(f"PyMuPDF hyperlink extraction failed: {e}")
            return {'hyperlinks': []}
```

- [ ] **Step 2: Commit**

```bash
git add extractors/pymupdf_extractor.py
git commit -m "refactor: rewrite PyMuPDF extractor as hyperlink-only"
```

---

### Task 6: Rewrite LLM extractor (3-provider, image-only)

**Files:**
- Rewrite: `extractors/llm_extractor.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/extractors/llm_extractor.py`

- [ ] **Step 1: Rewrite llm_extractor.py**

Port production version directly. Key changes from current:
- Remove `extract_single_page_pdf()` and `_call_claude_pdf()`
- Add Google Gemini via `google-genai` SDK (`_call_google_image()`)
- Route via `_call_llm_image()` which checks `config.llm.provider`
- Keep `_resize_image_if_needed()`, `extract_single_page_image()`, `extract_from_images()`

```python
"""
LLM (Claude/Gemini/GPT) based multimodal image text extractor
"""

import base64
import io
import json
from typing import Any, Dict, List

from PIL import Image
from anthropic import Anthropic
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from loguru import logger
from openai import OpenAI

from prompts import SINGLE_PAGE_IMAGE_PROMPT
from utils.config import Config


class LLMExtractor:
    """Extract text and structure using LLM multimodal capabilities"""

    def __init__(self, config: Config):
        """Initialize LLM extractor with client based on configuration"""
        self.name: str = "LLM"
        self.config = config

        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.google_client = genai.Client(api_key=self.config.llm.google_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)

    def _resize_image_if_needed(self, image_bytes: bytes, max_dimension: int = 7999) -> bytes:
        """Resize image if it exceeds the maximum dimension (default 7999px for Claude API)"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True)
                return output.getvalue()

            return image_bytes

        except Exception as e:
            logger.warning(f"Failed to resize image: {e}. Using original image.")
            return image_bytes

    def _call_llm_image(self, img_base64: str, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Call LLM API for image analysis"""
        if self.config.llm.provider == "anthropic":
            return self._call_claude_image(img_base64, prompt)
        elif self.config.llm.provider == "google":
            return self._call_google_image(image_bytes, prompt)
        else:
            return self._call_openai_image(img_base64, prompt)

    def _call_claude_image(self, img_base64: str, prompt: str) -> Dict[str, Any]:
        """Call Claude API for image analysis"""
        try:
            message = self.anthropic_client.messages.create(
                model=self.config.llm.claude_model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            response_text = message.content[0].text if message.content else ""
            result = self._parse_llm_response(response_text)
            result['llm_model'] = self.config.llm.claude_model
            result['llm_provider'] = 'Claude (Anthropic)'
            return result

        except Exception as e:
            logger.error(f"Claude Image API call failed: {e}")
            return {'text': '', 'error': str(e)}

    def _call_google_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Call Google API for image analysis"""
        try:
            message = self.google_client.models.generate_content(
                model=self.config.llm.google_model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                ],
                config=GenerateContentConfig(
                    system_instruction=prompt,
                    max_output_tokens=self.config.llm.max_tokens,
                    temperature=self.config.llm.temperature,
                )
            )

            response_text = message.text if message.text else ""
            result = self._parse_llm_response(response_text)
            result['llm_model'] = self.config.llm.google_model
            result['llm_provider'] = 'Gemini (Google)'
            return result

        except Exception as e:
            logger.error(f"Google Image API call failed: {e}")
            return {'text': '', 'error': str(e)}

    def _call_openai_image(self, img_base64: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API for image analysis"""
        try:
            completion_params = {
                "model": self.config.llm.openai_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }]
            }

            if "gpt-5" in self.config.llm.openai_model.lower():
                completion_params["max_completion_tokens"] = self.config.llm.max_tokens
            else:
                completion_params["max_tokens"] = self.config.llm.max_tokens
                completion_params["temperature"] = self.config.llm.temperature

            response = self.openai_client.chat.completions.create(**completion_params)

            response_text = response.choices[0].message.content if response.choices else ""
            result = self._parse_llm_response(response_text)
            result['llm_model'] = self.config.llm.openai_model
            result['llm_provider'] = 'OpenAI'
            return result

        except Exception as e:
            logger.error(f"OpenAI Image API call failed: {e}")
            return {'text': '', 'error': str(e)}

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            else:
                return {
                    'text': response_text,
                    'structure': {}
                }
        except json.JSONDecodeError:
            return {
                'text': response_text,
                'structure': {}
            }

    def extract_single_page_image(self, image_bytes: bytes, page_number: int, total_pages: int) -> Dict[str, Any]:
        """Extract text from a single page image using LLM"""
        try:
            resized_image_bytes = self._resize_image_if_needed(image_bytes)
            img_base64 = base64.b64encode(resized_image_bytes).decode('utf-8')

            prompt = SINGLE_PAGE_IMAGE_PROMPT.format(
                page_number=page_number,
                total_pages=total_pages
            )

            result = self._call_llm_image(img_base64, image_bytes, prompt)
            result['page_number'] = page_number

            return result

        except Exception as e:
            logger.error(f"LLM single page image extraction failed for page {page_number}: {e}")
            return {'text': '', 'error': str(e), 'page_number': page_number}

    def extract_from_images(self, images: List[bytes]) -> Dict[str, Any]:
        """Legacy method: Extract text from multiple images using LLM"""
        if not images:
            return {'text': '', 'error': 'No images provided'}

        try:
            if len(images) == 1:
                return self.extract_single_page_image(images[0], page_number=1, total_pages=1)

            all_text = []
            combined_results = {
                'text': '',
                'structure': {},
                'extraction_mode': 'multipage_images',
                'page_count': len(images)
            }

            for i, image_bytes in enumerate(images, 1):
                page_result = self.extract_single_page_image(image_bytes, page_number=i, total_pages=len(images))
                if page_result.get('text'):
                    all_text.append(f"=== Page {i} ===\n{page_result['text']}")

            combined_results['text'] = '\n\n'.join(all_text)
            combined_results['total_chars'] = len(combined_results['text'])

            return combined_results

        except Exception as e:
            logger.error(f"LLM images extraction failed: {e}")
            return {'text': '', 'error': str(e)}
```

- [ ] **Step 2: Commit**

```bash
git add extractors/llm_extractor.py
git commit -m "feat: rewrite LLM extractor with 3-provider support, remove PDF extraction"
```

---

### Task 7: Delete Tesseract extractor

**Files:**
- Delete: `extractors/tesseract_extractor.py`

- [ ] **Step 1: Delete the file**

```bash
git rm extractors/tesseract_extractor.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: remove Tesseract extractor (replaced by CLOVA OCR)"
```

---

## Chunk 3: Prompts and Processors

### Task 8: Rewrite prompts.py

**Files:**
- Rewrite: `prompts.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/prompts.py`

- [ ] **Step 1: Rewrite prompts.py with production version**

Port the production prompts.py directly. Key changes:
- Remove `PDF_EXTRACTION_PROMPT`, `SINGLE_PAGE_PDF_PROMPT`, `get_page_integration_prompt()`
- Replace `get_llm_merge_prompt()` with production's 4-extractor version
- Replace `format_extraction_data()` with production's hyperlink metadata version
- Keep `IMAGE_EXTRACTION_PROMPT`, `SINGLE_PAGE_IMAGE_PROMPT`, `get_final_document_prompt()` (identical)

Write the exact content from the production file at `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/prompts.py` — all 244 lines, adjusting only import paths.

- [ ] **Step 2: Commit**

```bash
git add prompts.py
git commit -m "feat: port production prompts with 4-extractor merge strategy"
```

---

### Task 9: Rewrite LLM merger

**Files:**
- Rewrite: `processors/llm_merger.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/processors/llm_merger.py`

- [ ] **Step 1: Rewrite llm_merger.py with production version**

Port the production LLMMerger directly. Key changes from current:
- Add Google Gemini client + `_call_google_merge()` via `google-genai`
- 3-way provider routing in `_call_llm_for_merge()`
- Google merge uses `max_output_tokens=20000`, temperature=0.1
- Import paths adjusted for pdf2md structure

```python
"""
LLM-based text merger for combining extraction results
"""

import asyncio
from typing import Any, Dict, List

from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from loguru import logger
from openai import OpenAI

from prompts import (
    format_extraction_data,
    get_llm_merge_prompt,
)
from utils.config import Config
from utils.rate_limiter import APIRateLimiters


class LLMMerger:
    """Text merger for combining multiple extraction results"""

    def __init__(self, config: Config):
        """Initialize LLM merger with all configurations"""
        self.config: Config = config
        self.rate_limiters: APIRateLimiters = APIRateLimiters()

        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.google_client = genai.Client(api_key=self.config.llm.google_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)

        self.provider: str = self.config.llm.provider
        if self.provider == "anthropic":
            self.model: str = self.config.llm.claude_model
            logger.info(f"LLM Merger preferring Claude: {self.model}")
        elif self.provider == "google":
            self.model: str = self.config.llm.google_model
            logger.info(f"LLM Merger preferring Google: {self.model}")
        else:
            self.model: str = self.config.llm.openai_model
            logger.info(f"LLM Merger preferring OpenAI: {self.model}")

    async def merge_text(self, extraction_results: Dict[str, Dict[str, Any]]) -> str:
        """Merge text from multiple extractors using LLM intelligence"""
        if not extraction_results:
            return ''

        valid_results = self._filter_valid_results(extraction_results)

        if not valid_results:
            logger.warning("No valid extraction results to merge")
            return ''

        if len(valid_results) == 1:
            result = next(iter(valid_results.values()))
            return result.get('text', '')

        extraction_data = format_extraction_data(valid_results)
        prompt = get_llm_merge_prompt(extraction_data)
        merged_text = await self._call_llm_for_merge(prompt)

        return merged_text

    def extract_metadata(self, extraction_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and combine metadata from extraction results"""
        valid_results = self._filter_valid_results(extraction_results)

        metadata = {
            'extractors_used': list(valid_results.keys()),
            'extraction_details': {}
        }

        for name, result in valid_results.items():
            if result.get('metadata'):
                metadata['extraction_details'][name] = result['metadata']

        return metadata

    def get_valid_sources(self, extraction_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of valid extraction sources"""
        valid_results = self._filter_valid_results(extraction_results)
        return list(valid_results.keys())

    def _filter_valid_results(self, extraction_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter out empty or error results"""
        valid_results = {}
        for name, result in extraction_results.items():
            if result.get('text') and not result.get('error'):
                valid_results[name] = result
        return valid_results

    async def _call_llm_for_merge(self, prompt: str) -> str:
        """Call LLM API with rate limiting for merging"""
        limiter = self.rate_limiters.get_limiter(self.provider)

        try:
            await limiter.acquire()

            if self.provider == "anthropic":
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            elif self.provider == "google":
                response = await asyncio.to_thread(
                    self.google_client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        max_output_tokens=20000,
                        temperature=0.1
                    )
                )
                return response.text

            elif self.provider == "openai":
                completion_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                if "gpt-5" not in self.model.lower():
                    completion_params["temperature"] = 0.1

                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    **completion_params
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return self._fallback_merge(prompt)

    def _fallback_merge(self, prompt: str) -> str:
        """Fallback merge when LLM is unavailable"""
        lines = prompt.split('\n')
        in_extraction = False
        texts = []

        for line in lines:
            if '=== ' in line and 'EXTRACTION ===' in line:
                in_extraction = True
                continue
            elif line.startswith('IMPORTANT INSTRUCTIONS:'):
                break
            elif in_extraction and line.strip():
                texts.append(line)

        return '\n'.join(texts)
```

- [ ] **Step 2: Commit**

```bash
git add processors/llm_merger.py
git commit -m "feat: rewrite LLM merger with 3-provider support and production merge strategy"
```

---

### Task 10: Rewrite final_orchestrator.py

**Files:**
- Rewrite: `processors/final_orchestrator.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/processors/final_orchestrator.py`

- [ ] **Step 1: Rewrite final_orchestrator.py with production version**

Port the production FinalOrchestrator directly. Key changes from current:
- Add Google Gemini client + `_call_google()` method
- 3-way provider routing in `_call_llm_for_final_generation()`
- Import paths adjusted for pdf2md structure
- All other logic (dynamic tokens, fallback, post-process) is identical

Write the exact content from the production file, adjusting only:
- `from controller.pdf2md.prompts import` → `from prompts import`
- `from model.appmodel.pdf2md import PipelineConfig` → `from utils.config import Config`
- All `PipelineConfig` references → `Config`

- [ ] **Step 2: Commit**

```bash
git add processors/final_orchestrator.py
git commit -m "feat: rewrite final orchestrator with 3-provider support"
```

---

### Task 11: Delete PageOrchestrator

**Files:**
- Delete: `processors/page_orchestrator.py`
- Modify: `processors/__init__.py`

- [ ] **Step 1: Delete page_orchestrator.py**

```bash
git rm processors/page_orchestrator.py
```

- [ ] **Step 2: Update processors/__init__.py**

Remove `PageOrchestrator` from exports. Keep `ImageConverter`, `LLMMerger`, `SinglePagePipeline`, `FinalOrchestrator`.

- [ ] **Step 3: Commit**

```bash
git add processors/page_orchestrator.py processors/__init__.py
git commit -m "refactor: remove PageOrchestrator (unnecessary LLM call)"
```

---

### Task 12: Rewrite SinglePagePipeline

**Files:**
- Rewrite: `processors/single_page_pipeline.py`

Production reference: `/Users/iml/Documents/GitHub/leslie-queue-worker/leslie-queue-worker/controller/pdf2md/processors/single_page_pipeline.py`

- [ ] **Step 1: Rewrite single_page_pipeline.py**

Port the production SinglePagePipeline, adapting for pdf2md:
- Replace Tesseract with CLOVA OCR in Phase 1
- Remove `_run_tesseract()`, `_run_llm_pdf()`
- Add `_run_clova_ocr()` (async, no `asyncio.to_thread` needed)
- Phase 1: PDFPlumber + PyMuPDF(hyperlinks) + CLOVA OCR (all parallel)
- Phase 2: LLM Image (rate-limited)
- PyMuPDF now calls `extract_hyperlinks()` only
- Import paths adjusted

```python
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

            # Step 2: Run extractors in two phases
            # Phase 1: Fast extractors - run fully in parallel
            fast_extraction_tasks = [
                ('pymupdf', self._run_pymupdf(page_pdf_bytes)),
                ('pdfplumber', self._run_pdfplumber(page_pdf_bytes)),
                ('clova_ocr', self._run_clova_ocr(page_pdf_bytes))
            ]
            # Phase 2: LLM image extractor - rate-limited
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
```

- [ ] **Step 2: Commit**

```bash
git add processors/single_page_pipeline.py
git commit -m "feat: rewrite SinglePagePipeline with CLOVA OCR and 4-extractor architecture"
```

---

## Chunk 4: Main Pipeline and Integration

### Task 13: Update main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update main.py**

Key changes:
- Remove `PageOrchestrator` import and usage
- Pass `config` to `SinglePagePipeline` constructor
- Add `--llm google` option
- `process_single_page()` no longer calls `page_orchestrator.integrate_page_results()`
- SinglePagePipeline returns result directly (no PageOrchestrator wrapper)

In `__init__`:
```python
# Remove: self.page_orchestrator = PageOrchestrator(self.config)
# Keep: self.final_orchestrator = FinalOrchestrator(self.config)
```

In `process_single_page()`:
```python
async def process_single_page(self, page_pdf_bytes, page_number, total_pages):
    try:
        single_page_pipeline = SinglePagePipeline(page_number, total_pages, self.config)
        result = await single_page_pipeline.process_page(page_pdf_bytes)
        return result
    except Exception as e:
        logger.error(f"Failed to process page {page_number}: {e}")
        return {
            'page_number': page_number,
            'content': '',
            'error': str(e)
        }
```

In `main()` argparse:
```python
parser.add_argument(
    '--llm',
    type=str,
    choices=['openai', 'anthropic', 'google'],
    default='anthropic',
    help='LLM provider to use'
)
```

Remove imports:
```python
# Remove: from processors.page_orchestrator import PageOrchestrator
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: update main pipeline with 4-extractor architecture and Google provider"
```

---

### Task 14: Verify and clean up

- [ ] **Step 1: Verify all imports resolve**

```bash
cd /Users/iml/Documents/GitHub/pdf2md && python -c "
from utils.config import get_config, Config
from utils.rate_limiter import APIRateLimiters
from extractors.pymupdf_extractor import PyMuPDFExtractor
from extractors.pdfplumber_extractor import PDFPlumberExtractor
from extractors.clova_ocr_extractor import ClovaOCRExtractor
from extractors.llm_extractor import LLMExtractor
from processors.image_converter import ImageConverter
from processors.llm_merger import LLMMerger
from processors.final_orchestrator import FinalOrchestrator
from processors.single_page_pipeline import SinglePagePipeline
from prompts import (
    IMAGE_EXTRACTION_PROMPT,
    SINGLE_PAGE_IMAGE_PROMPT,
    get_llm_merge_prompt,
    get_final_document_prompt,
    format_extraction_data,
)
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 2: Verify CLI help works**

```bash
python main.py --help
```

Expected: Shows `--llm {openai,anthropic,google}` option

- [ ] **Step 3: Verify deleted files don't exist**

```bash
test ! -f extractors/tesseract_extractor.py && echo "tesseract deleted OK"
test ! -f processors/page_orchestrator.py && echo "page_orchestrator deleted OK"
```

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git status
# Only commit if there are changes
git diff --cached --quiet || git commit -m "chore: final cleanup after pipeline migration"
```
