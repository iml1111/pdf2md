# Usecase Layer Refactoring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate pipeline core logic into usecase functions with dataclass I/O, preparing for Celery-based async worker migration.

**Architecture:** Insert a `usecases/` layer between `main.py` and `extractors/`+`processors/`. Each usecase function is a thin wrapper that handles config injection, error handling, and dataclass I/O conversion, calling existing extractors/processors for core logic. `main.py` chains usecase functions procedurally.

**Tech Stack:** Python 3.11, asyncio, dataclasses, PyMuPDF (fitz), pdfplumber, aiohttp, Anthropic SDK, OpenAI SDK

**Spec:** `docs/superpowers/specs/2026-03-13-usecase-refactoring-design.md`

**Known Tradeoffs:**
- LLM 클라이언트(`Anthropic`, `OpenAI`)가 페이지마다 새로 생성됨. 스펙에서 Celery 전환을 위해 각 유스케이스 함수를 self-contained하게 설계하기로 결정한 트레이드오프. 성능 최적화는 Celery 전환 시 워커 모듈 레벨 싱글턴으로 해결.
- `total_processing_time` 메타데이터가 항상 0 (기존 코드의 pre-existing bug — `processing_time`이 top-level 키인데 `metadata` 하위에서 찾음). 이 리팩토링에서는 기존 동작을 유지하고, 별도 이슈로 수정.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `usecases/__init__.py` | Package init |
| `usecases/models.py` | Dataclass definitions: PageInput, ExtractionResult, MergeInput, MergeResult, FinalizeInput, FinalizeResult |
| `usecases/extraction.py` | 4 extraction usecase functions (pdfplumber, clova_ocr, llm_image, hyperlinks) |
| `usecases/merging.py` | Page merge usecase function |
| `usecases/finalizing.py` | Document finalize usecase function |

### Modified Files
| File | Change |
|------|--------|
| `processors/image_converter.py` | Add `convert_page_to_image(pdf_bytes, dpi)` method (from SinglePagePipeline._convert_page_to_image) |
| `processors/llm_merger.py` | Convert LLMMerger class to module-level functions. Accept clients + rate_limiters as params |
| `processors/final_orchestrator.py` | Convert FinalOrchestrator class to module-level functions. Accept clients + config as params |
| `main.py` | Remove PDF2MDPipeline class. Replace with procedural main() + run_pipeline() + utility functions |

### Deleted Files
| File | Reason |
|------|--------|
| `processors/single_page_pipeline.py` | Orchestration moves to usecases/extraction.py. Image conversion moves to image_converter.py |

### Unchanged Files
`extractors/*.py`, `prompts.py`, `utils/*.py` — all unchanged.

---

## Chunk 1: Foundation (Models + ImageConverter)

### Task 1: Create usecases/models.py

**Files:**
- Create: `usecases/__init__.py`
- Create: `usecases/models.py`

- [ ] **Step 1: Create usecases package**

```python
# usecases/__init__.py
```

- [ ] **Step 2: Create dataclass models**

```python
# usecases/models.py
"""Dataclass models for usecase function I/O"""

from dataclasses import dataclass, field


@dataclass
class PageInput:
    """PDF 단일 페이지 입력"""
    page_bytes: bytes
    page_number: int
    total_pages: int


@dataclass
class ExtractionResult:
    """추출기 단일 결과"""
    extractor_name: str      # "pdfplumber" | "clova_ocr" | "llm_img" | "pymupdf"
    text: str
    tables: list[dict] | None = None
    hyperlinks: list[dict] | None = None
    metadata: dict | None = None
    error: str | None = None


@dataclass
class MergeInput:
    """병합 입력"""
    page_number: int
    extraction_results: list[ExtractionResult] = field(default_factory=list)


@dataclass
class MergeResult:
    """병합 결과"""
    page_number: int
    merged_text: str
    error: str | None = None


@dataclass
class FinalizeInput:
    """최종 문서 생성 입력"""
    merge_results: list[MergeResult] = field(default_factory=list)
    total_pages: int = 0
    source_file: str = ""


@dataclass
class FinalizeResult:
    """최종 문서 생성 결과"""
    markdown: str
    metadata: dict = field(default_factory=dict)
    error: str | None = None
```

- [ ] **Step 3: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from usecases.models import PageInput, ExtractionResult, MergeInput, MergeResult, FinalizeInput, FinalizeResult; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add usecases/__init__.py usecases/models.py
git commit -m "feat: add usecase dataclass models"
```

---

### Task 2: Add convert_page_to_image to ImageConverter

**Files:**
- Modify: `processors/image_converter.py`

This extracts the `_convert_page_to_image` logic from `SinglePagePipeline` (lines 126-148 of `processors/single_page_pipeline.py`) into `ImageConverter`.

- [ ] **Step 1: Add convert_page_to_image method**

Add the following method to the `ImageConverter` class in `processors/image_converter.py`, after the existing `optimize_for_ocr` method (after line 63):

```python
    def convert_page_to_image(self, page_pdf_bytes: bytes) -> bytes:
        """
        Convert single page PDF bytes to PNG image bytes

        Args:
            page_pdf_bytes: PDF bytes containing single page

        Returns:
            PNG image bytes
        """
        try:
            pdf_stream = io.BytesIO(page_pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            if doc.page_count > 0:
                page = doc[0]
                mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pix = None
                doc.close()
                return img_data

            doc.close()
            raise ValueError("No pages in PDF")

        except Exception as e:
            logger.error(f"Failed to convert page to image: {e}")
            raise
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from processors.image_converter import ImageConverter; ic = ImageConverter(300); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add processors/image_converter.py
git commit -m "feat: add convert_page_to_image to ImageConverter"
```

---

## Chunk 2: Processor Refactoring (Class → Functions)

### Task 3: Refactor processors/llm_merger.py to functions

**Files:**
- Modify: `processors/llm_merger.py`

Convert the `LLMMerger` class to module-level functions. The key change: clients and rate_limiters are received as parameters instead of created internally.

- [ ] **Step 1: Rewrite llm_merger.py as functions**

Replace the entire file content with:

```python
"""
LLM-based text merger for combining extraction results
"""

import asyncio
from typing import Any, Dict

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from prompts import (
    format_extraction_data,
    get_llm_merge_prompt,
)
from utils.config import LLMConfig
from utils.rate_limiter import RateLimiter


def filter_valid_results(
    extraction_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filter out empty or error results (text-based only)"""
    valid = {}
    for name, result in extraction_results.items():
        if result.get('text') and not result.get('error'):
            valid[name] = result
    return valid


async def call_llm_for_merge(
    prompt: str,
    config: LLMConfig,
    rate_limiter: RateLimiter,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Call LLM API with rate limiting for merging"""
    try:
        await rate_limiter.acquire()

        if config.provider == "anthropic":
            response = await asyncio.to_thread(
                anthropic_client.messages.create,
                model=config.claude_model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        elif config.provider == "openai":
            completion_params = {
                "model": config.openai_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if "gpt-5" in config.openai_model.lower():
                completion_params["reasoning_effort"] = "high"
            else:
                completion_params["temperature"] = 0.1

            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                **completion_params,
            )
            return response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM merge failed: {e}")
        return fallback_merge(prompt)


def fallback_merge(prompt: str) -> str:
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


async def merge_text(
    extraction_results: Dict[str, Dict[str, Any]],
    config: LLMConfig,
    rate_limiter: RateLimiter,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Merge text from multiple extractors using LLM intelligence"""
    if not extraction_results:
        return ''

    valid_results = filter_valid_results(extraction_results)

    if not valid_results and 'pymupdf' not in extraction_results:
        logger.warning("No valid extraction results to merge")
        return ''

    if len(valid_results) == 1 and 'pymupdf' not in extraction_results:
        result = next(iter(valid_results.values()))
        return result.get('text', '')

    all_results_for_format = dict(valid_results)
    if 'pymupdf' in extraction_results:
        all_results_for_format['pymupdf'] = extraction_results['pymupdf']

    extraction_data = format_extraction_data(all_results_for_format)
    prompt = get_llm_merge_prompt(extraction_data)
    merged_text = await call_llm_for_merge(
        prompt, config, rate_limiter, anthropic_client, openai_client,
    )

    return merged_text


def extract_metadata(
    extraction_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract and combine metadata from extraction results"""
    valid_results = filter_valid_results(extraction_results)

    metadata = {
        'extractors_used': list(valid_results.keys()),
        'extraction_details': {},
    }

    for name, result in valid_results.items():
        if result.get('metadata'):
            metadata['extraction_details'][name] = result['metadata']

    return metadata


def get_valid_sources(
    extraction_results: Dict[str, Dict[str, Any]],
) -> list[str]:
    """Get list of valid extraction sources"""
    return list(filter_valid_results(extraction_results).keys())
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from processors.llm_merger import merge_text, filter_valid_results, call_llm_for_merge; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add processors/llm_merger.py
git commit -m "refactor: convert LLMMerger class to module-level functions"
```

---

### Task 4: Refactor processors/final_orchestrator.py to functions

**Files:**
- Modify: `processors/final_orchestrator.py`

Convert `FinalOrchestrator` class to module-level functions. Clients and config received as parameters.

- [ ] **Step 1: Rewrite final_orchestrator.py as functions**

Replace the entire file content with:

```python
"""
Final Orchestrator for generating complete markdown document from all pages
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from prompts import get_final_document_prompt
from utils.config import LLMConfig

SYSTEM_PROMPT = "You are a markdown formatting expert. Preserve ALL content while creating well-formatted documents."


def combine_page_contents(page_results: List[Dict[str, Any]]) -> str:
    """Combine all page contents with page markers"""
    combined = []
    for result in page_results:
        page_num = result.get('page_number', 0)
        content = result.get('content', '')
        if content.strip():
            combined.append(f"[PAGE {page_num}]")
            combined.append(content)
            combined.append("")
    return "\n".join(combined)


def generate_metadata(
    page_results: List[Dict[str, Any]],
    source_file: str,
) -> Dict[str, Any]:
    """Generate document metadata"""
    return {
        'source_file': source_file,
        'total_pages': len(page_results),
        'total_processing_time': sum(
            r.get('metadata', {}).get('processing_time', 0)
            for r in page_results
        ),
        'successful_pages': sum(
            1 for r in page_results if not r.get('error')
        ),
    }


def estimate_tokens(text: str) -> int:
    """Estimate token count for Korean/English mixed text"""
    if not text:
        return 0
    korean_chars = sum(
        1 for c in text
        if '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ'
    )
    total_chars = len(text)
    if total_chars == 0:
        return 0
    korean_ratio = korean_chars / total_chars
    estimated_tokens = (korean_ratio * 1.5 + (1 - korean_ratio) * 0.25) * total_chars
    return int(estimated_tokens * 1.2)


def calculate_dynamic_max_tokens(prompt: str, config: LLMConfig) -> int:
    """Calculate dynamic max_tokens based on input size"""
    if not config.dynamic_token_adjustment:
        return config.max_tokens
    estimated_output = estimate_tokens(prompt)
    required_tokens = max(estimated_output * 2, config.max_tokens)
    adjusted_tokens = min(required_tokens, config.max_tokens_limit)
    if adjusted_tokens > config.max_tokens:
        logger.info(
            f"📊 Dynamic token adjustment: {config.max_tokens} → {adjusted_tokens}"
        )
    return adjusted_tokens


def call_llm_for_final_generation(
    prompt: str,
    config: LLMConfig,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Call LLM for final document generation with dynamic token adjustment"""
    max_tokens = calculate_dynamic_max_tokens(prompt, config)

    if config.provider == "anthropic":
        return _call_claude(prompt, max_tokens, config, anthropic_client)
    else:
        return _call_openai(prompt, max_tokens, config, openai_client)


def _call_claude(
    prompt: str,
    max_tokens: int,
    config: LLMConfig,
    client: Anthropic,
) -> str:
    """Call Claude API for final generation"""
    try:
        message = client.messages.create(
            model=config.claude_model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "text":
                return block.text
        return ""
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        raise


def _call_openai(
    prompt: str,
    max_tokens: int,
    config: LLMConfig,
    client: OpenAI,
) -> str:
    """Call OpenAI API for final generation"""
    try:
        completion_params = {
            "model": config.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if "gpt-5" in config.openai_model.lower():
            completion_params["max_completion_tokens"] = max_tokens
            completion_params["reasoning_effort"] = "high"
        else:
            completion_params["max_tokens"] = max_tokens
            completion_params["temperature"] = 0.3

        response = client.chat.completions.create(**completion_params)
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def post_process_markdown(markdown: str) -> str:
    """Post-process the generated markdown"""
    markdown = re.sub(r'\[PAGE \d+\]', '', markdown)
    return markdown.strip()


def fallback_generation(
    page_results: List[Dict[str, Any]],
    source_file: str,
) -> str:
    """Fallback generation without LLM"""
    logger.warning("Using fallback generation without LLM formatting")
    stem = Path(source_file).stem

    lines = [f"# {stem}", ""]
    for result in page_results:
        page_num = result.get('page_number', 0)
        content = result.get('content', '')
        if content.strip():
            lines.append(f"## Page {page_num}")
            lines.append("")
            lines.append(content)
            lines.append("")

    lines.extend([
        "---",
        f"*Extracted from: {source_file}*",
        f"*Total pages: {len(page_results)}*",
    ])

    return "\n".join(lines)


def generate_final_document(
    page_results: List[Dict[str, Any]],
    source_file: str,
    config: LLMConfig,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Generate final markdown document from all page results"""
    if not page_results:
        logger.warning("No page results to process")
        return "# Empty Document\n\nNo content was extracted from the PDF."

    try:
        logger.info(f"📝 Generating final markdown from {len(page_results)} pages")

        combined_content = combine_page_contents(page_results)
        metadata = generate_metadata(page_results, source_file)
        prompt = get_final_document_prompt(metadata, combined_content)
        final_markdown = call_llm_for_final_generation(
            prompt, config, anthropic_client, openai_client,
        )
        final_markdown = post_process_markdown(final_markdown)

        logger.info(f"✅ Final markdown generated: {len(final_markdown)} characters")
        return final_markdown

    except Exception as e:
        logger.error(f"Final document generation failed: {e}")
        return fallback_generation(page_results, source_file)
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from processors.final_orchestrator import generate_final_document, combine_page_contents; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add processors/final_orchestrator.py
git commit -m "refactor: convert FinalOrchestrator class to module-level functions"
```

---

## Chunk 3: Usecase Functions

### Task 5: Create usecases/extraction.py

**Files:**
- Create: `usecases/extraction.py`

Four async extraction functions. Each wraps an existing extractor, converting PageInput → ExtractionResult.

Key references:
- `processors/single_page_pipeline.py` lines 150-239 (current extraction orchestration)
- `extractors/*.py` (called by these functions)

- [ ] **Step 1: Create extraction.py**

```python
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
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from usecases.extraction import extract_pdfplumber, extract_clova_ocr, extract_llm_image, extract_hyperlinks; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add usecases/extraction.py
git commit -m "feat: add extraction usecase functions"
```

---

### Task 6: Create usecases/merging.py

**Files:**
- Create: `usecases/merging.py`

Key reference: `processors/single_page_pipeline.py` lines 93-98 (current merge call) and `processors/llm_merger.py` (refactored in Task 3).

The critical conversion: `list[ExtractionResult]` → `Dict[str, Dict]` that `format_extraction_data()` expects. extractor_name values (`"pdfplumber"`, `"clova_ocr"`, `"llm_img"`, `"pymupdf"`) must match `prompts.py` line 211/233 checks.

- [ ] **Step 1: Create merging.py**

```python
# usecases/merging.py
"""Page merge usecase function"""

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from processors.llm_merger import merge_text
from usecases.models import ExtractionResult, MergeInput, MergeResult
from utils.config import Config
from utils.rate_limiter import APIRateLimiters


def _to_results_dict(
    extraction_results: list[ExtractionResult],
) -> dict[str, dict]:
    """Convert list[ExtractionResult] to Dict[str, Dict] for prompts.py compatibility"""
    results_dict = {}
    for r in extraction_results:
        entry = {'text': r.text}
        if r.tables:
            entry['tables'] = r.tables
        if r.hyperlinks:
            entry['hyperlinks'] = r.hyperlinks
        if r.metadata:
            entry['metadata'] = r.metadata
        if r.error:
            entry['error'] = r.error
        results_dict[r.extractor_name] = entry
    return results_dict


async def merge_page(
    merge_input: MergeInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> MergeResult:
    """4개 추출 결과를 LLM으로 병합"""
    try:
        results_dict = _to_results_dict(merge_input.extraction_results)

        anthropic_client = Anthropic(api_key=config.llm.anthropic_api_key)
        openai_client = OpenAI(api_key=config.llm.openai_api_key)
        rate_limiter = rate_limiters.get_limiter(config.llm.provider)

        merged_text = await merge_text(
            results_dict,
            config.llm,
            rate_limiter,
            anthropic_client,
            openai_client,
        )

        return MergeResult(
            page_number=merge_input.page_number,
            merged_text=merged_text,
        )
    except Exception as e:
        logger.error(
            f"Merge failed for page {merge_input.page_number}: {e}"
        )
        return MergeResult(
            page_number=merge_input.page_number,
            merged_text="",
            error=str(e),
        )
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from usecases.merging import merge_page; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add usecases/merging.py
git commit -m "feat: add merge_page usecase function"
```

---

### Task 7: Create usecases/finalizing.py

**Files:**
- Create: `usecases/finalizing.py`

Key reference: `processors/final_orchestrator.py` (refactored in Task 4). The conversion: `list[MergeResult]` → `list[dict]` with `{page_number, content}` keys that `combine_page_contents()` expects.

- [ ] **Step 1: Create finalizing.py**

```python
# usecases/finalizing.py
"""Document finalize usecase function"""

import asyncio

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from processors.final_orchestrator import (
    fallback_generation,
    generate_final_document,
)
from usecases.models import FinalizeInput, FinalizeResult
from utils.config import Config


async def finalize_document(
    finalize_input: FinalizeInput,
    config: Config,
) -> FinalizeResult:
    """병합된 페이지들을 최종 마크다운으로 생성"""
    try:
        # Convert MergeResult list to dict list for processor compatibility
        page_results = [
            {
                'page_number': mr.page_number,
                'content': mr.merged_text,
                'error': mr.error,
            }
            for mr in sorted(
                finalize_input.merge_results,
                key=lambda x: x.page_number,
            )
        ]

        anthropic_client = Anthropic(api_key=config.llm.anthropic_api_key)
        openai_client = OpenAI(api_key=config.llm.openai_api_key)

        # generate_final_document is synchronous — wrap in to_thread
        markdown = await asyncio.to_thread(
            generate_final_document,
            page_results,
            finalize_input.source_file,
            config.llm,
            anthropic_client,
            openai_client,
        )

        return FinalizeResult(
            markdown=markdown,
            metadata={
                'total_pages': finalize_input.total_pages,
                'source_file': finalize_input.source_file,
            },
        )
    except Exception as e:
        logger.error(f"Finalize failed: {e}")

        # Fallback
        page_results = [
            {
                'page_number': mr.page_number,
                'content': mr.merged_text,
            }
            for mr in finalize_input.merge_results
        ]
        fallback_md = fallback_generation(
            page_results, finalize_input.source_file,
        )
        return FinalizeResult(
            markdown=fallback_md,
            metadata={'total_pages': finalize_input.total_pages},
            error=str(e),
        )
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from usecases.finalizing import finalize_document; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add usecases/finalizing.py
git commit -m "feat: add finalize_document usecase function"
```

---

## Chunk 4: Main Rewrite + Cleanup

### Task 8: Rewrite main.py with procedural chaining

**Files:**
- Modify: `main.py`

Replace the entire `PDF2MDPipeline` class with procedural `main()` + `run_pipeline()` + utility functions. Preserve: CLI args, credential validation, PDF validation, JSON report output, exit codes.

Key references:
- Current `main.py` lines 271-348 (CLI parsing, credential validation, report output)
- Current `main.py` lines 65-98 (`_split_pdf_into_pages`, batch processing)
- Spec `main.py` chaining section

- [ ] **Step 1: Rewrite main.py**

Replace the entire file content with:

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "import py_compile; py_compile.compile('main.py', doraise=True); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "refactor: replace PDF2MDPipeline with procedural usecase chaining"
```

---

### Task 9: Delete single_page_pipeline.py

**Files:**
- Delete: `processors/single_page_pipeline.py`

This file's orchestration is now in `usecases/extraction.py` and its image conversion is in `processors/image_converter.py`.

- [ ] **Step 1: Verify no other imports**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && grep -r "single_page_pipeline\|SinglePagePipeline" --include="*.py" | grep -v "single_page_pipeline.py" | grep -v "__pycache__"`
Expected: No output (main.py no longer imports it after Task 8)

- [ ] **Step 2: Delete file**

```bash
git rm processors/single_page_pipeline.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove single_page_pipeline.py (replaced by usecases/extraction.py)"
```

---

### Task 10: Verify full pipeline runs

This is a smoke test to confirm the refactored pipeline produces output.

- [ ] **Step 1: Verify all imports resolve**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python -c "from main import main; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 2: Verify help works**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python main.py --help`
Expected: Usage help with `--in`, `--out`, `--llm` options

- [ ] **Step 3: Run with a test PDF (if available)**

Run: `cd /Users/iml/Documents/GitHub/pdf2md && python main.py --in <test.pdf> --out /tmp/test_output.md --llm anthropic`
Expected: Pipeline runs, JSON report on stdout, markdown file at `/tmp/test_output.md`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor: complete usecase layer refactoring"
```
