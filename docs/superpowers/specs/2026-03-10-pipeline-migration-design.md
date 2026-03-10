# PDF2MD Pipeline Migration Design

프로덕션(leslie-queue-worker)에서 검증된 파이프라인 구조를 pdf2md CLI에 이식하는 설계.

---

## 파이프라인 구조 변경

```
변경 전:
  Split → [PyMuPDF(텍스트) + PDFPlumber + Tesseract + LLM PDF + LLM Image]
       → LLMMerger → PageOrchestrator(LLM) → FinalOrchestrator

변경 후 (프로덕션 일치):
  Split → [PDFPlumber(텍스트+테이블+메타) + PyMuPDF(링크) + CLOVA OCR + LLM Image]
       → LLMMerger → FinalOrchestrator
```

핵심: PageOrchestrator 제거 (LLM 호출 1회 절약), PyMuPDF를 하이퍼링크 전용으로 재정의.

---

## 파일 변경 요약

| 작업 | 파일 |
|------|------|
| 삭제 | `extractors/tesseract_extractor.py` |
| 삭제 | `processors/page_orchestrator.py` |
| 신규 | `extractors/clova_ocr_extractor.py` |
| 수정 | `extractors/pymupdf_extractor.py` |
| 수정 | `extractors/pdfplumber_extractor.py` |
| 수정 | `extractors/llm_extractor.py` |
| 수정 | `processors/single_page_pipeline.py` |
| 수정 | `processors/llm_merger.py` |
| 수정 | `processors/final_orchestrator.py` |
| 수정 | `utils/config.py` |
| 수정 | `utils/rate_limiter.py` |
| 수정 | `prompts.py` |
| 수정 | `main.py` |
| 수정 | `requirements.txt` |

---

## Extractors 상세

### 삭제: tesseract_extractor.py
CLOVA OCR로 완전 교체.

### 신규: clova_ocr_extractor.py
- CLOVA OCR API에 단일 페이지 PDF bytes 직접 전송
- POST {CLOVA_OCR_URL}, Header: X-OCR-SECRET
- 응답 파싱: fields → inferText + lineBreak
- 실패 시 None 반환, 파이프라인 계속 진행
- async HTTP (aiohttp)

### 수정: pymupdf_extractor.py
텍스트 추출 제거, 하이퍼링크 전용으로 재작성:
- `extract_hyperlinks(page_pdf_bytes)` → 6단계 폴백 링크 타입 판별
  - text → image → form_field → annotation → drawing → area
- 외부 URL 링크만 수집, 내부 페이지 링크 제외
- 반환: `{"hyperlinks": [{url, text, link_type, rect, page_number, type}]}`

### 수정: pdfplumber_extractor.py
프로덕션과 일치하도록 3개 메서드:
- `extract_text()` — 네이티브 PDF 텍스트 추출
- `extract_tables()` — 테이블 → 마크다운 변환 (헤더 + 구분선 + 데이터)
- `extract_metadata()` — dimensions(width, height) + chars_sample(100자, text/fontname/size)

### 수정: llm_extractor.py
- `extract_single_page_pdf()` 삭제 (프로덕션 미사용)
- `extract_single_page_image()` 유지
- Google Gemini 프로바이더 추가 (google-genai SDK)
- 3개 프로바이더 이미지 포맷 분기:
  - Anthropic: `{"type": "image", "source": {"type": "base64", ...}}`
  - Google: `types.Part.from_bytes(...)`
  - OpenAI: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`

---

## Processors 상세

### 삭제: page_orchestrator.py
프로덕션에서 불필요한 LLM 호출로 판단하여 제거.

### 수정: single_page_pipeline.py
2-Phase 실행 구조:

```
Phase 1 (병렬, asyncio.gather):
  ├─ PDFPlumber: extract_text() + extract_tables() + extract_metadata()  [asyncio.to_thread]
  ├─ PyMuPDF: extract_hyperlinks()  [asyncio.to_thread]
  └─ CLOVA OCR: extract_text()  [async HTTP]

Phase 2 (레이트 리밋):
  └─ LLM Image: extract_single_page_image()

→ LLMMerger로 4개 소스 병합
```

### 수정: llm_merger.py
프로덕션의 적응형 병합 전략:
- 4개 소스: clova_ocr, pdfplumber, llm_img, pymupdf
- PDFPlumber 품질 평가 → 빈/단편적이면 CLOVA OCR + LLM Image 의존
- PyMuPDF 하이퍼링크 → `[text](#)` 인라인 통합
- format_extraction_data()로 PyMuPDF를 HYPERLINK METADATA 별도 섹션 처리

### 수정: final_orchestrator.py
- Google Gemini 프로바이더 추가 (3사 지원)
- temperature 0.3 고정
- 동적 토큰 조정 유지

### image_converter.py
변경 없음 (그레이스케일 + 선명도 1.2x + 대비 1.1x).

---

## Prompts 변경

### 삭제:
- `PDF_EXTRACTION_PROMPT` — 미사용
- `SINGLE_PAGE_PDF_PROMPT` — LLM PDF 추출 삭제
- `get_page_integration_prompt()` — PageOrchestrator 삭제

### 유지 (동일):
- `IMAGE_EXTRACTION_PROMPT`
- `SINGLE_PAGE_IMAGE_PROMPT`
- `get_final_document_prompt()`

### 프로덕션 버전으로 교체:
- `get_llm_merge_prompt()` — 4개 추출기 특성, 10개 병합 전략, 하이퍼링크 통합 규칙
- `format_extraction_data()` — PyMuPDF 하이퍼링크 메타데이터 섹션, CLOVA OCR 헤더

---

## Config 변경

```python
class LLMConfig:
    provider: str  # "anthropic" | "google" | "openai"
    anthropic_api_key: str   # ANTHROPIC_API_KEY
    google_api_key: str      # GOOGLE_API_KEY (신규)
    openai_api_key: str      # OPENAI_API_KEY
    claude_model: str = "claude-sonnet-4-20250514"
    google_model: str = "gemini-2.5-flash"          # 신규
    openai_model: str = "gpt-5-2025-08-07"
    max_tokens: int = 16384
    max_tokens_limit: int = 128000
    dynamic_token_adjustment: bool = True
    temperature: float = 0.1

class ClovaOCRConfig:    # 신규
    url: str               # CLOVA_OCR_URL
    secret_key: str        # CLOVA_OCR_SECRET

class Config:
    llm: LLMConfig
    clova_ocr: ClovaOCRConfig
    image_dpi: int = 300
```

## Rate Limiter
- Anthropic: 5 req/s
- OpenAI: 10 req/s
- Google: 3 req/s (신규)

## CLI 변경
- `--llm` 선택지: `anthropic | google | openai`
- PageOrchestrator import/호출 제거

## .env 추가
```
GOOGLE_API_KEY=...
CLOVA_OCR_URL=...
CLOVA_OCR_SECRET=...
```

## requirements.txt 변경
- 삭제: pytesseract, opencv-python, numpy, pandas
- 추가: google-genai, aiohttp

---

## 참조
- 프로덕션 설계 문서: leslie-queue-worker/docs/resume-extract-pipeline.md
- 프로덕션 코드: leslie-queue-worker/controller/pdf2md/
