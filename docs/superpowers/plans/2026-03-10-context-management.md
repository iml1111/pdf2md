# AI 에이전트 컨텍스트 관리 개선 구현 계획

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLAUDE.md를 경량 인덱스로 재작성하고, 상세 문서 4개를 docs/에 분리하여 AI 에이전트의 JIT 컨텍스트 로딩을 가능하게 한다.

**Architecture:** Leslie-queue-worker PR #71 패턴 적용. CLAUDE.md는 기술 스택 키워드 + 핵심 진입점 + docs/ 테이블 링크로만 구성. 상세 내용은 docs/architecture.md, docs/pipeline.md, docs/integrations.md, docs/conventions.md 4개 파일로 분리.

**Tech Stack:** Markdown documentation only (no code changes)

---

## File Structure

| 작업 | 파일 | 역할 |
|------|------|------|
| 수정 | `CLAUDE.md` | 경량 인덱스 (~20줄) |
| 신규 | `docs/architecture.md` | 디렉토리 구조, 실행 모델, 모듈 의존성 |
| 신규 | `docs/pipeline.md` | 4개 추출기 파이프라인, 병합 전략, 처리 흐름 |
| 신규 | `docs/integrations.md` | LLM 3사, CLOVA OCR, 환경변수, 설정 |
| 신규 | `docs/conventions.md` | 개발 규칙, 추출기/프로바이더 추가 절차 |

---

### Task 1: docs/architecture.md 작성

**Files:**
- Create: `docs/architecture.md`

- [ ] **Step 1: docs/architecture.md 파일 작성**

Leslie의 `docs/architecture.md` 패턴을 따라 작성. 기술 스택 테이블, 디렉토리 트리(인라인 설명), 주요 파일 경로 테이블, 실행 모델 흐름도, 모듈 의존성을 포함.

```markdown
# Architecture

pdf2md — 다중 추출 엔진과 LLM 병합을 활용하는 PDF to Markdown 변환 CLI 도구.

## 기술 스택

| 영역 | 기술 |
|------|------|
| Runtime | Python 3.11 |
| PDF 처리 | PyMuPDF (fitz), pdfplumber, pypdfium2 |
| OCR | CLOVA OCR (Naver, 외부 API) |
| LLM | Anthropic Claude, Google Gemini, OpenAI GPT |
| 이미지 | Pillow (그레이스케일, 선명도/대비 최적화) |
| 설정 | Pydantic BaseModel, python-dotenv (.env) |
| Logging | Loguru |
| Async | asyncio, aiohttp |

## 디렉토리 구조

```
pdf2md/
├── main.py                              # CLI 진입점 (argparse)
├── prompts.py                           # LLM 프롬프트 중앙 관리
├── extractors/                          # PDF 추출 엔진 (4개)
│   ├── pdfplumber_extractor.py          # 텍스트 + 테이블 + 메타데이터
│   ├── pymupdf_extractor.py            # 하이퍼링크 전용 (6단계 폴백)
│   ├── clova_ocr_extractor.py          # CLOVA OCR API (async aiohttp)
│   └── llm_extractor.py               # LLM 이미지 추출 (3사 지원)
├── processors/                          # 처리 파이프라인
│   ├── single_page_pipeline.py         # 페이지별 4개 추출기 병렬 실행 + 병합
│   ├── llm_merger.py                   # 적응형 4소스 LLM 병합
│   ├── final_orchestrator.py           # 최종 마크다운 생성
│   └── image_converter.py             # PDF → 이미지 변환 (OCR 최적화)
└── utils/                               # 유틸리티
    ├── config.py                        # Pydantic 설정 (LLMConfig, ClovaOCRConfig)
    ├── rate_limiter.py                 # API 레이트 리밋 (프로바이더별)
    ├── logger.py                       # Loguru 로깅 (파일 로테이션)
    └── validators.py                   # PDF 유효성 검증
```

## 주요 파일 경로

| 파일 | 역할 |
|------|------|
| `main.py` | CLI 진입점 — `--in`, `--out`, `--llm` 인자 처리 |
| `prompts.py` | 모든 LLM 프롬프트 및 포맷팅 함수 |
| `utils/config.py` | 환경변수 → Pydantic Config 로딩 |
| `processors/single_page_pipeline.py` | 페이지별 4개 추출기 오케스트레이션 |
| `processors/llm_merger.py` | 추출 결과 병합 (LLM 호출) |
| `processors/final_orchestrator.py` | 전체 페이지 → 최종 마크다운 생성 |

## 실행 모델

```
CLI (main.py)
  ▼
PDF2MDPipeline.process_pdf()
  ├─ _split_pdf_into_pages()          # PyMuPDF로 페이지별 PDF bytes 분할
  ├─ process_pdf_pages()              # 배치 단위 비동기 처리
  │   └─ process_single_page() × N    # asyncio.gather로 병렬
  │       └─ SinglePagePipeline       # 4개 추출기 + 병합
  └─ FinalOrchestrator                # 전체 페이지 → 최종 마크다운
      └─ LLM 호출 (temperature=0.3)
```

- 10페이지 이하: 전체 동시 처리, 초과 시 10페이지 배치
- 모든 LLM 호출은 `APIRateLimiters`로 프로바이더별 레이트 리밋 적용

## 모듈 의존성

```
main.py
  ├── processors/single_page_pipeline.py
  │     ├── extractors/pdfplumber_extractor.py
  │     ├── extractors/pymupdf_extractor.py
  │     ├── extractors/clova_ocr_extractor.py
  │     ├── extractors/llm_extractor.py
  │     ├── processors/image_converter.py
  │     └── processors/llm_merger.py
  ├── processors/final_orchestrator.py
  ├── utils/config.py
  ├── utils/rate_limiter.py
  └── prompts.py
```
```

- [ ] **Step 2: 커밋**

```bash
git add docs/architecture.md
git commit -m "docs: add architecture.md for AI agent context"
```

---

### Task 2: docs/pipeline.md 작성

**Files:**
- Create: `docs/pipeline.md`

- [ ] **Step 1: docs/pipeline.md 파일 작성**

Leslie의 `docs/resume-extract-pipeline.md` 패턴을 따라 작성. 파이프라인 흐름도, 추출기 4개 상세(입출력, 강점/한계), 적응형 병합 전략, 동적 토큰 조정, 에러 핸들링을 포함.

```markdown
# PDF to Markdown 파이프라인

PDF를 마크다운으로 변환하는 3단계 파이프라인 상세 설계.

---

## 파이프라인 개요

```
Stage 1: Extract (페이지별 병렬)
  │  PDF를 페이지별로 분할
  │  각 페이지에 4개 추출기 동시 실행
  │  페이지 이미지 생성 및 OCR 최적화
  ▼
Stage 2: Merge (페이지별)
  │  4개 소스(PDFPlumber, PyMuPDF, CLOVA OCR, LLM Image)를
  │  LLM으로 지능적 병합 → 페이지별 마크다운
  ▼
Stage 3: Finalize (1회/PDF)
  │  모든 페이지 병합 결과를 조합
  │  LLM으로 최종 마크다운 문서 생성
  ▼
  출력: 단일 마크다운 문서
```

**핵심 원리**: 각 페이지를 독립 처리한 뒤 최종 단계에서 하나의 문서로 조합. 4개 추출기가 각각의 강점으로 텍스트를 추출하고, LLM이 이를 지능적으로 병합.

---

## Stage 1: Extract

> 소스: `processors/single_page_pipeline.py`

### 1-1. PDF 페이지 분할

PyMuPDF(fitz)로 페이지 단위 바이너리로 분할:
```python
doc = fitz.open(stream=file_obj, filetype="pdf")
for page_num in range(doc.page_count):
    single_page_doc = fitz.open()
    single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    page_pdfs.append(single_page_doc.tobytes())
```

### 1-2. 4개 추출기 동시 실행

`asyncio.gather`로 4개 추출기를 병렬 실행. LLM 호출은 내부적으로 레이트 리밋 적용.

| 추출기 | 메서드 | 실행 방식 | 출력 |
|--------|--------|----------|------|
| PDFPlumber | `extract_text()` + `extract_tables()` + `extract_metadata()` | `asyncio.to_thread` | 텍스트, 테이블 마크다운, 메타데이터 |
| PyMuPDF | `extract_hyperlinks()` | `asyncio.to_thread` | 하이퍼링크 목록 |
| CLOVA OCR | `extract_text()` | native async (aiohttp) | OCR 텍스트 |
| LLM Image | `extract_single_page_image()` | `asyncio.to_thread` + 레이트 리밋 | LLM 추출 텍스트 |

### 1-3. PDFPlumber 상세

> 소스: `extractors/pdfplumber_extractor.py`

**extract_text**: `pdfplumber.open(pdf_stream).pages[0].extract_text()` — 네이티브 PDF 텍스트 레이어 직접 추출. 이미지 기반 PDF에서는 빈 문자열 반환.

**extract_tables**: `page.extract_tables()` → 마크다운 테이블 변환:
```
| Header1 | Header2 |
| --- | --- |
| Value1 | Value2 |
```
첫 행을 헤더, `---` 구분선, 나머지를 데이터로. None 셀은 빈 문자열.

**extract_metadata**: 페이지 dimensions(width, height) + 처음 100개 문자의 text/fontname/size 샘플.

### 1-4. PyMuPDF 하이퍼링크 추출 상세

> 소스: `extractors/pymupdf_extractor.py`

`page.get_links()`로 링크 목록을 가져온 뒤, 각 링크 영역의 콘텐츠 타입을 6단계 폴백으로 판별:

```
1. 링크 영역 내 텍스트 검색 (span rect 교차) → link_type="text"
2. 링크 영역 내 이미지 검색 → link_type="image"
3. 폼 필드 확인 → link_type="form_field"
4. 주석(annotation) 확인 → link_type="annotation"
5. 도형(drawing) 확인 → link_type="drawing"
6. 위 모두 없으면 → link_type="area"
```

**외부 URL 링크(`link['uri']`)만 수집**, 내부 페이지 링크 제외.

### 1-5. 페이지 이미지 생성 및 OCR 최적화

**PDF → 이미지**: PyMuPDF DPI 기반 렌더링 (기본 300 DPI).

**OCR 최적화** (`processors/image_converter.py`):
1. 그레이스케일 변환
2. 선명도 × 1.2
3. 대비 × 1.1

### 1-6. CLOVA OCR

> 소스: `extractors/clova_ocr_extractor.py`

**원본 PDF bytes를 직접 전송** (이미지 아님):

```
POST {CLOVA_OCR_URL}
Headers: X-OCR-SECRET: {secret_key}
Body (multipart): file (PDF bytes) + message (JSON)
```

응답 파싱: `fields[] → inferText + lineBreak` 조합으로 텍스트 생성.
실패 시 None 반환, 파이프라인 계속 진행.

### 1-7. LLM Image 추출

> 소스: `extractors/llm_extractor.py`

OCR 최적화된 이미지를 LLM Vision 모델에 전달. 프로바이더별 이미지 포맷:

| 프로바이더 | 포맷 |
|-----------|------|
| Anthropic | `{"type": "image", "source": {"type": "base64", ...}}` |
| Google | `types.Part.from_bytes(data=..., mime_type="image/png")` |
| OpenAI | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` |

프롬프트: `prompts.py`의 `SINGLE_PAGE_IMAGE_PROMPT` 사용.

---

## Stage 2: Merge

> 소스: `processors/llm_merger.py`, `prompts.py`

### 2-1. 4개 소스 수집

```python
extraction_results = {
    "clova_ocr": {"text": clova_text},
    "pdfplumber": {"text": pdfplumber_text, "tables": [...], ...},
    "llm_img": {"text": llm_image_text, ...},
    "pymupdf": {"hyperlinks": [...]}
}
```

### 2-2. 추출 데이터 포맷팅

`prompts.py`의 `format_extraction_data()`:

```
=== CLOVA OCR EXTRACTION (텍스트 정밀 추출) ===
[CLOVA OCR 텍스트]

=== PDFPLUMBER EXTRACTION (정적 텍스트/레이아웃 추출) ===
[PDFPlumber 텍스트]

=== LLM_IMG EXTRACTION (텍스트/레이아웃 시각적 관점 추출) ===
[LLM Image 텍스트]

=== HYPERLINK METADATA ===
Text with link: 'GitHub'
```

PyMuPDF는 텍스트 섹션이 아닌 HYPERLINK METADATA로 별도 처리.

### 2-3. 적응형 병합 전략

`prompts.py`의 `get_llm_merge_prompt()` — 10개 병합 규칙:

| 문서 유형 | 주요 소스 | 보조 소스 |
|----------|----------|----------|
| 텍스트 기반 네이티브 PDF | PDFPlumber (구조) | CLOVA OCR + LLM Image (검증) |
| 이미지 기반/스캔 PDF | CLOVA OCR + LLM Image | PDFPlumber (가능한 경우만) |
| 혼합 PDF | 4개 소스 균형 활용 | — |

하이퍼링크 통합: `[text](#)` 마크다운 형식 (URL 미보존, 텍스트 링크만 대상).

---

## Stage 3: Finalize

> 소스: `processors/final_orchestrator.py`

### 3-1. 페이지 결합

모든 페이지를 `[PAGE N]` 마커와 함께 결합.

### 3-2. 최종 마크다운 생성

`prompts.py`의 `get_final_document_prompt()` 사용. temperature=0.3.

### 3-3. 동적 토큰 조정

`dynamic_token_adjustment` 활성 시:
```
1. 한국어 비율 계산: korean_chars / total_chars
2. 토큰 추정: (korean_ratio × 1.5 + (1 - korean_ratio) × 0.25) × text_length × 1.2
3. 필요 토큰: max(추정값 × 2, 기본 max_tokens)
4. 상한: min(필요 토큰, max_tokens_limit)
```

### 3-4. 후처리

- `[PAGE N]` 마커 제거
- 양쪽 공백 strip

### 3-5. 폴백 (LLM 실패 시)

LLM 없이 단순 페이지별 연결로 마크다운 생성.

---

## 에러 핸들링

| 상황 | 처리 |
|------|------|
| CLOVA OCR 실패 | None 반환, 나머지 3개로 계속 |
| PDFPlumber/PyMuPDF 예외 | 빈 결과 반환, 다른 추출기가 보완 |
| LLM API 실패 (Stage 1, 2) | 예외 전파, 상위에서 처리 |
| FinalOrchestrator 실패 | 폴백: LLM 없이 단순 연결 |
| 이미지 최적화 실패 | 원본 이미지로 폴백 |

---

## 소스 파일 참조

| 파일 | 역할 |
|------|------|
| `processors/single_page_pipeline.py` | Stage 1: 페이지별 추출 오케스트레이션 |
| `processors/llm_merger.py` | Stage 2: 4소스 LLM 병합 |
| `processors/final_orchestrator.py` | Stage 3: 최종 마크다운 생성 |
| `processors/image_converter.py` | 이미지 변환 및 OCR 최적화 |
| `extractors/pdfplumber_extractor.py` | PDFPlumber 추출기 |
| `extractors/pymupdf_extractor.py` | PyMuPDF 하이퍼링크 추출기 |
| `extractors/clova_ocr_extractor.py` | CLOVA OCR 추출기 |
| `extractors/llm_extractor.py` | LLM 이미지 추출기 |
| `prompts.py` | 모든 프롬프트 및 포맷팅 함수 |
```

- [ ] **Step 2: 커밋**

```bash
git add docs/pipeline.md
git commit -m "docs: add pipeline.md for AI agent context"
```

---

### Task 3: docs/integrations.md 작성

**Files:**
- Create: `docs/integrations.md`

- [ ] **Step 1: docs/integrations.md 파일 작성**

Leslie의 `docs/integrations.md` 패턴을 따라 작성. LLM 프로바이더 3사 상세, CLOVA OCR API, 환경변수, Config 클래스 구조를 포함.

```markdown
# 외부 서비스 연동

pdf2md가 연동하는 외부 서비스 목록과 구성 방식.

## 1. LLM 프로바이더

3개 프로바이더를 동일 인터페이스로 지원. `utils/config.py`의 `LLMConfig.provider`로 선택.

| 프로바이더 | 모델 | SDK | 레이트 리밋 |
|-----------|------|-----|-----------|
| Anthropic | claude-sonnet-4-20250514 | `anthropic` | 5 req/s |
| Google | gemini-2.5-flash | `google-genai` | 3 req/s |
| OpenAI | gpt-5-2025-08-07 | `openai` | 10 req/s |

**사용 위치:**

| 모듈 | 파일 | LLM 호출 목적 |
|------|------|-------------|
| LLM 이미지 추출 | `extractors/llm_extractor.py` | 페이지 이미지 → 텍스트 추출 |
| LLM 병합 | `processors/llm_merger.py` | 4개 소스 지능적 병합 |
| 최종 생성 | `processors/final_orchestrator.py` | 전체 페이지 → 마크다운 |

**프로바이더별 차이점:**

| 항목 | Anthropic | Google | OpenAI |
|------|----------|--------|--------|
| 이미지 전달 | base64 image block | `types.Part.from_bytes()` | data URL |
| 토큰 파라미터 | `max_tokens` | `max_output_tokens` | `max_tokens` / `max_completion_tokens` (모델별) |
| System prompt | `system` 파라미터 | `system_instruction` / config | `messages[0].role="system"` |
| Temperature | 항상 적용 | 항상 적용 | gpt-5는 미지원 |

**Settings:** `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`

### 프롬프트 관리

모든 프롬프트는 `prompts.py`에서 중앙 관리:
- `SINGLE_PAGE_IMAGE_PROMPT` — LLM 이미지 추출용
- `get_llm_merge_prompt()` — 4소스 병합용
- `get_final_document_prompt()` — 최종 마크다운 생성용
- `format_extraction_data()` — 추출 결과 포맷팅

## 2. CLOVA OCR (Naver)

> 소스: `extractors/clova_ocr_extractor.py`

**API 프로토콜:**
```
POST {CLOVA_OCR_URL}
Headers: X-OCR-SECRET: {secret_key}
Body (multipart):
  - file: 단일 페이지 PDF bytes
  - message: JSON {"version": "V2", "requestId": "...", "timestamp": ..., "images": [{"format": "pdf", "name": "..."}]}
```

**응답 파싱:** `images[0].fields[] → inferText + lineBreak` 조합.

**실패 처리:** status != 200이면 None 반환, 다른 추출기로 계속.

**Settings:** `CLOVA_OCR_URL`, `CLOVA_OCR_SECRET`

## 3. 레이트 리밋

> 소스: `utils/rate_limiter.py`

`APIRateLimiters` 클래스가 프로바이더별 제한 관리:

| 프로바이더 | 제한 | 라우팅 키워드 |
|-----------|------|-------------|
| Anthropic | 5 req/s | `anthropic`, `claude` |
| OpenAI | 10 req/s | `openai`, `gpt` |
| Google/기타 | 3 req/s | `google`, `gemini`, 기타 |

`asyncio.Lock` + `asyncio.Semaphore`로 동시성 제어.

## 환경변수 (.env)

| 변수 | 필수 | 용도 |
|------|------|------|
| `ANTHROPIC_API_KEY` | provider=anthropic 시 | Anthropic Claude API 키 |
| `GOOGLE_API_KEY` | provider=google 시 | Google Gemini API 키 |
| `OPENAI_API_KEY` | provider=openai 시 | OpenAI API 키 |
| `CLOVA_OCR_URL` | 선택 | CLOVA OCR API 엔드포인트 |
| `CLOVA_OCR_SECRET` | 선택 | CLOVA OCR 시크릿 키 |

`python-dotenv`로 프로젝트 루트의 `.env` 파일에서 로딩.

## Config 클래스 구조

> 소스: `utils/config.py`

```python
class LLMConfig(BaseModel):
    provider: str           # "anthropic" | "google" | "openai"
    anthropic_api_key: str  # ANTHROPIC_API_KEY
    google_api_key: str     # GOOGLE_API_KEY
    openai_api_key: str     # OPENAI_API_KEY
    claude_model: str       # "claude-sonnet-4-20250514"
    google_model: str       # "gemini-2.5-flash"
    openai_model: str       # "gpt-5-2025-08-07"
    max_tokens: int         # 16384
    max_tokens_limit: int   # 128000
    dynamic_token_adjustment: bool  # True
    temperature: float      # 0.1

class ClovaOCRConfig(BaseModel):
    url: str         # CLOVA_OCR_URL
    secret_key: str  # CLOVA_OCR_SECRET

class Config(BaseModel):
    llm: LLMConfig
    clova_ocr: ClovaOCRConfig
    image_dpi: int    # 300 (범위: 150-900)
    output_dir: str   # "output"
```
```

- [ ] **Step 2: 커밋**

```bash
git add docs/integrations.md
git commit -m "docs: add integrations.md for AI agent context"
```

---

### Task 4: docs/conventions.md 작성

**Files:**
- Create: `docs/conventions.md`

- [ ] **Step 1: docs/conventions.md 파일 작성**

Leslie의 `docs/conventions.md` 패턴을 따라 작성. 새 추출기/프로바이더 추가 체크리스트, 프롬프트 관리 규칙, Import/네이밍 규칙, 에러 핸들링 정책을 포함.

```markdown
# 개발 컨벤션

## 새 추출기 추가 체크리스트

1. `extractors/` 디렉토리에 `{name}_extractor.py` 파일 생성 (snake_case)
2. 클래스 구현: `{Name}Extractor` — 주요 메서드는 `extract_text(pdf_bytes, page_number)` 또는 특수 목적 메서드
3. `processors/single_page_pipeline.py`에 추출기 인스턴스 생성 및 `asyncio.gather`에 추가
4. `prompts.py`의 `get_llm_merge_prompt()`와 `format_extraction_data()`에 새 추출기 특성 반영

> 참고: 기존 추출기 패턴은 `extractors/clova_ocr_extractor.py` (async) 또는 `extractors/pdfplumber_extractor.py` (sync + asyncio.to_thread) 참조

## 새 LLM 프로바이더 추가 체크리스트

1. `utils/config.py`의 `LLMConfig`에 API 키 필드와 모델명 필드 추가, `provider` 패턴 업데이트
2. `extractors/llm_extractor.py`의 `_call_llm_image()`에 프로바이더 분기 추가
3. `processors/llm_merger.py`의 `_call_llm_for_merge()`에 프로바이더 분기 추가
4. `processors/final_orchestrator.py`의 `_call_llm_for_final_generation()`에 프로바이더 분기 추가
5. `utils/rate_limiter.py`의 `APIRateLimiters.get_limiter()`에 라우팅 키워드 추가

> 참고: 3개 파일(llm_extractor, llm_merger, final_orchestrator) 모두에 동일한 프로바이더 분기 패턴 적용 필요

## 프롬프트 관리 규칙

- **모든 LLM 프롬프트는 `prompts.py`에서 관리** — 추출기/프로세서 파일에 프롬프트 하드코딩 금지
- 프롬프트 변수: 상수는 대문자 (`IMAGE_EXTRACTION_PROMPT`), 동적 생성은 함수 (`get_llm_merge_prompt()`)
- 포맷팅 유틸리티: `format_extraction_data()` 등 데이터 포맷팅도 `prompts.py`에 위치

## Import 규칙

- 모든 `from`/`import` 구문은 **파일 최상단**에 배치 (함수/메서드 내부 import 금지)

## 네이밍 규칙

| 대상 | 규칙 | 예시 |
|------|------|------|
| 추출기 클래스 | PascalCase + `Extractor` | `ClovaOCRExtractor`, `PDFPlumberExtractor` |
| 프로세서 클래스 | PascalCase | `LLMMerger`, `FinalOrchestrator` |
| 파일명 | snake_case | `clova_ocr_extractor.py`, `llm_merger.py` |
| 디렉토리 | snake_case | `extractors/`, `processors/` |
| 설정 클래스 | PascalCase + `Config` | `LLMConfig`, `ClovaOCRConfig` |

## 에러 핸들링 정책

- 개별 추출기 실패는 전체 파이프라인을 중단하지 않음 — 다른 추출기가 보완
- CLOVA OCR 실패: None 반환, 로그 기록
- PDFPlumber/PyMuPDF 예외: 빈 결과 반환, 로그 기록
- LLM API 실패: 예외 전파 → 상위에서 재시도 또는 폴백
- FinalOrchestrator LLM 실패: 단순 페이지 연결로 폴백 마크다운 생성
- **LLM Key 미설정 시: 즉시 실행 중단하고 사용자에게 key 설정 요청**
```

- [ ] **Step 2: 커밋**

```bash
git add docs/conventions.md
git commit -m "docs: add conventions.md for AI agent context"
```

---

### Task 5: CLAUDE.md 경량 인덱스로 재작성

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md를 경량 인덱스로 재작성**

기존 ~70줄 인라인 CLAUDE.md를 Leslie 패턴의 ~20줄 인덱스로 교체.

```markdown
# pdf2md

PDF to Markdown 변환 CLI 도구. 다중 추출 엔진 + LLM 기반 병합 파이프라인.

## 기술 스택

Python 3.11 · PyMuPDF · pdfplumber · CLOVA OCR · Anthropic/Google/OpenAI LLM · Pydantic · asyncio

## 핵심 진입점

- `main.py` — CLI 진입점 (argparse)
- `prompts.py` — LLM 프롬프트 중앙 관리
- `utils/config.py` — 환경변수 설정 (Pydantic)

## 핵심 규칙

- LLM Key가 없으면 즉시 중단하고 key 설정 요청
- 테스트 스크립트 타임아웃: 30분
- 모든 LLM 프롬프트는 `prompts.py`에서 관리
- 모든 `from`, `import`는 코드 최상단 배치

## 상세 문서

| 문서 | 내용 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 디렉토리 구조, 실행 모델, 모듈 의존성 |
| [docs/pipeline.md](docs/pipeline.md) | 4개 추출기 파이프라인, 병합 전략, 처리 흐름 |
| [docs/integrations.md](docs/integrations.md) | LLM 3사, CLOVA OCR, 레이트 리밋, 설정 |
| [docs/conventions.md](docs/conventions.md) | 개발 규칙, 추출기/프로바이더 추가 절차, 네이밍 |
```

- [ ] **Step 2: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE.md as lightweight index"
```

---

### Task 6: README.md 상세 문서 테이블 반영

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README.md에 상세 문서 섹션 추가**

README.md의 Project Structure 섹션 아래에 Documentation 섹션 추가:

```markdown
## Documentation

| Document | Content |
|----------|---------|
| [docs/architecture.md](docs/architecture.md) | Directory structure, execution model, module dependencies |
| [docs/pipeline.md](docs/pipeline.md) | 4-extractor pipeline, merge strategy, processing flow |
| [docs/integrations.md](docs/integrations.md) | LLM providers, CLOVA OCR, rate limiting, configuration |
| [docs/conventions.md](docs/conventions.md) | Development rules, adding extractors/providers |
```

- [ ] **Step 2: 최종 커밋**

```bash
git add README.md
git commit -m "docs: add documentation reference table to README"
```
