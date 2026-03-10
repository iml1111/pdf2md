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
2. 선명도 x 1.2
3. 대비 x 1.1

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
2. 토큰 추정: (korean_ratio x 1.5 + (1 - korean_ratio) x 0.25) x text_length x 1.2
3. 필요 토큰: max(추정값 x 2, 기본 max_tokens)
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
