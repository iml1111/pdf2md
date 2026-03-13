# Usecase Layer Refactoring Design

파이프라인 핵심 로직을 유스케이스 함수로 분리하여 Celery 기반 비동기 워커 전환을 준비한다.

## 목표

- 파이프라인의 병목 로직(추출, OCR, LLM 호출, 병합, 최종화)을 독립 유스케이스 함수로 분리
- 각 함수를 dataclass 기반 입출력으로 정의하여 직렬화 용이하게 설계
- `main.py`에서 유스케이스 함수를 절차적으로 체이닝하여 호출
- 향후 Celery + Redis/RabbitMQ 기반 워커로 전환 시, 유스케이스 함수에 `@celery.task` 데코레이터만 추가하면 되는 구조

## 비목표

- Celery, Redis, RabbitMQ 등 실제 인프라스트럭처 적용
- 기존 추출기(`extractors/`) 내부 로직 변경
- 프롬프트(`prompts.py`) 변경

## 아키텍처

### 현재

```
main.py (PDF2MDPipeline 클래스)
  → processors/single_page_pipeline.py (오케스트레이션 + 추출 병렬 실행)
    → extractors/*.py (추출 로직)
  → processors/llm_merger.py (설정 로딩 + 에러 핸들링 + 병합 로직)
  → processors/final_orchestrator.py (설정 로딩 + 에러 핸들링 + 최종화 로직)
```

### 변경 후

```
main.py (절차적 체이닝)
  → usecases/extraction.py (오케스트레이션: 설정 주입, 에러 핸들링, 입출력 변환)
  → usecases/merging.py (오케스트레이션)
  → usecases/finalizing.py (오케스트레이션)
    → extractors/*.py (순수 추출 로직, 변경 없음)
    → processors/*.py (순수 처리 로직, 오케스트레이션 제거)
```

### 레이어 역할

| 레이어 | 역할 | 담당 |
|--------|------|------|
| `main.py` | CLI 파싱 + 유스케이스 절차적 호출 + 리포트 출력 | 체이닝, 배치 처리, 출력 |
| `usecases/` | 태스크 경계. 설정 주입, 에러 핸들링, 입출력 변환 | Celery 전환 시 `@task` 데코레이터 대상 |
| `extractors/` | 순수 추출 로직 (I/O, 파싱) | 변경 없음 |
| `processors/` | 순수 처리 로직 (병합, 최종화 핵심) | 클래스 → 함수로 전환. LLM 클라이언트를 파라미터로 주입받음 |

### 공유 리소스 관리

현재 `PDF2MDPipeline.__init__`에서 생성하는 공유 인스턴스들의 처리:

| 인스턴스 | 현재 | 변경 후 |
|----------|------|---------|
| `APIRateLimiters` | Pipeline 클래스에서 1회 생성 | `main.py`에서 1회 생성, 유스케이스 함수에 파라미터로 전달 |
| `LLMExtractor` | Pipeline 클래스에서 1회 생성 | 유스케이스 함수 내부에서 직접 생성 (Config 주입) |
| `LLMMerger` | Pipeline 클래스에서 1회 생성 | `processors/llm_merger.py`를 함수로 전환, 유스케이스에서 호출 |
| `FinalOrchestrator` | Pipeline 클래스에서 1회 생성 | `processors/final_orchestrator.py`를 함수로 전환, 유스케이스에서 호출 |
| `PDFPlumberExtractor` | Pipeline 클래스에서 1회 생성 | 유스케이스 함수 내부에서 직접 생성 (stateless) |
| `PyMuPDFExtractor` | Pipeline 클래스에서 1회 생성 | 유스케이스 함수 내부에서 직접 생성 (stateless) |
| `ClovaOCRExtractor` | Pipeline 클래스에서 1회 생성 | 유스케이스 함수 내부에서 직접 생성 (Config 주입) |
| `ImageConverter` | Pipeline 클래스에서 1회 생성 | 유스케이스 함수 내부에서 직접 생성 (Config.image_dpi 주입) |

Celery 전환 시: LLM 클라이언트(`Anthropic`, `OpenAI`)와 `APIRateLimiters`는 pickle 불가. 워커 프로세스 내에서 초기화하거나 Celery worker `--pool=solo` 모드에서 모듈 레벨 싱글턴으로 관리한다.

## Dataclass 모델

`usecases/models.py`에 정의한다.

```python
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
    text: str                # (prompts.py의 format_extraction_data()가 이 이름으로 매칭)
    tables: list[dict] | None = None        # pdfplumber 전용. [{index, data, markdown}]
    hyperlinks: list[dict] | None = None     # pymupdf 전용. [{url, text, link_type, rect}]
    metadata: dict | None = None             # 추출기별 부가 정보 (structure, llm_model 등)
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
    source_file: str = ""    # PDF 파일명 (프롬프트 메타데이터용)

@dataclass
class FinalizeResult:
    """최종 문서 생성 결과"""
    markdown: str
    metadata: dict = field(default_factory=dict)
    error: str | None = None
```

설계 의도:
- `bytes` 필드(`page_bytes`)는 Celery 전환 시 S3/Redis 참조 키로 대체 가능
- `error` 필드로 각 스텝의 실패를 전파하되 파이프라인은 계속 진행 (현재 동작 유지)
- `ExtractionResult`를 4개 추출기가 공유. `metadata` dict에 추출기별 부가 정보(`structure`, `llm_model`, `llm_provider` 등) 저장
- `ExtractionResult.tables`는 `list[dict]` — 기존 `PDFPlumberExtractor.extract_tables()` 반환 형태(`[{index, data, markdown}]`)와 일치
- `FinalizeInput.source_file` — `FinalOrchestrator._generate_metadata()`에서 필요한 파일명 전달
- `dataclasses.asdict()`로 JSON 직렬화 가능. 모든 필드는 JSON-serializable 타입 (`bytes` 제외, Celery 전환 시 참조 키로 대체)

## 유스케이스 함수

### `usecases/extraction.py`

4개 추출 유스케이스 함수. 모두 `async def`로 정의한다.

```python
from utils.rate_limiter import APIRateLimiters

async def extract_pdfplumber(input: PageInput) -> ExtractionResult:
    """PDFPlumber 텍스트/테이블/메타데이터 추출"""
    # 1. PDFPlumberExtractor() 생성 (stateless, config 불필요)
    # 2. asyncio.to_thread()로 extract_text, extract_tables, extract_metadata 순차 호출
    # 3. 결과를 ExtractionResult로 변환
    #    - text: extract_text() 결과
    #    - tables: extract_tables() 결과 (list[dict] 그대로)
    #    - metadata: extract_metadata() 결과
    # 4. 예외 시 error 필드에 기록, 빈 결과 반환

async def extract_clova_ocr(input: PageInput, config: Config) -> ExtractionResult:
    """CLOVA OCR API 호출 (네이티브 async)"""
    # 1. ClovaOCRExtractor(config.clova_ocr) 생성
    # 2. 기존 async 로직 호출 (재시도 포함)
    # 3. 결과를 ExtractionResult로 변환

async def extract_llm_image(
    input: PageInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> ExtractionResult:
    """LLM 멀티모달 비전 추출"""
    # 1. ImageConverter(dpi=config.image_dpi) 생성
    # 2. image_converter로 PDF bytes → 이미지 변환 + OCR 최적화
    # 3. LLMExtractor(config) 생성
    # 4. rate_limiters.get_limiter(config.llm.provider).acquire()
    # 5. asyncio.to_thread()로 llm_extractor.extract_single_page_image() 호출
    # 6. 결과를 ExtractionResult로 변환
    #    - text: result['text']
    #    - metadata: {structure, llm_model, llm_provider} (추출기 부가 정보)

async def extract_hyperlinks(input: PageInput) -> ExtractionResult:
    """PyMuPDF 하이퍼링크 추출"""
    # 1. PyMuPDFExtractor() 생성 (stateless)
    # 2. asyncio.to_thread()로 extract_hyperlinks() 호출
    # 3. 결과를 ExtractionResult로 변환
    #    - text: "" (하이퍼링크 전용, 텍스트 없음)
    #    - hyperlinks: 추출된 링크 리스트
```

각 함수의 책임:
- dataclass 입력 → 기존 추출기 인스턴스 생성 및 호출 → dataclass 출력 변환
- 에러 핸들링: 예외를 잡아서 `ExtractionResult(error=...)` 반환
- Config/rate_limiters는 파라미터로 주입 (전역 상태 의존 없음)
- 모든 함수가 `Config`를 받는 통일된 인터페이스 (`extract_pdfplumber`, `extract_hyperlinks`는 config 불필요하지만 일관성을 위해 생략)
- `extractor_name`은 기존 `prompts.py`의 `format_extraction_data()`가 매칭하는 이름과 동일하게 유지: `"pdfplumber"`, `"clova_ocr"`, `"llm_img"`, `"pymupdf"`

### `usecases/merging.py`

```python
async def merge_page(
    input: MergeInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> MergeResult:
    """4개 추출 결과를 LLM으로 병합"""
    # 1. input.extraction_results를 Dict[str, Dict] 형태로 변환
    #    (prompts.format_extraction_data()가 기대하는 형태)
    #    변환 로직:
    #      results_dict = {}
    #      for r in input.extraction_results:
    #          entry = {'text': r.text}
    #          if r.tables: entry['tables'] = r.tables
    #          if r.hyperlinks: entry['hyperlinks'] = r.hyperlinks
    #          if r.metadata: entry['metadata'] = r.metadata
    #          if r.error: entry['error'] = r.error
    #          results_dict[r.extractor_name] = entry
    #
    # 2. 유효한 결과 필터링 (error=None이고 text가 비어있지 않은 것)
    # 3. 유효 결과 0개: 빈 MergeResult 반환
    # 4. 유효 결과 1개 (hyperlinks 제외): LLM 호출 스킵, 해당 텍스트 직접 반환
    # 5. 유효 결과 2개 이상: format_extraction_data() → get_llm_merge_prompt() → LLM 호출
    # 6. LLM 호출: processors/llm_merger의 핵심 로직 사용 (레이트 리미터 적용)
    # 7. MergeResult로 변환
```

### `usecases/finalizing.py`

```python
async def finalize_document(
    input: FinalizeInput,
    config: Config,
) -> FinalizeResult:
    """병합된 페이지들을 최종 마크다운으로 생성"""
    # 1. input.merge_results를 page_number 순서로 정렬
    # 2. [{page_number, content, error}] 형태로 변환
    #    (processors/final_orchestrator가 기대하는 형태)
    # 3. processors/final_orchestrator의 핵심 로직 호출:
    #    - _combine_page_contents(): 페이지 결합
    #    - _generate_metadata(): 메타데이터 생성 (source_file 전달)
    #    - _calculate_dynamic_max_tokens(): 다이나믹 토큰 조정
    #    - LLM API 호출 (asyncio.to_thread()로 감싸기 — 기존 동기 함수)
    #    - _post_process_markdown(): 후처리
    # 4. LLM 실패 시 폴백: _fallback_generation()
    # 5. FinalizeResult로 변환
```

## main.py 체이닝 구조

```python
import asyncio
import itertools
from pathlib import Path
from usecases.models import PageInput, MergeInput, FinalizeInput
from usecases.extraction import (
    extract_pdfplumber, extract_clova_ocr,
    extract_llm_image, extract_hyperlinks,
)
from usecases.merging import merge_page
from usecases.finalizing import finalize_document
from utils.rate_limiter import APIRateLimiters


def batched(iterable, n):
    """itertools.batched polyfill for Python 3.11"""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


async def extract_all_for_page(page, config, rate_limiters):
    """단일 페이지의 4개 추출기를 병렬 실행하는 편의 함수"""
    return await asyncio.gather(
        extract_pdfplumber(page),
        extract_clova_ocr(page, config),
        extract_llm_image(page, config, rate_limiters),
        extract_hyperlinks(page),
    )


async def run_pipeline(args, config):
    """파이프라인 메인 로직"""
    rate_limiters = APIRateLimiters()

    # --- Step 0: PDF 검증 + 분할 ---
    pdf_path = Path(args.input_pdf)
    validate_pdf_file(str(pdf_path))
    page_bytes_list = split_pdf(str(pdf_path))  # -> list[bytes]
    total_pages = len(page_bytes_list)
    pages = [
        PageInput(page_bytes=b, page_number=i+1, total_pages=total_pages)
        for i, b in enumerate(page_bytes_list)
    ]

    # --- Step 1: 추출 (페이지별 병렬, 배치 단위) ---
    all_extractions = []
    for batch in batched(pages, 10):
        batch_results = await asyncio.gather(*[
            extract_all_for_page(page, config, rate_limiters)
            for page in batch
        ])
        all_extractions.extend(batch_results)

    # --- Step 2: 병합 (페이지별) ---
    merge_results = []
    for page_input, extractions in zip(pages, all_extractions):
        result = await merge_page(
            MergeInput(
                page_number=page_input.page_number,
                extraction_results=list(extractions),
            ),
            config,
            rate_limiters,
        )
        merge_results.append(result)

    # --- Step 3: 최종 문서 생성 ---
    final = await finalize_document(
        FinalizeInput(
            merge_results=merge_results,
            total_pages=total_pages,
            source_file=pdf_path.name,
        ),
        config,
    )

    return final, pages, merge_results


def split_pdf(pdf_path: str) -> list[bytes]:
    """PDF를 페이지별 bytes로 분할. 기존 _split_pdf_into_pages 로직 추출."""
    import fitz
    page_pdfs = []
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        single_page_doc = fitz.open()
        single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        page_pdfs.append(single_page_doc.tobytes())
        single_page_doc.close()
    doc.close()
    return page_pdfs


def main():
    """CLI 진입점: 파싱 → 검증 → 파이프라인 실행 → 리포트 출력"""
    args = parse_args()
    config = get_config()
    config.llm.provider = args.llm
    config.llm.validate_credentials()
    config.clova_ocr.validate_credentials()

    final, pages, merge_results = asyncio.run(run_pipeline(args, config))

    # --- 출력 ---
    output_path = write_output(final, args)

    # --- JSON 리포트 (현재 동작 유지) ---
    print_report(pages, merge_results, final, output_path)
```

설계 의도:
- 각 스텝이 명시적으로 분리되어 파이프라인 흐름이 한눈에 보임
- `extract_all_for_page()`는 편의 함수 — Celery 전환 시 `chord(extract_*.s()) | merge_page.s()` 패턴으로 대체
- `batched()`: Python 3.11 polyfill. Python 3.12+ 전환 시 `itertools.batched`로 교체
- `split_pdf()`: 기존 `PDF2MDPipeline._split_pdf_into_pages()` 로직을 독립 함수로 추출
- `write_output()`, `print_report()`: 기존 `process_pdf()`의 파일 저장 + JSON 리포트 출력 로직 추출
- `validate_pdf_file()`: 기존과 동일하게 파이프라인 시작 전 PDF 검증
- `PDF2MDPipeline` 클래스 제거, 절차적 함수 구조로 단순화
- Step 2 병합은 페이지별 순차 실행. 현재 코드는 배치 내에서 추출+병합이 페이지 단위로 병렬 실행되므로, 이 변경은 병합 단계에서 처리량 감소가 발생한다. 이는 스텝 분리의 트레이드오프로 수용하며, Celery 전환 시 각각 독립 태스크로 병렬화하여 복원 가능

## 기존 모듈 변경 사항

### 변경

| 파일 | 변경 내용 |
|------|-----------|
| `main.py` | `PDF2MDPipeline` 클래스 제거. 절차적 `main()` + `run_pipeline()` + 유틸 함수로 교체 |
| `processors/single_page_pipeline.py` | **삭제**. 오케스트레이션은 `usecases/extraction.py`로 이동. `_convert_page_to_image()` 로직은 `image_converter.py`로 이동 |
| `processors/image_converter.py` | `SinglePagePipeline._convert_page_to_image()` (fitz 기반 PDF bytes → PNG 이미지 변환) 로직을 `convert_page_to_image(pdf_bytes, dpi)` 메서드로 흡수. 기존 `optimize_for_ocr()` 유지 |
| `processors/llm_merger.py` | `LLMMerger` 클래스의 핵심 LLM 호출 로직을 함수로 전환. 클라이언트(`Anthropic`, `OpenAI`)와 rate_limiters를 파라미터로 주입받는 형태. `_filter_valid_results`, `_call_llm_for_merge`, `_fallback_merge` 로직 유지 |
| `processors/final_orchestrator.py` | `FinalOrchestrator` 클래스의 핵심 로직을 함수로 전환. 클라이언트와 config를 파라미터로 주입받는 형태. `_combine_page_contents`, `_generate_metadata`, `_calculate_dynamic_max_tokens`, `_call_llm_for_final_generation`, `_post_process_markdown`, `_fallback_generation` 로직 유지 |

### 미변경

| 파일 | 이유 |
|------|------|
| `extractors/pdfplumber_extractor.py` | 순수 추출 로직. 그대로 유지 |
| `extractors/pymupdf_extractor.py` | 순수 추출 로직. 그대로 유지 |
| `extractors/clova_ocr_extractor.py` | 순수 추출 로직 (async). 그대로 유지 |
| `extractors/llm_extractor.py` | 순수 추출 로직. 그대로 유지 |
| `prompts.py` | 프롬프트 중앙 관리. 그대로 유지 |
| `utils/config.py` | Pydantic 설정. 그대로 유지 |
| `utils/rate_limiter.py` | 레이트 리미팅. 그대로 유지 |
| `utils/logger.py` | 로깅. 그대로 유지 |
| `utils/validators.py` | PDF 검증. 그대로 유지 |

### 신규 생성

| 파일 | 내용 |
|------|------|
| `usecases/__init__.py` | 패키지 초기화 |
| `usecases/models.py` | dataclass 입출력 모델 정의 |
| `usecases/extraction.py` | 4개 추출 유스케이스 함수 |
| `usecases/merging.py` | 페이지 병합 유스케이스 함수 |
| `usecases/finalizing.py` | 최종 문서 생성 유스케이스 함수 |

### 모델 선택: dataclass vs Pydantic

`utils/config.py`는 Pydantic `BaseModel`을 사용하지만, 유스케이스 모델은 `dataclass`를 사용한다.
- Pydantic: 환경변수 파싱, 값 검증이 필요한 설정에 적합
- dataclass: 순수 데이터 전달 객체에 적합. 검증 불필요, `asdict()` 직렬화가 간결

## 검증 전략

이 리팩토링은 동작 변경 없는 순수 구조 변경이다. 검증 방법:

1. **동일 PDF 입력 → 동일 마크다운 출력 비교**: 리팩토링 전후 동일 PDF를 실행하여 출력 마크다운이 동일한지 diff 비교
2. **JSON 리포트 비교**: 리포트의 `successful_pages`, `content_length` 등이 일치하는지 확인
3. **에러 시나리오**: 잘못된 PDF, 네트워크 에러 시 기존과 동일한 폴백 동작 확인

## Celery 전환 경로

이 설계가 완성되면, 향후 Celery 전환은 다음과 같이 진행된다:

1. `usecases/` 함수에 `@celery.task` 데코레이터 추가
2. `main.py`의 절차적 호출을 Celery 체인으로 교체:
   ```python
   # 현재 (절차적)
   result = await extract_pdfplumber(page)

   # Celery 전환 후
   result = extract_pdfplumber.delay(asdict(page))
   ```
3. `PageInput.page_bytes`를 S3/Redis 참조 키로 대체 (대용량 바이너리 직렬화 회피)
4. `extract_all_for_page()`를 `chord(group(extract_*.s()), merge_page.s())` 패턴으로 교체
5. 추출기별 워커 concurrency를 독립 설정 (예: CLOVA OCR 워커 3개, LLM 워커 5개)
6. LLM 클라이언트와 `APIRateLimiters`는 워커 프로세스 내 모듈 레벨 싱글턴으로 관리 (pickle 불가)
7. `finalize_document`에 rate_limiters 추가 (현재는 단일 실행이라 불필요하지만, 동시 실행 시 필요)

dataclass의 `asdict()` / 생성자 패턴으로 직렬화/역직렬화를 처리한다.
