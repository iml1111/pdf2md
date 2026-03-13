# 개발 컨벤션

## 새 추출기 추가 체크리스트

1. `extractors/` 디렉토리에 `{name}_extractor.py` 파일 생성 (snake_case)
2. 클래스 구현: `{Name}Extractor` — 주요 메서드는 `extract_text(pdf_bytes, page_number)` 또는 특수 목적 메서드
3. `usecases/extraction.py`에 유스케이스 함수 추가 (async, PageInput → ExtractionResult)
4. `main.py`의 `extract_all_for_page()`에서 새 유스케이스 함수를 `asyncio.gather`에 추가
5. `prompts.py`의 `get_llm_merge_prompt()`와 `format_extraction_data()`에 새 추출기 특성 반영

> 참고: 기존 추출기 패턴은 `extractors/clova_ocr_extractor.py` (async) 또는 `extractors/pdfplumber_extractor.py` (sync + asyncio.to_thread) 참조
> 유스케이스 함수 패턴은 `usecases/extraction.py`의 `extract_pdfplumber()` 참조

## 새 LLM 프로바이더 추가 체크리스트

1. `utils/config.py`의 `LLMConfig`에 API 키 필드와 모델명 필드 추가, `provider` 패턴 업데이트
2. `extractors/llm_extractor.py`의 `_call_llm_image()`에 프로바이더 분기 추가
3. `processors/llm_merger.py`의 `call_llm_for_merge()`에 프로바이더 분기 추가
4. `processors/final_orchestrator.py`의 `call_llm_for_final_generation()`에 프로바이더 분기 추가
5. `utils/rate_limiter.py`의 `APIRateLimiters.get_limiter()`에 라우팅 키워드 추가

> 참고: 3개 파일(llm_extractor, llm_merger, final_orchestrator) 모두에 동일한 프로바이더 분기 패턴 적용 필요. llm_merger와 final_orchestrator는 모듈 레벨 함수 (클래스 아님)

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
| 프로세서 모듈 | snake_case 함수 | `merge_text()`, `generate_final_document()` |
| 유스케이스 함수 | snake_case 동사 | `extract_pdfplumber()`, `merge_page()`, `finalize_document()` |
| 유스케이스 모델 | PascalCase | `PageInput`, `ExtractionResult`, `MergeResult` |
| 파일명 | snake_case | `clova_ocr_extractor.py`, `llm_merger.py` |
| 디렉토리 | snake_case | `extractors/`, `processors/`, `usecases/` |
| 설정 클래스 | PascalCase + `Config` | `LLMConfig`, `ClovaOCRConfig` |

## 에러 핸들링 정책

- 개별 추출기 실패는 전체 파이프라인을 중단하지 않음 — 다른 추출기가 보완
- CLOVA OCR 실패: None 반환, 로그 기록
- PDFPlumber/PyMuPDF 예외: 빈 결과 반환, 로그 기록
- LLM API 실패: 예외 전파 → 상위에서 재시도 또는 폴백
- final_orchestrator LLM 실패: 단순 페이지 연결로 폴백 마크다운 생성
- **LLM Key 미설정 시: 즉시 실행 중단하고 사용자에게 key 설정 요청**
