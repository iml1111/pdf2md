# Architecture

pdf2md — 다중 추출 엔진과 LLM 병합을 활용하는 PDF to Markdown 변환 CLI 도구.

## 기술 스택

| 영역 | 기술 |
|------|------|
| Runtime | Python 3.11 |
| PDF 처리 | PyMuPDF (fitz), pdfplumber, pypdfium2 |
| OCR | CLOVA OCR (Naver, 외부 API) |
| LLM | Anthropic Claude, OpenAI GPT |
| 이미지 | Pillow (그레이스케일, 선명도/대비 최적화) |
| 설정 | Pydantic BaseModel, python-dotenv (.env) |
| Logging | Loguru |
| Async | asyncio, aiohttp |

## 디렉토리 구조

```
pdf2md/
├── main.py                              # CLI 진입점 + 유스케이스 절차적 체이닝
├── prompts.py                           # LLM 프롬프트 중앙 관리
├── usecases/                            # 유스케이스 레이어 (태스크 경계, dataclass I/O)
│   ├── models.py                        # I/O 데이터클래스 (PageInput, ExtractionResult 등)
│   ├── extraction.py                    # 4개 추출 유스케이스 함수
│   ├── merging.py                       # 페이지별 LLM 병합 유스케이스
│   └── finalizing.py                    # 최종 문서 생성 유스케이스
├── extractors/                          # PDF 추출 엔진 (순수 로직, 4개)
│   ├── pdfplumber_extractor.py          # 텍스트 + 테이블 + 메타데이터
│   ├── pymupdf_extractor.py            # 하이퍼링크 전용 (6단계 폴백)
│   ├── clova_ocr_extractor.py          # CLOVA OCR API (async aiohttp)
│   └── llm_extractor.py               # LLM 이미지 추출 (2사 지원)
├── processors/                          # 처리 로직 (모듈 레벨 함수)
│   ├── llm_merger.py                   # 적응형 4소스 LLM 병합 함수
│   ├── final_orchestrator.py           # 최종 마크다운 생성 함수
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
| `main.py` | CLI 진입점 + 유스케이스 절차적 체이닝 (`--in`, `--out`, `--llm`) |
| `prompts.py` | 모든 LLM 프롬프트 및 포맷팅 함수 |
| `utils/config.py` | 환경변수 → Pydantic Config 로딩 |
| `usecases/models.py` | I/O 데이터클래스 (PageInput, ExtractionResult, MergeInput 등) |
| `usecases/extraction.py` | 4개 추출 유스케이스 함수 (설정 주입 + 에러 핸들링) |
| `usecases/merging.py` | 페이지별 LLM 병합 유스케이스 |
| `usecases/finalizing.py` | 최종 마크다운 생성 유스케이스 |
| `processors/llm_merger.py` | 추출 결과 병합 로직 (모듈 레벨 함수) |
| `processors/final_orchestrator.py` | 최종 마크다운 생성 로직 (모듈 레벨 함수) |

## 실행 모델

```
CLI (main.py)
  ▼
run_pipeline(pdf_path, config)
  ├─ split_pdf()                       # PyMuPDF로 페이지별 PDF bytes 분할
  ├─ Step 1: Extract (배치 단위)
  │   └─ extract_all_for_page() × N    # asyncio.gather로 4개 추출기 병렬
  │       ├─ extract_pdfplumber()       # usecases/extraction.py
  │       ├─ extract_clova_ocr()
  │       ├─ extract_llm_image()
  │       └─ extract_hyperlinks()
  ├─ Step 2: Merge (페이지별)
  │   └─ merge_page() × N              # usecases/merging.py
  └─ Step 3: Finalize (1회)
      └─ finalize_document()            # usecases/finalizing.py
```

- 10페이지 이하: 전체 동시 처리, 초과 시 10페이지 배치
- 모든 LLM 호출은 `APIRateLimiters`로 프로바이더별 레이트 리밋 적용
- 유스케이스 함수는 dataclass I/O로 독립적 실행 가능 (향후 Celery 태스크 전환 대비)

## 모듈 의존성

```
main.py
  ├── usecases/extraction.py
  │     ├── extractors/pdfplumber_extractor.py
  │     ├── extractors/pymupdf_extractor.py
  │     ├── extractors/clova_ocr_extractor.py
  │     ├── extractors/llm_extractor.py
  │     └── processors/image_converter.py
  ├── usecases/merging.py
  │     └── processors/llm_merger.py
  ├── usecases/finalizing.py
  │     └── processors/final_orchestrator.py
  ├── usecases/models.py
  ├── utils/config.py
  ├── utils/rate_limiter.py
  └── prompts.py
```
