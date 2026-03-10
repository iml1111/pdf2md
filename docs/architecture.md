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
