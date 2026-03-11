# pdf2md

PDF to Markdown 변환 CLI 도구. 다중 추출 엔진 + LLM 기반 병합 파이프라인.

## 기술 스택

Python 3.11 · PyMuPDF · pdfplumber · CLOVA OCR · Anthropic/OpenAI LLM · Pydantic · asyncio

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
| [docs/integrations.md](docs/integrations.md) | LLM 2사, CLOVA OCR, 레이트 리밋, 설정 |
| [docs/conventions.md](docs/conventions.md) | 개발 규칙, 추출기/프로바이더 추가 절차, 네이밍 |
