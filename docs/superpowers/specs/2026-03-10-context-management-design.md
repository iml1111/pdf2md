# pdf2md AI 에이전트 컨텍스트 관리 개선 설계

Leslie-queue-worker PR #71의 JIT 로딩 패턴을 pdf2md에 적용.

---

## 현재 문제

- CLAUDE.md에 ~70줄의 모든 정보가 인라인으로 존재
- 별도 참조 문서 없음 — 에이전트가 항상 전체 컨텍스트를 로딩
- 상세 정보(파이프라인, 연동, 규칙)가 분리되지 않아 JIT 로딩 불가

## 목표 패턴

Leslie-queue-worker와 동일한 구조:
- **CLAUDE.md** = 경량 인덱스 (~20줄): 기술 스택 키워드, 핵심 진입점, docs/ 테이블 링크
- **docs/** = 독립 참조 문서 4개: 필요할 때만 에이전트가 읽음

## 변경 대상

### CLAUDE.md (재작성 → 경량 인덱스)

```markdown
# pdf2md

PDF to Markdown 변환 CLI 도구. 다중 추출 엔진 + LLM 기반 병합 파이프라인.

## 기술 스택

Python 3.11 · PyMuPDF · pdfplumber · CLOVA OCR · Anthropic/Google/OpenAI LLM · Pydantic · asyncio

## 핵심 진입점

- `main.py` — CLI 진입점
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

### docs/architecture.md (신규)

프로젝트 구조와 실행 모델 문서화.

내용:
- 기술 스택 테이블 (Runtime, PDF, OCR, LLM, Config, Logging)
- 디렉토리 구조 ASCII 트리 (인라인 설명 포함)
- 주요 파일 경로 테이블
- 실행 모델 흐름도: CLI → PDF Split → SinglePagePipeline × N → FinalOrchestrator → Markdown
- 모듈 의존성 관계

### docs/pipeline.md (신규)

4개 추출기 파이프라인 상세. Leslie의 `resume-extract-pipeline.md` 패턴 참조.

내용:
- 파이프라인 전체 흐름도 (ASCII)
- 추출기 4개 상세 (역할, 입출력, 강점/한계)
  - PDFPlumber: extract_text() + extract_tables() + extract_metadata()
  - PyMuPDF: extract_hyperlinks() — 6단계 폴백 링크 타입 판별
  - CLOVA OCR: async HTTP POST, PDF bytes 직접 전송
  - LLM Image: 3사 프로바이더별 이미지 포맷 분기
- 적응형 병합 전략 (PDFPlumber 품질 기반)
- 하이퍼링크 통합 규칙 (`[text](#)`)
- 동적 토큰 조정 로직

### docs/integrations.md (신규)

외부 서비스 연동 참조.

내용:
- LLM 프로바이더 3사 (Anthropic/Google/OpenAI)
  - 모델명, API SDK, 레이트 리밋
  - 프로바이더별 호출 차이점
- CLOVA OCR API
  - 엔드포인트, 인증 (X-OCR-SECRET)
  - 요청/응답 형식
- 환경변수 테이블 (.env)
- Config 클래스 구조 (LLMConfig, ClovaOCRConfig, Config)

### docs/conventions.md (신규)

개발 규칙 및 체크리스트.

내용:
- 새 추출기 추가 체크리스트 (4단계)
- 새 LLM 프로바이더 추가 체크리스트 (5단계)
- 프롬프트 관리 규칙 (prompts.py 중앙 관리)
- Import 규칙 (최상단 배치)
- 네이밍 규칙 테이블 (클래스, 파일, 디렉토리)
- 에러 핸들링 정책

## 삭제 대상

- 기존 CLAUDE.md 인라인 내용 (docs/로 이동)
- `docs/superpowers/` 내 설계/계획 문서는 유지 (별도 관심사)

## 작성 원칙

Leslie PR #71과 동일:
1. CLAUDE.md는 인라인 상세 내용 없이 참조 기반 (JIT 로딩)
2. 각 docs/ 문서는 독립적으로 읽을 수 있도록 구성
3. 테이블, 리스트, 코드 블록 위주 — 산문 최소화
4. 파일 경로는 정확하게 명시
5. 한국어 기술 문서 + 영어 기술 용어 혼용
