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
