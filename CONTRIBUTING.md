# Contributing to pdf2md

Thank you for considering contributing to pdf2md!

## How Can I Contribute?

### Reporting Bugs

When creating a bug report, please include:

* Clear and descriptive title
* Steps to reproduce the problem
* Expected vs actual behavior
* Environment details (OS, Python version, LLM provider)

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. If you've changed APIs, update documentation
4. Ensure the test suite passes
5. Follow existing code style

## Development Setup

```bash
git clone https://github.com/yourusername/pdf2md.git
cd pdf2md
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
pdf2md/
├── main.py                  # CLI entry point + usecase chaining
├── prompts.py               # Centralized LLM prompts
├── usecases/                # Usecase layer (task boundaries, dataclass I/O)
│   ├── models.py                 # I/O dataclasses (PageInput, ExtractionResult, etc.)
│   ├── extraction.py             # 4 extraction usecase functions
│   ├── merging.py                # Per-page LLM merge usecase
│   └── finalizing.py             # Final document generation usecase
├── extractors/              # Pure extraction logic
│   ├── pdfplumber_extractor.py   # Text + tables + metadata
│   ├── pymupdf_extractor.py      # Hyperlink-only extraction
│   ├── clova_ocr_extractor.py    # CLOVA OCR API
│   └── llm_extractor.py          # Multimodal LLM extraction
├── processors/              # Processing logic (module-level functions)
│   ├── llm_merger.py             # Adaptive 4-source merging
│   ├── final_orchestrator.py     # Final markdown generation
│   └── image_converter.py        # PDF-to-image conversion
└── utils/                   # Utilities
    ├── config.py                 # Pydantic configuration
    ├── rate_limiter.py           # API rate limiting
    ├── logger.py                 # Loguru logging
    └── validators.py             # PDF validation
```

## Adding a New Extractor

1. Create a new file in `extractors/`
2. Implement with async support:

```python
class MyExtractor:
    def __init__(self, config):
        self.name = "MyExtractor"

    async def extract_text(self, page_pdf_bytes: bytes, page_number: int) -> Optional[str]:
        """Extract text from a single PDF page"""
        pass
```

3. Add a usecase function in `usecases/extraction.py` (async, `PageInput` → `ExtractionResult`)
4. Add the usecase function to `main.py`'s `extract_all_for_page()` in `asyncio.gather`
5. Update the merge prompt in `prompts.py`

## Adding a New LLM Provider

1. Add API key field to `LLMConfig` in `utils/config.py`
2. Add provider routing in `extractors/llm_extractor.py`
3. Add provider routing in `processors/llm_merger.py` and `processors/final_orchestrator.py` (module-level functions)
4. Add rate limit config in `utils/rate_limiter.py`

## Commit Messages

Follow Conventional Commits:

* `feat:` New feature
* `fix:` Bug fix
* `docs:` Documentation changes
* `refactor:` Code restructuring
* `test:` Test changes

## All LLM Prompts

All prompts used by LLM modules must be managed in `prompts.py`. Do not hardcode prompts in extractor or processor files.
