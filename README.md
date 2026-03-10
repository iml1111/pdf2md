# pdf2md

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A high-quality PDF to Markdown converter that uses multiple extraction engines and AI-powered merging to produce superior results, especially for complex layouts, tables, and mixed-language documents.

## Key Features

- **Multi-Engine Extraction**: Runs 4 specialized extractors in parallel for comprehensive text capture
- **AI-Powered Merging**: Uses LLMs (Claude/Gemini/GPT) to intelligently merge and correct extraction results
- **Adaptive Merge Strategy**: Dynamically adjusts merging approach based on PDF type (native vs scanned)
- **Multilingual Support**: Excellent support for mixed-language documents (English, Korean, Chinese, Japanese, etc.)
- **Superior Table Extraction**: PDFPlumber-based table detection with markdown conversion
- **Hyperlink Preservation**: Detects and preserves hyperlinks from PDF annotations
- **CLOVA OCR Integration**: Naver CLOVA OCR for precise character-level text recognition

## Quick Start

### Prerequisites

- Python 3.11+
- API key for at least one LLM provider (Anthropic, Google, or OpenAI)
- CLOVA OCR credentials (optional, for enhanced OCR)

### Installation

```bash
git clone https://github.com/yourusername/pdf2md.git
cd pdf2md
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# LLM Provider (at least one required)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# CLOVA OCR (optional)
CLOVA_OCR_URL=https://your-clova-ocr-endpoint
CLOVA_OCR_SECRET=your-secret-key
```

### Basic Usage

```bash
# Convert with default provider (Anthropic)
python main.py --in document.pdf

# Specify output path
python main.py --in document.pdf --out output.md

# Use Google Gemini
python main.py --in document.pdf --llm google

# Use OpenAI
python main.py --in document.pdf --llm openai
```

## Architecture

### Pipeline Overview

```
PDF Document
    │
    ▼
Page Splitter (PyMuPDF)
    │
    ▼ (per page, parallel)
┌───────────────────────────────────────────┐
│  4 Extractors (concurrent)                │
│  ├── PDFPlumber  (text + tables + meta)   │
│  ├── PyMuPDF     (hyperlinks only)        │
│  ├── CLOVA OCR   (async HTTP)             │
│  └── LLM Image   (multimodal vision)      │
└───────────────────────────────────────────┘
    │
    ▼
LLM Merger (adaptive 4-source merge)
    │
    ▼
Final Orchestrator (markdown generation)
    │
    ▼
Markdown Output
```

### Extraction Engines

| Engine | Role | Strengths |
|--------|------|-----------|
| **PDFPlumber** | Text + Tables + Metadata | Native PDF text, table structures, layout measurements |
| **PyMuPDF** | Hyperlink Detection | Link type classification (text/image/form/annotation/drawing/area) |
| **CLOVA OCR** | Precise Text Recognition | Character-level precision, Korean text, multilingual support |
| **LLM Image** | Visual Understanding | Visual layout interpretation, reading order, document flow |

### LLM Providers

| Provider | Model | Rate Limit | Use Case |
|----------|-------|------------|----------|
| Anthropic | claude-sonnet-4-20250514 | 5 req/s | Default, balanced performance |
| Google | gemini-2.5-flash | 3 req/s | Fast, cost-effective |
| OpenAI | gpt-5-2025-08-07 | 10 req/s | High throughput |

### Adaptive Merge Strategy

The LLM Merger evaluates PDFPlumber output quality and adjusts its strategy:

- **Text-rich native PDFs**: PDFPlumber as primary source, CLOVA OCR for validation
- **Image-heavy/scanned PDFs**: CLOVA OCR + LLM Image as primary, PDFPlumber deprioritized
- **Hyperlinks**: PyMuPDF links integrated inline as `[text](#)` markdown format

## Project Structure

```
pdf2md/
├── main.py                  # CLI entry point
├── prompts.py               # Centralized LLM prompts
├── extractors/
│   ├── pdfplumber_extractor.py   # Text + tables + metadata
│   ├── pymupdf_extractor.py      # Hyperlink-only extraction
│   ├── clova_ocr_extractor.py    # CLOVA OCR API integration
│   └── llm_extractor.py          # Multimodal LLM extraction
├── processors/
│   ├── single_page_pipeline.py   # Per-page orchestration
│   ├── llm_merger.py             # Adaptive 4-source merging
│   ├── final_orchestrator.py     # Final markdown generation
│   └── image_converter.py        # PDF-to-image conversion
└── utils/
    ├── config.py                  # Pydantic configuration
    ├── rate_limiter.py            # API rate limiting
    ├── logger.py                  # Loguru logging
    └── validators.py              # PDF validation
```

## Documentation

| Document | Content |
|----------|---------|
| [docs/architecture.md](docs/architecture.md) | Directory structure, execution model, module dependencies |
| [docs/pipeline.md](docs/pipeline.md) | 4-extractor pipeline, merge strategy, processing flow |
| [docs/integrations.md](docs/integrations.md) | LLM providers, CLOVA OCR, rate limiting, configuration |
| [docs/conventions.md](docs/conventions.md) | Development rules, adding extractors/providers |

## Configuration

Configuration is managed via Pydantic models with environment variable defaults:

| Config | Field | Default | Description |
|--------|-------|---------|-------------|
| LLM | `provider` | `anthropic` | LLM provider (`anthropic`/`google`/`openai`) |
| LLM | `max_tokens` | `16384` | Max output tokens |
| LLM | `temperature` | `0.1` | LLM temperature |
| LLM | `dynamic_token_adjustment` | `true` | Auto-adjust tokens for Korean/English ratio |
| Image | `image_dpi` | `300` | Image conversion DPI (150-900) |

## Dependencies

### Core
- **PyMuPDF** >= 1.24.0 — PDF processing and hyperlink extraction
- **pdfplumber** >= 0.11.0 — Table detection and text extraction
- **Pillow** >= 10.3.0 — Image processing

### LLM Providers
- **anthropic** >= 0.34.0 — Claude API
- **google-genai** >= 1.0.0 — Gemini API
- **openai** >= 1.40.0 — OpenAI API

### Infrastructure
- **aiohttp** >= 3.9.0 — Async HTTP for CLOVA OCR
- **pydantic** >= 2.0.0 — Configuration management
- **loguru** >= 0.7.0 — Structured logging

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- [pdfplumber](https://github.com/jsvine/pdfplumber) for table extraction
- [Naver CLOVA OCR](https://clova.ai/) for precise text recognition
- [Anthropic](https://www.anthropic.com/), [Google](https://ai.google.dev/), and [OpenAI](https://openai.com/) for LLM APIs
