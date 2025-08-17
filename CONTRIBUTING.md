# Contributing to pdf2md

First off, thank you for considering contributing to pdf2md! It's people like you that make pdf2md such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include details about your configuration and environment**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain which behavior you expected to see instead**
* **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the existing code style.
6. Issue that pull request!

## Development Process

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/pdf2md.git
cd pdf2md

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Code Style

We use Black for Python code formatting. Before committing:

```bash
# Format your code
black .

# Check if formatting is correct
black --check .

# Run linting
pylint pdf2md/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pdf2md

# Run specific test file
pytest tests/test_extractors.py
```

### Project Structure

```
pdf2md/
├── extractors/          # PDF extraction engines
│   ├── pymupdf_extractor.py
│   ├── pdfplumber_extractor.py
│   ├── tesseract_extractor.py
│   └── llm_extractor.py
├── processors/          # Processing pipeline
│   ├── single_page_pipeline.py
│   ├── page_orchestrator.py
│   └── final_orchestrator.py
├── utils/              # Utility modules
│   ├── config.py
│   ├── logger.py
│   └── validators.py
├── tests/              # Test files
└── main.py            # CLI entry point
```

### Adding a New Extractor

To add a new extraction engine:

1. Create a new file in `extractors/` directory
2. Implement the base extractor interface:

```python
class MyExtractor:
    def __init__(self):
        self.name = "MyExtractor"
    
    def extract_text(self, pdf_bytes: bytes, page_number: int) -> str:
        """Extract text from a single PDF page"""
        # Your implementation here
        pass
    
    def extract_structure(self, pdf_bytes: bytes, page_number: int) -> Dict:
        """Extract document structure"""
        # Your implementation here
        pass
```

3. Add the extractor to `single_page_pipeline.py`
4. Update the default weights in `config.py`
5. Add tests for your extractor

### Commit Messages

We follow the Conventional Commits specification:

* `feat:` A new feature
* `fix:` A bug fix
* `docs:` Documentation only changes
* `style:` Changes that don't affect the meaning of the code
* `refactor:` A code change that neither fixes a bug nor adds a feature
* `perf:` A code change that improves performance
* `test:` Adding missing tests or correcting existing tests
* `chore:` Changes to the build process or auxiliary tools

Examples:
```
feat: add support for encrypted PDFs
fix: handle unicode characters in table extraction
docs: update installation instructions for Windows
```

## Review Process

1. A maintainer will review your PR within 3-5 business days
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Recognition

Contributors will be recognized in our README and release notes. Thank you for your contributions!

## Questions?

Feel free to open an issue with the tag "question" or reach out to the maintainers directly.