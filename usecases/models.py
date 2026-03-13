"""Dataclass models for usecase function I/O"""

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
    text: str
    tables: list[dict] | None = None
    hyperlinks: list[dict] | None = None
    metadata: dict | None = None
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
    source_file: str = ""


@dataclass
class FinalizeResult:
    """최종 문서 생성 결과"""
    markdown: str
    metadata: dict = field(default_factory=dict)
    error: str | None = None
