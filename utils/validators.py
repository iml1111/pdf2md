"""
Validation utilities for the pipeline
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if the provided file is a valid PDF
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if valid PDF, False otherwise
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a file
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    
    # Check extension
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Not a PDF file: {file_path}")
    
    # Check if file is not empty
    if path.stat().st_size == 0:
        raise ValueError(f"Empty file: {file_path}")
    
    # Check PDF header
    with open(path, 'rb') as f:
        header = f.read(5)
        if header != b'%PDF-':
            raise ValueError(f"Invalid PDF header: {file_path}")
    
    return True


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_extraction_result(result: Dict[str, Any]) -> float:
    """
    Validate and score extraction result
    
    Args:
        result: Extraction result dictionary
        
    Returns:
        Quality score (0-10)
    """
    score = 0.0
    
    # Check if text exists
    if result.get('text'):
        score += 2.0
        # Check text length
        text_length = len(result['text'])
        if text_length > 100:
            score += 2.0
        if text_length > 500:
            score += 1.0
    
    # Check if structure is preserved
    if result.get('structure'):
        score += 2.0
        # Check for headers
        if result['structure'].get('headers'):
            score += 1.0
        # Check for paragraphs
        if result['structure'].get('paragraphs'):
            score += 1.0
    
    # Check for metadata
    if result.get('metadata'):
        score += 1.0
    
    return min(score, 10.0)


def ensure_output_dir(output_path: str) -> Path:
    """
    Ensure output directory exists
    
    Args:
        output_path: Path to output file or directory
        
    Returns:
        Path object for the directory
    """
    path = Path(output_path)
    
    # If it's a file path, get the parent directory
    if path.suffix:
        directory = path.parent
    else:
        directory = path
    
    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)
    
    return path