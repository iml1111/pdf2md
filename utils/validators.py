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
