"""
Configuration management for hybrid PDF to Markdown pipeline
"""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExtractorWeights(BaseModel):
    """Weights for different extractors in the final decision"""
    pymupdf: float = Field(default=0.20, ge=0, le=1)
    pdfplumber: float = Field(default=0.25, ge=0, le=1)
    tesseract: float = Field(default=0.20, ge=0, le=1)
    llm_img: float = Field(default=0.15, ge=0, le=1)
    llm_pdf: float = Field(default=0.20, ge=0, le=1)


class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: str = Field(default="anthropic", pattern="^(anthropic|openai)$")
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    openai_model: str = Field(default="gpt-5-2025-08-07")
    max_tokens: int = Field(default=16384)
    max_tokens_limit: int = Field(default=128000)
    dynamic_token_adjustment: bool = Field(default=True)
    temperature: float = Field(default=0.1)


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    image_dpi: int = Field(default=300, ge=150, le=600)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4)
    quality_threshold: float = Field(default=8.0, ge=0, le=10)
    enable_ocr: bool = Field(default=True)
    enable_llm: bool = Field(default=True)
    verbose: bool = Field(default=False)


class Config(BaseModel):
    """Main pipeline configuration"""
    extractor_weights: ExtractorWeights = Field(default_factory=ExtractorWeights)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image_dpi: int = Field(default=600, ge=150, le=900)
    output_dir: str = Field(default="output")
    
    @model_validator(mode='after')
    def validate_and_load_api_keys(self):
        """Load API keys from environment and validate them"""
        
        # Load API keys from environment if not provided
        if not self.llm.anthropic_api_key:
            self.llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.llm.openai_api_key:
            self.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate required API keys are present
        if self.llm.provider == 'anthropic':
            if not self.llm.anthropic_api_key:
                raise ValueError(
                    "❌ ANTHROPIC_API_KEY is required but not found.\n"
                    "Please set it in .env file or as environment variable:\n"
                    "export ANTHROPIC_API_KEY='your-api-key-here'"
                )
        elif self.llm.provider == 'openai':
            if not self.llm.openai_api_key:
                raise ValueError(
                    "❌ OPENAI_API_KEY is required but not found.\n"
                    "Please set it in .env file or as environment variable:\n"
                    "export OPENAI_API_KEY='your-api-key-here'"
                )
        
        return self
    
    @classmethod
    def from_file(cls, path: str):
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get current configuration"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def update_config(**kwargs) -> Config:
    """Update global configuration with provided values"""
    global _config
    if _config is None:
        _config = Config()
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    return _config