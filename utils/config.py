"""
Configuration management for hybrid PDF to Markdown pipeline
"""

import os
from typing import Dict
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: str = Field(default="anthropic", pattern="^(anthropic|openai)$")
    anthropic_api_key: str = Field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    openai_api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
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
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image_dpi: int = Field(default=600, ge=150, le=900)
    output_dir: str = Field(default="output")
    


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get current configuration"""
    global _config
    if _config is None:
        _config = Config()
    return _config


