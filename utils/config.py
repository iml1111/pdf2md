"""
Configuration management for hybrid PDF to Markdown pipeline
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: str = Field(default="anthropic", pattern="^(anthropic|google|openai)$")
    anthropic_api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    google_model: str = Field(default="gemini-2.5-flash")
    openai_model: str = Field(default="gpt-5-2025-08-07")
    max_tokens: int = Field(default=16384)
    max_tokens_limit: int = Field(default=128000)
    dynamic_token_adjustment: bool = Field(default=True)
    temperature: float = Field(default=0.1)


class ClovaOCRConfig(BaseModel):
    """CLOVA OCR configuration"""
    url: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_URL", ""))
    secret_key: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_SECRET", ""))


class Config(BaseModel):
    """Main pipeline configuration"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    clova_ocr: ClovaOCRConfig = Field(default_factory=ClovaOCRConfig)
    image_dpi: int = Field(default=300, ge=150, le=900)
    output_dir: str = Field(default="output")


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get current configuration"""
    global _config
    if _config is None:
        _config = Config()
    return _config
