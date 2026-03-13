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
    provider: str = Field(default="anthropic", pattern="^(anthropic|openai)$")
    anthropic_api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    claude_model: str = Field(default="claude-opus-4-6")
    openai_model: str = Field(default="gpt-5.4")
    max_tokens: int = Field(default=16384)
    max_tokens_limit: int = Field(default=128000)
    dynamic_token_adjustment: bool = Field(default=True)
    temperature: float = Field(default=0.1)

    def validate_credentials(self) -> None:
        """Validate that LLM API keys are configured. Raises ValueError if missing."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise ValueError(
                f"LLM API 키가 설정되지 않았습니다: {', '.join(missing)}\n"
                f".env 파일에 다음 변수를 설정하세요:\n"
                f"  ANTHROPIC_API_KEY=<your-anthropic-api-key>\n"
                f"  OPENAI_API_KEY=<your-openai-api-key>"
            )


class ClovaOCRConfig(BaseModel):
    """CLOVA OCR configuration"""
    url: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_URL", ""))
    secret_key: str = Field(default_factory=lambda: os.environ.get("CLOVA_OCR_SECRET", ""))

    def validate_credentials(self) -> None:
        """Validate that CLOVA OCR credentials are configured. Raises ValueError if missing."""
        missing = []
        if not self.url:
            missing.append("CLOVA_OCR_URL")
        if not self.secret_key:
            missing.append("CLOVA_OCR_SECRET")
        if missing:
            raise ValueError(
                f"CLOVA OCR 환경변수가 설정되지 않았습니다: {', '.join(missing)}\n"
                f".env 파일에 다음 변수를 설정하세요:\n"
                f"  CLOVA_OCR_URL=<your-clova-ocr-url>\n"
                f"  CLOVA_OCR_SECRET=<your-clova-ocr-secret>"
            )


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
