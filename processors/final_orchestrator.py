"""
Final Orchestrator for generating complete markdown document from all pages
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from prompts import get_final_document_prompt
from utils.config import LLMConfig

SYSTEM_PROMPT = "You are a markdown formatting expert. Preserve ALL content while creating well-formatted documents."


def combine_page_contents(page_results: List[Dict[str, Any]]) -> str:
    """Combine all page contents with page markers"""
    combined = []
    for result in page_results:
        page_num = result.get('page_number', 0)
        content = result.get('content', '')
        if content.strip():
            combined.append(f"[PAGE {page_num}]")
            combined.append(content)
            combined.append("")
    return "\n".join(combined)


def generate_metadata(
    page_results: List[Dict[str, Any]],
    source_file: str,
) -> Dict[str, Any]:
    """Generate document metadata"""
    return {
        'source_file': source_file,
        'total_pages': len(page_results),
        'total_processing_time': sum(
            r.get('metadata', {}).get('processing_time', 0)
            for r in page_results
        ),
        'successful_pages': sum(
            1 for r in page_results if not r.get('error')
        ),
    }


def estimate_tokens(text: str) -> int:
    """Estimate token count for Korean/English mixed text"""
    if not text:
        return 0
    korean_chars = sum(
        1 for c in text
        if '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ'
    )
    total_chars = len(text)
    if total_chars == 0:
        return 0
    korean_ratio = korean_chars / total_chars
    estimated_tokens = (korean_ratio * 1.5 + (1 - korean_ratio) * 0.25) * total_chars
    return int(estimated_tokens * 1.2)


def calculate_dynamic_max_tokens(prompt: str, config: LLMConfig) -> int:
    """Calculate dynamic max_tokens based on input size"""
    if not config.dynamic_token_adjustment:
        return config.max_tokens
    estimated_output = estimate_tokens(prompt)
    required_tokens = max(estimated_output * 2, config.max_tokens)
    adjusted_tokens = min(required_tokens, config.max_tokens_limit)
    if adjusted_tokens > config.max_tokens:
        logger.info(
            f"📊 Dynamic token adjustment: {config.max_tokens} → {adjusted_tokens}"
        )
    return adjusted_tokens


def call_llm_for_final_generation(
    prompt: str,
    config: LLMConfig,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Call LLM for final document generation with dynamic token adjustment"""
    max_tokens = calculate_dynamic_max_tokens(prompt, config)

    if config.provider == "anthropic":
        return _call_claude(prompt, max_tokens, config, anthropic_client)
    else:
        return _call_openai(prompt, max_tokens, config, openai_client)


def _call_claude(
    prompt: str,
    max_tokens: int,
    config: LLMConfig,
    client: Anthropic,
) -> str:
    """Call Claude API for final generation"""
    try:
        message = client.messages.create(
            model=config.claude_model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "text":
                return block.text
        return ""
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        raise


def _call_openai(
    prompt: str,
    max_tokens: int,
    config: LLMConfig,
    client: OpenAI,
) -> str:
    """Call OpenAI API for final generation"""
    try:
        completion_params = {
            "model": config.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if "gpt-5" in config.openai_model.lower():
            completion_params["max_completion_tokens"] = max_tokens
            completion_params["reasoning_effort"] = "high"
        else:
            completion_params["max_tokens"] = max_tokens
            completion_params["temperature"] = 0.3

        response = client.chat.completions.create(**completion_params)
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def post_process_markdown(markdown: str) -> str:
    """Post-process the generated markdown"""
    markdown = re.sub(r'\[PAGE \d+\]', '', markdown)
    return markdown.strip()


def fallback_generation(
    page_results: List[Dict[str, Any]],
    source_file: str,
) -> str:
    """Fallback generation without LLM"""
    logger.warning("Using fallback generation without LLM formatting")
    stem = Path(source_file).stem

    lines = [f"# {stem}", ""]
    for result in page_results:
        page_num = result.get('page_number', 0)
        content = result.get('content', '')
        if content.strip():
            lines.append(f"## Page {page_num}")
            lines.append("")
            lines.append(content)
            lines.append("")

    lines.extend([
        "---",
        f"*Extracted from: {source_file}*",
        f"*Total pages: {len(page_results)}*",
    ])

    return "\n".join(lines)


def generate_final_document(
    page_results: List[Dict[str, Any]],
    source_file: str,
    config: LLMConfig,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Generate final markdown document from all page results"""
    if not page_results:
        logger.warning("No page results to process")
        return "# Empty Document\n\nNo content was extracted from the PDF."

    try:
        logger.info(f"📝 Generating final markdown from {len(page_results)} pages")

        combined_content = combine_page_contents(page_results)
        metadata = generate_metadata(page_results, source_file)
        prompt = get_final_document_prompt(metadata, combined_content)
        final_markdown = call_llm_for_final_generation(
            prompt, config, anthropic_client, openai_client,
        )
        final_markdown = post_process_markdown(final_markdown)

        logger.info(f"✅ Final markdown generated: {len(final_markdown)} characters")
        return final_markdown

    except Exception as e:
        logger.error(f"Final document generation failed: {e}")
        return fallback_generation(page_results, source_file)
