"""
LLM-based text merger for combining extraction results
"""

import asyncio
from typing import Any, Dict

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from prompts import (
    format_extraction_data,
    get_llm_merge_prompt,
)
from utils.config import LLMConfig
from utils.rate_limiter import RateLimiter


def filter_valid_results(
    extraction_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filter out empty or error results (text-based only)"""
    valid = {}
    for name, result in extraction_results.items():
        if result.get('text') and not result.get('error'):
            valid[name] = result
    return valid


async def call_llm_for_merge(
    prompt: str,
    config: LLMConfig,
    rate_limiter: RateLimiter,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Call LLM API with rate limiting for merging"""
    try:
        await rate_limiter.acquire()

        if config.provider == "anthropic":
            response = await asyncio.to_thread(
                anthropic_client.messages.create,
                model=config.claude_model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        elif config.provider == "openai":
            completion_params = {
                "model": config.openai_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if "gpt-5" in config.openai_model.lower():
                completion_params["reasoning_effort"] = "high"
            else:
                completion_params["temperature"] = 0.1

            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                **completion_params,
            )
            return response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM merge failed: {e}")
        return fallback_merge(prompt)


def fallback_merge(prompt: str) -> str:
    """Fallback merge when LLM is unavailable"""
    lines = prompt.split('\n')
    in_extraction = False
    texts = []

    for line in lines:
        if '=== ' in line and 'EXTRACTION ===' in line:
            in_extraction = True
            continue
        elif line.startswith('IMPORTANT INSTRUCTIONS:'):
            break
        elif in_extraction and line.strip():
            texts.append(line)

    return '\n'.join(texts)


async def merge_text(
    extraction_results: Dict[str, Dict[str, Any]],
    config: LLMConfig,
    rate_limiter: RateLimiter,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
) -> str:
    """Merge text from multiple extractors using LLM intelligence"""
    if not extraction_results:
        return ''

    valid_results = filter_valid_results(extraction_results)

    if not valid_results and 'pymupdf' not in extraction_results:
        logger.warning("No valid extraction results to merge")
        return ''

    if len(valid_results) == 1 and 'pymupdf' not in extraction_results:
        result = next(iter(valid_results.values()))
        return result.get('text', '')

    all_results_for_format = dict(valid_results)
    if 'pymupdf' in extraction_results:
        all_results_for_format['pymupdf'] = extraction_results['pymupdf']

    extraction_data = format_extraction_data(all_results_for_format)
    prompt = get_llm_merge_prompt(extraction_data)
    merged_text = await call_llm_for_merge(
        prompt, config, rate_limiter, anthropic_client, openai_client,
    )

    return merged_text


def extract_metadata(
    extraction_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract and combine metadata from extraction results"""
    valid_results = filter_valid_results(extraction_results)

    metadata = {
        'extractors_used': list(valid_results.keys()),
        'extraction_details': {},
    }

    for name, result in valid_results.items():
        if result.get('metadata'):
            metadata['extraction_details'][name] = result['metadata']

    return metadata


def get_valid_sources(
    extraction_results: Dict[str, Dict[str, Any]],
) -> list[str]:
    """Get list of valid extraction sources"""
    return list(filter_valid_results(extraction_results).keys())
