# usecases/merging.py
"""Page merge usecase function"""

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from processors.llm_merger import merge_text
from usecases.models import ExtractionResult, MergeInput, MergeResult
from utils.config import Config
from utils.rate_limiter import APIRateLimiters


def _to_results_dict(
    extraction_results: list[ExtractionResult],
) -> dict[str, dict]:
    """Convert list[ExtractionResult] to Dict[str, Dict] for prompts.py compatibility"""
    results_dict = {}
    for r in extraction_results:
        entry = {'text': r.text}
        if r.tables:
            entry['tables'] = r.tables
        if r.hyperlinks:
            entry['hyperlinks'] = r.hyperlinks
        if r.metadata:
            entry['metadata'] = r.metadata
        if r.error:
            entry['error'] = r.error
        results_dict[r.extractor_name] = entry
    return results_dict


async def merge_page(
    merge_input: MergeInput,
    config: Config,
    rate_limiters: APIRateLimiters,
) -> MergeResult:
    """4개 추출 결과를 LLM으로 병합"""
    try:
        results_dict = _to_results_dict(merge_input.extraction_results)

        anthropic_client = Anthropic(api_key=config.llm.anthropic_api_key)
        openai_client = OpenAI(api_key=config.llm.openai_api_key)
        rate_limiter = rate_limiters.get_limiter(config.llm.provider)

        merged_text = await merge_text(
            results_dict,
            config.llm,
            rate_limiter,
            anthropic_client,
            openai_client,
        )

        return MergeResult(
            page_number=merge_input.page_number,
            merged_text=merged_text,
        )
    except Exception as e:
        logger.error(
            f"Merge failed for page {merge_input.page_number}: {e}"
        )
        return MergeResult(
            page_number=merge_input.page_number,
            merged_text="",
            error=str(e),
        )
