# usecases/finalizing.py
"""Document finalize usecase function"""

import asyncio

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from processors.final_orchestrator import (
    fallback_generation,
    generate_final_document,
)
from usecases.models import FinalizeInput, FinalizeResult
from utils.config import Config


async def finalize_document(
    finalize_input: FinalizeInput,
    config: Config,
) -> FinalizeResult:
    """병합된 페이지들을 최종 마크다운으로 생성"""
    try:
        # Convert MergeResult list to dict list for processor compatibility
        page_results = [
            {
                'page_number': mr.page_number,
                'content': mr.merged_text,
                'error': mr.error,
            }
            for mr in sorted(
                finalize_input.merge_results,
                key=lambda x: x.page_number,
            )
        ]

        anthropic_client = Anthropic(api_key=config.llm.anthropic_api_key)
        openai_client = OpenAI(api_key=config.llm.openai_api_key)

        # generate_final_document is synchronous — wrap in to_thread
        markdown = await asyncio.to_thread(
            generate_final_document,
            page_results,
            finalize_input.source_file,
            config.llm,
            anthropic_client,
            openai_client,
        )

        return FinalizeResult(
            markdown=markdown,
            metadata={
                'total_pages': finalize_input.total_pages,
                'source_file': finalize_input.source_file,
            },
        )
    except Exception as e:
        logger.error(f"Finalize failed: {e}")

        # Fallback
        page_results = [
            {
                'page_number': mr.page_number,
                'content': mr.merged_text,
            }
            for mr in finalize_input.merge_results
        ]
        fallback_md = fallback_generation(
            page_results, finalize_input.source_file,
        )
        return FinalizeResult(
            markdown=fallback_md,
            metadata={'total_pages': finalize_input.total_pages},
            error=str(e),
        )
