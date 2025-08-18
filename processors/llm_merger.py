"""
LLM-based text merger for combining extraction results
"""

from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from openai import OpenAI

from utils.config import get_config, Config
from utils.logger import logger
from utils.rate_limiter import APIRateLimiters
from prompts import get_llm_merge_prompt, format_extraction_data
import asyncio


class LLMMerger:
    """Text merger for combining multiple extraction results"""
    
    def __init__(self):
        """Initialize LLM merger with all configurations"""
        self.config: Config = get_config()
        self.rate_limiters: APIRateLimiters = APIRateLimiters()
        
        # Initialize both LLM clients for flexibility
        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)
        
        # Set provider preference from config
        self.provider: str = self.config.llm.provider
        if self.provider == "anthropic":
            self.model: str = self.config.llm.claude_model
            logger.info(f"LLM Merger preferring Claude: {self.model}")
        else:
            self.model: str = self.config.llm.openai_model
            logger.info(f"LLM Merger preferring OpenAI: {self.model}")
    
    async def merge_text(self, extraction_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Merge text from multiple extractors using LLM intelligence
        
        Args:
            extraction_results: Dictionary mapping extractor name to results
            
        Returns:
            Merged text string
        """
        if not extraction_results:
            return ''
        
        # Filter out empty or error results
        valid_results = self._filter_valid_results(extraction_results)
        
        if not valid_results:
            logger.warning("No valid extraction results to merge")
            return ''
        
        # If only one valid result, return its text directly
        if len(valid_results) == 1:
            result = next(iter(valid_results.values()))
            return result.get('text', '')
        
        # Prepare extraction data for LLM using centralized formatter
        extraction_data = format_extraction_data(valid_results)
        
        # Create merging prompt using centralized prompts
        prompt = get_llm_merge_prompt(extraction_data)
        
        # Call LLM with rate limiting
        merged_text = await self._call_llm_for_merge(prompt)
        
        return merged_text
    
    def extract_metadata(self, extraction_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and combine metadata from extraction results
        
        Args:
            extraction_results: Dictionary mapping extractor name to results
            
        Returns:
            Combined metadata dictionary
        """
        valid_results = self._filter_valid_results(extraction_results)
        
        metadata = {
            'extractors_used': list(valid_results.keys()),
            'extraction_details': {}
        }
        
        for name, result in valid_results.items():
            if result.get('metadata'):
                metadata['extraction_details'][name] = result['metadata']
        
        return metadata
    
    def get_valid_sources(self, extraction_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Get list of valid extraction sources
        
        Args:
            extraction_results: Dictionary mapping extractor name to results
            
        Returns:
            List of valid extractor names
        """
        valid_results = self._filter_valid_results(extraction_results)
        return list(valid_results.keys())
    
    def _filter_valid_results(self, extraction_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter out empty or error results
        
        Args:
            extraction_results: Dictionary mapping extractor name to results
            
        Returns:
            Dictionary containing only valid results
        """
        valid_results = {}
        for name, result in extraction_results.items():
            if result.get('text') and not result.get('error'):
                valid_results[name] = result
        return valid_results
    
    async def _call_llm_for_merge(self, prompt: str) -> str:
        """
        Call LLM API with rate limiting for merging
        
        Args:
            prompt: Merge prompt
            
        Returns:
            Merged text from LLM
        """
        limiter = self.rate_limiters.get_limiter(self.provider)
        
        try:
            # Acquire rate limit
            await limiter.acquire()
            
            if self.provider == "anthropic":
                # Use asyncio.to_thread for sync API call
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == "openai":
                # Build parameters for OpenAI
                completion_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # GPT-5 only supports temperature=1, so don't set it for GPT-5
                if "gpt-5" not in self.model.lower():
                    completion_params["temperature"] = 0.1
                
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    **completion_params
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback to simple concatenation
            return self._fallback_merge(prompt)
    
    def _fallback_merge(self, prompt: str) -> str:
        """
        Fallback merge when LLM is unavailable
        
        Args:
            prompt: Original merge prompt (contains extraction data)
            
        Returns:
            Simple concatenated text
        """
        # Extract the extraction data from prompt
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