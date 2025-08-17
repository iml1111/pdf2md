"""
Page Orchestrator for integrating single page extraction results
"""

from typing import Dict, Any, Optional
import json

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from utils.config import get_config, Config
from utils.logger import logger
from prompts import get_page_integration_prompt


class PageOrchestrator:
    """Orchestrate page-level integration without markdown formatting"""
    
    # System prompt for text integration
    SYSTEM_PROMPT = "You are a precise text integration assistant. Preserve all content without adding formatting."
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize page orchestrator with LLM client"""
        self.config = config or get_config()
        
        # Initialize LLM client based on configuration
        # API keys are already validated during Config creation
        if self.config.llm.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.config.llm.anthropic_api_key)
            logger.debug(f"Page orchestrator using Claude: {self.config.llm.claude_model}")
        elif self.config.llm.provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.config.llm.openai_api_key)
            logger.debug(f"Page orchestrator using OpenAI: {self.config.llm.openai_model}")
        else:
            raise ValueError(f"LLM provider {self.config.llm.provider} not available")
    
    def integrate_page_results(self, merged_result: Dict[str, Any], page_number: int, total_pages: int) -> Dict[str, Any]:
        """
        Integrate page results using LLM without markdown formatting
        
        Args:
            merged_result: Merged extraction result from TextMerger
            page_number: Current page number
            total_pages: Total number of pages
            
        Returns:
            Integrated page content without formatting
        """
        if not self.client:
            error_msg = "LLM client not initialized - API key missing or invalid"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Prepare page context
            context = {
                'page_number': page_number,
                'total_pages': total_pages,
                'text': merged_result.get('text', ''),
                'sources': merged_result.get('sources', []),
                'structure': merged_result.get('structure', {})
            }
            
            # Create integration prompt using centralized prompts
            prompt = get_page_integration_prompt(
                page_number=page_number,
                total_pages=total_pages,
                text=context['text'],
                sources=context['sources']
            )
            
            # Call LLM for page integration
            integrated_content = self._call_llm_for_integration(prompt, context)
            
            # Return integrated result
            result = {
                'page_number': page_number,
                'content': integrated_content,
                'metadata': {
                    'total_pages': total_pages,
                    'successful_extractors': merged_result.get('successful_extractors', 0),
                    'processing_time': merged_result.get('processing_time', 0)
                }
            }
            
            logger.debug(f"Page {page_number} integrated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Page integration failed for page {page_number}: {e}")
            # Fallback to raw merged text
            return {
                'page_number': page_number,
                'content': merged_result.get('text', ''),
                'error': str(e)
            }
    
    def _call_llm_for_integration(self, prompt: str, context: Dict[str, Any]) -> str:
        """Call LLM for page integration"""
        if self.config.llm.provider == "anthropic":
            return self._call_claude(prompt)
        else:
            return self._call_openai(prompt)
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API for integration"""
        try:
            message = self.client.messages.create(
                model=self.config.llm.claude_model,
                max_tokens=self.config.llm.max_tokens,
                temperature=0.1,  # Low temperature for consistency
                system=self.SYSTEM_PROMPT,  # Use system parameter for Claude
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return message.content[0].text if message.content else ""
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for integration"""
        try:
            # Check if client is actually Anthropic (misconfigured)
            if isinstance(self.client, Anthropic):
                # Fallback to Anthropic call
                return self._call_claude(prompt)
            
            completion_params = {
                "model": self.config.llm.openai_model,
                "messages": [{
                    "role": "system",
                    "content": self.SYSTEM_PROMPT  # Use the same system prompt constant
                }, {
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.1  # Low temperature for consistency
            }
            
            # Handle different OpenAI model parameters
            if "gpt-5" in self.config.llm.openai_model.lower():
                completion_params["max_completion_tokens"] = self.config.llm.max_tokens
            else:
                completion_params["max_tokens"] = self.config.llm.max_tokens
            
            response = self.client.chat.completions.create(**completion_params)
            
            return response.choices[0].message.content if response.choices else ""
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise