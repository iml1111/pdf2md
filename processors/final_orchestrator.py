"""
Final Orchestrator for generating complete markdown document from all pages
"""

from typing import Dict, Any, List, Optional
import re

from anthropic import Anthropic
from openai import OpenAI

from utils.config import get_config, Config
from utils.logger import logger
from prompts import get_final_document_prompt


class FinalOrchestrator:
    """Generate final markdown document from all integrated pages"""
    
    # System prompt for markdown formatting
    SYSTEM_PROMPT = "You are a markdown formatting expert. Preserve ALL content while creating well-formatted documents."
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize final orchestrator with LLM client"""
        self.config = config or get_config()
        
        # Initialize both LLM clients for flexibility
        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)
        
        # Log provider preference
        if self.config.llm.provider == "anthropic":
            logger.info(f"✅ Final orchestrator preferring Claude: {self.config.llm.claude_model}")
        else:
            logger.info(f"✅ Final orchestrator preferring OpenAI: {self.config.llm.openai_model}")
    
    def generate_final_document(self, page_results: List[Dict[str, Any]], pdf_path: str) -> str:
        """
        Generate final markdown document from all page results
        
        Args:
            page_results: List of integrated page results
            pdf_path: Original PDF path for metadata
            
        Returns:
            Final markdown document with proper formatting
        """
        if not page_results:
            logger.warning("No page results to process")
            return "# Empty Document\n\nNo content was extracted from the PDF."
        
        try:
            logger.info(f"📝 Generating final markdown from {len(page_results)} pages")
            
            # Combine all page contents
            combined_content = self._combine_page_contents(page_results)
            
            # Generate metadata section
            metadata = self._generate_metadata(page_results, pdf_path)
            
            # Create final generation prompt using centralized prompts
            prompt = get_final_document_prompt(metadata, combined_content)
            
            # Call LLM for final markdown generation
            final_markdown = self._call_llm_for_final_generation(prompt)
            
            # Post-process markdown
            final_markdown = self._post_process_markdown(final_markdown, metadata)
            
            logger.info(f"✅ Final markdown generated: {len(final_markdown)} characters")
            return final_markdown
            
        except Exception as e:
            logger.error(f"Final document generation failed: {e}")
            # Fallback to simple concatenation
            return self._fallback_generation(page_results, pdf_path)
    
    def _combine_page_contents(self, page_results: List[Dict[str, Any]]) -> str:
        """Combine all page contents with page markers"""
        combined = []
        
        for result in page_results:
            page_num = result.get('page_number', 0)
            content = result.get('content', '')
            
            if content.strip():
                # Add page marker for reference
                combined.append(f"[PAGE {page_num}]")
                combined.append(content)
                combined.append("")  # Empty line between pages
        
        return "\n".join(combined)
    
    def _generate_metadata(self, page_results: List[Dict[str, Any]], pdf_path: str) -> Dict[str, Any]:
        """Generate document metadata"""
        from pathlib import Path
        
        total_pages = len(page_results)
        total_processing_time = sum(r.get('metadata', {}).get('processing_time', 0) for r in page_results)
        
        return {
            'source_file': Path(pdf_path).name,
            'total_pages': total_pages,
            'total_processing_time': total_processing_time,
            'successful_pages': sum(1 for r in page_results if not r.get('error'))
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Korean/English mixed text
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Count Korean characters
        korean_chars = sum(1 for c in text if '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ')
        total_chars = len(text)
        
        if total_chars == 0:
            return 0
            
        korean_ratio = korean_chars / total_chars
        
        # Korean: ~1.5 tokens/char, English/others: ~0.25 tokens/char
        # Add 20% buffer for safety
        estimated_tokens = (korean_ratio * 1.5 + (1 - korean_ratio) * 0.25) * total_chars
        return int(estimated_tokens * 1.2)
    
    def _calculate_dynamic_max_tokens(self, prompt: str) -> int:
        """
        Calculate dynamic max_tokens based on input size
        
        Args:
            prompt: The input prompt
            
        Returns:
            Adjusted max_tokens value
        """
        if not self.config.llm.dynamic_token_adjustment:
            return self.config.llm.max_tokens
        
        # Estimate tokens needed for output (usually similar to input for translation/formatting tasks)
        estimated_output = self._estimate_tokens(prompt)
        
        # We need at least 2x the input size for safety
        required_tokens = max(estimated_output * 2, self.config.llm.max_tokens)
        
        # Limit to maximum allowed
        adjusted_tokens = min(required_tokens, self.config.llm.max_tokens_limit)
        
        if adjusted_tokens > self.config.llm.max_tokens:
            logger.info(f"📊 Dynamic token adjustment: {self.config.llm.max_tokens} → {adjusted_tokens}")
        
        return adjusted_tokens
    
    def _call_llm_for_final_generation(self, prompt: str) -> str:
        """Call LLM for final document generation with dynamic token adjustment"""
        # Calculate dynamic max_tokens
        max_tokens = self._calculate_dynamic_max_tokens(prompt)
        
        if self.config.llm.provider == "anthropic":
            return self._call_claude(prompt, max_tokens)
        else:
            return self._call_openai(prompt, max_tokens)
    
    def _call_claude(self, prompt: str, max_tokens: int = None) -> str:
        """Call Claude API for final generation"""
        try:
            if max_tokens is None:
                max_tokens = self.config.llm.max_tokens
                
            message = self.anthropic_client.messages.create(
                model=self.config.llm.claude_model,
                max_tokens=max_tokens,
                temperature=0.3,  # Slightly higher for better formatting
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
    
    def _call_openai(self, prompt: str, max_tokens: int = None) -> str:
        """Call OpenAI API for final generation"""
        try:
            if max_tokens is None:
                max_tokens = self.config.llm.max_tokens
                
            completion_params = {
                "model": self.config.llm.openai_model,
                "messages": [{
                    "role": "system",
                    "content": self.SYSTEM_PROMPT  # Use the same system prompt constant
                }, {
                    "role": "user",
                    "content": prompt
                }]
            }
            
            # Handle different OpenAI model parameters
            if "gpt-5" in self.config.llm.openai_model.lower():
                completion_params["max_completion_tokens"] = max_tokens
                # GPT-5 only supports temperature=1, so don't set it
            else:
                completion_params["max_tokens"] = max_tokens
                completion_params["temperature"] = 0.3  # Slightly higher for better formatting
            
            response = self.openai_client.chat.completions.create(**completion_params)
            
            return response.choices[0].message.content if response.choices else ""
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _post_process_markdown(self, markdown: str, metadata: Dict[str, Any]) -> str:
        """Post-process the generated markdown"""
        # Remove any remaining page markers
        markdown = re.sub(r'\[PAGE \d+\]', '', markdown)
        
        # Ensure proper line endings
        markdown = markdown.strip()
        
        return markdown
    
    def _fallback_generation(self, page_results: List[Dict[str, Any]], pdf_path: str) -> str:
        """Fallback generation without LLM"""
        from pathlib import Path
        
        logger.warning("Using fallback generation without LLM formatting")
        
        lines = [f"# {Path(pdf_path).stem}", ""]
        
        for result in page_results:
            page_num = result.get('page_number', 0)
            content = result.get('content', '')
            
            if content.strip():
                lines.append(f"## Page {page_num}")
                lines.append("")
                lines.append(content)
                lines.append("")
        
        # Add metadata footer
        lines.extend([
            "---",
            f"*Extracted from: {Path(pdf_path).name}*",
            f"*Total pages: {len(page_results)}*"
        ])
        
        return "\n".join(lines)