"""
LLM (Claude/GPT) based multimodal PDF/Image text extractor
"""

import base64
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import io
from PIL import Image

from anthropic import Anthropic
from openai import OpenAI

from utils.logger import logger, log_extraction_result
from utils.config import get_config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import (
    PDF_EXTRACTION_PROMPT, 
    IMAGE_EXTRACTION_PROMPT,
    SINGLE_PAGE_PDF_PROMPT,
    SINGLE_PAGE_IMAGE_PROMPT
)


class LLMExtractor:
    """Extract text and structure using LLM multimodal capabilities"""
    
    def __init__(self):
        """Initialize LLM extractor with client based on configuration"""
        self.name: str = "LLM"
        self.config = get_config()
        
        # Initialize both LLM clients for flexibility
        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)
        logger.info(f"✅ Both LLM clients initialized - Claude: {self.config.llm.claude_model}, OpenAI: {self.config.llm.openai_model}")
    
    def _resize_image_if_needed(self, image_bytes: bytes, max_dimension: int = 7999) -> bytes:
        """
        Resize image if it exceeds the maximum dimension (default 7999px for Claude API)
        
        Args:
            image_bytes: Original image bytes
            max_dimension: Maximum allowed dimension (default 7999)
            
        Returns:
            Resized image bytes if needed, otherwise original bytes
        """
        try:
            # Open image from bytes
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            # Check if resizing is needed
            if max(width, height) > max_dimension:
                # Calculate scale factor to fit within max_dimension
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                logger.info(f"📐 Resizing image from {width}x{height} to {new_width}x{new_height} (Claude API limit)")
                
                # Resize image with high quality resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                output = io.BytesIO()
                # Save as PNG to maintain quality
                img.save(output, format='PNG', optimize=True)
                return output.getvalue()
            
            return image_bytes
            
        except Exception as e:
            logger.warning(f"Failed to resize image: {e}. Using original image.")
            return image_bytes
    
    def _call_llm_image(self, img_base64: str, prompt: str) -> Dict[str, Any]:
        """Call LLM API for image analysis"""
        if self.config.llm.provider == "anthropic":
            return self._call_claude_image(img_base64, prompt)
        else:
            return self._call_openai_image(img_base64, prompt)
    
    def _call_claude_pdf(self, pdf_base64: str, prompt: str) -> Dict[str, Any]:
        """Call Claude API for PDF analysis"""
        try:
            message = self.anthropic_client.messages.create(
                model=self.config.llm.claude_model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response - for PDF, we expect plain text
            response_text = message.content[0].text if message.content else ""
            # For PDF extraction, treat as plain text directly
            return {
                'text': response_text,
                'structure': {},
                'extraction_mode': 'full_text',
                'llm_model': self.config.llm.claude_model,
                'llm_provider': 'Claude (Anthropic)'
            }
            
        except Exception as e:
            logger.error(f"Claude PDF API call failed: {e}")
            return {'text': '', 'error': str(e)}
    
    def _call_claude_image(self, img_base64: str, prompt: str) -> Dict[str, Any]:
        """Call Claude API for image analysis"""
        try:
            message = self.anthropic_client.messages.create(
                model=self.config.llm.claude_model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            response_text = message.content[0].text if message.content else ""
            result = self._parse_llm_response(response_text)
            result['llm_model'] = self.config.llm.claude_model
            result['llm_provider'] = 'Claude (Anthropic)'
            return result
            
        except Exception as e:
            logger.error(f"Claude Image API call failed: {e}")
            return {'text': '', 'error': str(e)}
    
    
    def _call_openai_image(self, img_base64: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API for image analysis"""
        try:
            # Build parameters
            completion_params = {
                "model": self.config.llm.openai_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }]
            }
            
            # GPT-5 has different parameter requirements
            if "gpt-5" in self.config.llm.openai_model.lower():
                completion_params["max_completion_tokens"] = self.config.llm.max_tokens
                # GPT-5 only supports temperature=1
            else:
                completion_params["max_tokens"] = self.config.llm.max_tokens
                completion_params["temperature"] = self.config.llm.temperature
            
            response = self.openai_client.chat.completions.create(**completion_params)
            
            # Parse response
            response_text = response.choices[0].message.content if response.choices else ""
            result = self._parse_llm_response(response_text)
            result['llm_model'] = self.config.llm.openai_model
            result['llm_provider'] = 'OpenAI'
            return result
            
        except Exception as e:
            logger.error(f"OpenAI Image API call failed: {e}")
            return {'text': '', 'error': str(e)}
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            # Try to parse as JSON
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            else:
                # Fallback to simple text extraction
                return {
                    'text': response_text,
                    'structure': {}
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, return as plain text
            return {
                'text': response_text,
                'structure': {}
            }
    
    def extract_single_page_pdf(self, pdf_bytes: bytes, page_number: int, total_pages: int) -> Dict[str, Any]:
        """
        Extract text from a single PDF page using LLM
        
        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Current page number
            total_pages: Total number of pages
            
        Returns:
            Dictionary containing extracted text for single page
        """
        try:
            # Encode PDF to base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Prepare page-specific prompt
            prompt = SINGLE_PAGE_PDF_PROMPT.format(
                page_number=page_number,
                total_pages=total_pages
            )
            
            # Call Claude for single page (OpenAI doesn't support direct PDF processing)
            result = self._call_claude_pdf(pdf_base64, prompt)
            result['page_number'] = page_number
            
            return result
            
        except Exception as e:
            logger.error(f"LLM single page PDF extraction failed for page {page_number}: {e}")
            return {'text': '', 'error': str(e), 'page_number': page_number}
    
    def extract_single_page_image(self, image_bytes: bytes, page_number: int, total_pages: int) -> Dict[str, Any]:
        """
        Extract text from a single page image using LLM
        
        Args:
            image_bytes: Image bytes for single page
            page_number: Current page number
            total_pages: Total number of pages
            
        Returns:
            Dictionary containing extracted text for single page
        """
        try:
            # Resize image if needed to comply with API limits
            resized_image_bytes = self._resize_image_if_needed(image_bytes)
            
            # Encode image to base64
            img_base64 = base64.b64encode(resized_image_bytes).decode('utf-8')
            
            # Prepare page-specific prompt
            prompt = SINGLE_PAGE_IMAGE_PROMPT.format(
                page_number=page_number,
                total_pages=total_pages
            )
            
            # Call LLM for single image
            result = self._call_llm_image(img_base64, prompt)
            result['page_number'] = page_number
            
            return result
            
        except Exception as e:
            logger.error(f"LLM single page image extraction failed for page {page_number}: {e}")
            return {'text': '', 'error': str(e), 'page_number': page_number}
    
    def extract_from_images(self, images: List[bytes]) -> Dict[str, Any]:
        """
        Legacy method: Extract text from multiple images using LLM
        
        Args:
            images: List of image bytes
            
        Returns:
            Dictionary containing extracted text from all images
        """
        if not images:
            return {'text': '', 'error': 'No images provided'}
        
        try:
            # For legacy compatibility, process first image with generic prompt
            # This is used by single_page_pipeline for individual page processing
            if len(images) == 1:
                # Single image - treat as page 1 of 1 for consistency
                return self.extract_single_page_image(images[0], page_number=1, total_pages=1)
            
            # Multiple images - process all and combine results
            all_text = []
            combined_results = {
                'text': '',
                'structure': {},
                'extraction_mode': 'multipage_images',
                'page_count': len(images)
            }
            
            for i, image_bytes in enumerate(images, 1):
                # extract_single_page_image already handles resizing internally
                page_result = self.extract_single_page_image(image_bytes, page_number=i, total_pages=len(images))
                if page_result.get('text'):
                    all_text.append(f"=== Page {i} ===\n{page_result['text']}")
            
            combined_results['text'] = '\n\n'.join(all_text)
            combined_results['total_chars'] = len(combined_results['text'])
            
            return combined_results
            
        except Exception as e:
            logger.error(f"LLM images extraction failed: {e}")
            return {'text': '', 'error': str(e)}