"""
LLM (Claude/GPT) based multimodal image text extractor
"""

import base64
import io
import json
from typing import Any, Dict, List

from PIL import Image
from anthropic import Anthropic
from loguru import logger
from openai import OpenAI

from prompts import SINGLE_PAGE_IMAGE_PROMPT
from utils.config import Config


class LLMExtractor:
    """Extract text and structure using LLM multimodal capabilities"""

    def __init__(self, config: Config):
        """Initialize LLM extractor with client based on configuration"""
        self.name: str = "LLM"
        self.config = config

        self.anthropic_client = Anthropic(api_key=self.config.llm.anthropic_api_key)
        self.openai_client = OpenAI(api_key=self.config.llm.openai_api_key)

    def _resize_image_if_needed(self, image_bytes: bytes, max_dimension: int = 7999) -> bytes:
        """Resize image if it exceeds the maximum dimension (default 7999px for Claude API)"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True)
                return output.getvalue()

            return image_bytes

        except Exception as e:
            logger.warning(f"Failed to resize image: {e}. Using original image.")
            return image_bytes

    def _call_llm_image(self, img_base64: str, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Call LLM API for image analysis"""
        if self.config.llm.provider == "anthropic":
            return self._call_claude_image(img_base64, prompt)
        else:
            return self._call_openai_image(img_base64, prompt)

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

            if "gpt-5" in self.config.llm.openai_model.lower():
                completion_params["max_completion_tokens"] = self.config.llm.max_tokens
            else:
                completion_params["max_tokens"] = self.config.llm.max_tokens
                completion_params["temperature"] = self.config.llm.temperature

            response = self.openai_client.chat.completions.create(**completion_params)

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
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            else:
                return {
                    'text': response_text,
                    'structure': {}
                }
        except json.JSONDecodeError:
            return {
                'text': response_text,
                'structure': {}
            }

    def extract_single_page_image(self, image_bytes: bytes, page_number: int, total_pages: int) -> Dict[str, Any]:
        """Extract text from a single page image using LLM"""
        try:
            resized_image_bytes = self._resize_image_if_needed(image_bytes)
            img_base64 = base64.b64encode(resized_image_bytes).decode('utf-8')

            prompt = SINGLE_PAGE_IMAGE_PROMPT.format(
                page_number=page_number,
                total_pages=total_pages
            )

            result = self._call_llm_image(img_base64, image_bytes, prompt)
            result['page_number'] = page_number

            return result

        except Exception as e:
            logger.error(f"LLM single page image extraction failed for page {page_number}: {e}")
            return {'text': '', 'error': str(e), 'page_number': page_number}

    def extract_from_images(self, images: List[bytes]) -> Dict[str, Any]:
        """Legacy method: Extract text from multiple images using LLM"""
        if not images:
            return {'text': '', 'error': 'No images provided'}

        try:
            if len(images) == 1:
                return self.extract_single_page_image(images[0], page_number=1, total_pages=1)

            all_text = []
            combined_results = {
                'text': '',
                'structure': {},
                'extraction_mode': 'multipage_images',
                'page_count': len(images)
            }

            for i, image_bytes in enumerate(images, 1):
                page_result = self.extract_single_page_image(image_bytes, page_number=i, total_pages=len(images))
                if page_result.get('text'):
                    all_text.append(f"=== Page {i} ===\n{page_result['text']}")

            combined_results['text'] = '\n\n'.join(all_text)
            combined_results['total_chars'] = len(combined_results['text'])

            return combined_results

        except Exception as e:
            logger.error(f"LLM images extraction failed: {e}")
            return {'text': '', 'error': str(e)}
