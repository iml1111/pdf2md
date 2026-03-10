"""
CLOVA OCR based text extractor
"""

import json
import time
from typing import Optional

import aiohttp
from loguru import logger

from utils.config import ClovaOCRConfig


class ClovaOCRExtractor:
    """Extract text using CLOVA OCR API"""

    def __init__(self, config: ClovaOCRConfig):
        self.name = "CLOVA OCR"
        self.config = config

    async def extract_text(self, page_pdf_bytes: bytes, page_number: int) -> Optional[str]:
        """
        Extract text from a single PDF page using CLOVA OCR API

        Args:
            page_pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference

        Returns:
            Extracted text string, or None on failure
        """
        if not self.config.url or not self.config.secret_key:
            logger.warning("CLOVA OCR credentials not configured")
            return None

        try:
            clova_message = {
                "version": "V2",
                "requestId": f"pdf2md_page_{page_number}",
                "timestamp": int(time.time()),
                "images": [{
                    "format": "pdf",
                    "name": f"page_{page_number}"
                }]
            }

            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                page_pdf_bytes,
                filename=f'page_{page_number}.pdf',
                content_type='application/pdf'
            )
            form_data.add_field(
                'message',
                json.dumps(clova_message),
                content_type='application/json'
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=self.config.url,
                    data=form_data,
                    headers={'X-OCR-SECRET': self.config.secret_key}
                ) as response:
                    if response.status == 200:
                        clova_result = await response.json()
                        logger.info(f"✅ CLOVA OCR completed for page {page_number}")
                    else:
                        logger.error(f"❌ CLOVA OCR failed for page {page_number}: {response.status}")
                        return None

            # Parse CLOVA OCR response
            clova_text = ""
            for field in clova_result['images'][0]['fields']:
                clova_text += field['inferText'] + ('\n' if field['lineBreak'] else ' ')

            return clova_text.strip()

        except Exception as e:
            logger.error(f"CLOVA OCR extraction failed for page {page_number}: {e}")
            return None
