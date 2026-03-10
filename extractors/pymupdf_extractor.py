"""
PyMuPDF (fitz) based PDF hyperlink extractor
"""

import io
from typing import Any, Dict

import fitz  # PyMuPDF
from loguru import logger


class PyMuPDFExtractor:
    """Extract hyperlinks using PyMuPDF"""

    def __init__(self):
        self.name = "PyMuPDF"

    def extract_hyperlinks(self, pdf_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """
        Extract hyperlinks from a single PDF page

        Args:
            pdf_bytes: PDF bytes containing single page
            page_number: Page number for reference

        Returns:
            Dictionary containing list of hyperlinks with URLs and positions
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            if doc.page_count == 0:
                return {'hyperlinks': []}

            page = doc[0]
            links = page.get_links()

            hyperlinks = []
            for link in links:
                link_rect = link.get('from')
                link_text = ""
                link_type = "text"

                if link_rect:
                    # 1. 링크 영역 내 텍스트 검색
                    text_dict = page.get_text("dict")
                    text_found = False

                    for block in text_dict.get('blocks', []):
                        if block.get('type') == 0:
                            for line in block.get('lines', []):
                                for span in line.get('spans', []):
                                    span_rect = fitz.Rect(span.get('bbox', []))
                                    if span_rect.intersects(link_rect):
                                        link_text += span.get('text', '')
                                        text_found = True

                    if not text_found:
                        # 2. 이미지 확인
                        image_list = page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                                if img_rect and img_rect.intersects(link_rect):
                                    link_type = "image"
                                    link_text = f"[Image Link {img_index + 1}]"
                                    break
                            except Exception:
                                continue

                        # 3. 폼 필드 확인
                        if link_type == "text":
                            widgets = page.widgets()
                            for widget in widgets:
                                widget_rect = widget.rect
                                if widget_rect and widget_rect.intersects(link_rect):
                                    field_type = widget.field_type_string
                                    field_name = widget.field_name or f"Field_{len(widgets)}"
                                    link_type = "form_field"
                                    link_text = f"[{field_type} Field: {field_name}]"
                                    break

                        # 4. 주석 확인
                        if link_type == "text":
                            annotations = page.annots() or []
                            for annot in annotations:
                                annot_rect = annot.rect
                                if annot_rect and annot_rect.intersects(link_rect):
                                    annot_type = annot.type[1] if annot.type else "Unknown"
                                    link_type = "annotation"
                                    link_text = f"[{annot_type} Annotation]"
                                    break

                        # 5. 도형 확인
                        if link_type == "text":
                            drawings = page.get_drawings()
                            for i, drawing in enumerate(drawings):
                                if hasattr(drawing, 'rect') and drawing.rect and drawing.rect.intersects(link_rect):
                                    link_type = "drawing"
                                    link_text = f"[Drawing Element {i + 1}]"
                                    break

                        # 6. 클릭 가능 영역
                        if link_type == "text":
                            link_type = "area"
                            link_text = "[Clickable Area]"

                if link.get('uri'):  # External URL only
                    hyperlinks.append({
                        'url': link['uri'],
                        'text': link_text.strip(),
                        'link_type': link_type,
                        'rect': list(link.get('from', [])),
                        'page_number': page_number,
                        'type': 'external'
                    })

            doc.close()
            return {'hyperlinks': hyperlinks}

        except Exception as e:
            logger.error(f"PyMuPDF hyperlink extraction failed: {e}")
            return {'hyperlinks': []}
