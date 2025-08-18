"""
Tesseract OCR Extractor - 경량 OCR 엔진
PaddleOCR를 대체하는 Tesseract 기반 OCR 추출기
"""

from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

# Tesseract OCR
import pytesseract

from utils.logger import logger, log_extraction_result


class TesseractExtractor:
    """Tesseract 기반 OCR 텍스트 추출기"""
    
    def __init__(self):
        """초기화"""
        self.name = "Tesseract-OCR"
        
        
        # Tesseract 설정
        self.languages = 'kor+eng'  # 한국어 + 영어
        # PSM 6: 균일한 텍스트 블록 (문서에 적합)
        # OEM 3: LSTM + Legacy 엔진 조합
        self.config = '--oem 3 --psm 6'
        self.confidence_threshold = 30  # 신뢰도 임계값
        
        # Tesseract 설치 확인
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract OCR initialized (version: {version})")
        except Exception as e:
            logger.error(f"❌ Tesseract not found. Install with: brew install tesseract tesseract-lang")
            logger.error(f"Error: {e}")
            self.languages = 'eng'  # 영어로 폴백
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """이미지 전처리로 OCR 정확도 향상"""
        try:
            # 그레이스케일 변환
            if img.mode != 'L':
                img = img.convert('L')
            
            # 노이즈 제거 (문서 스캔에 효과적)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            # 대비 향상 (적당한 수준으로 조정)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            # 선명도 향상
            img = img.filter(ImageFilter.SHARPEN)
            
            # 크기 조정 (OCR 최적 크기로 조정)
            width, height = img.size
            if width < 1500:  # 임계값 증가
                scale = 1500 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 이진화 처리 (문서에 효과적)
            # Otsu's 방법을 사용한 적응형 이진화
            import numpy as np
            img_array = np.array(img)
            threshold = np.mean(img_array)
            img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            return img
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return img
    
    def extract_from_images(self, images: List[bytes]) -> Dict[str, Any]:
        """이미지에서 텍스트 추출 (부분 결과 반환 지원)"""
        
        try:
            result = {
                'text': '',
                'pages': [],
                'structure': {'blocks': [], 'lines': []},
                'metadata': {
                    'total_texts': 0,
                    'avg_confidence': 0.0,
                    'language': self.languages,
                    'processed_pages': 0,
                    'total_pages': len(images)
                }
            }
            
            all_texts = []
            total_confidence = 0.0
            total_count = 0
            page_timeout = 300.0  # 페이지당 최대 5분 (300초)
            
            for page_num, img_bytes in enumerate(images):
                
                try:
                    # 이미지 로드
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # 전처리
                    img = self._preprocess_image(img)
                    
                    # OCR 실행
                    try:
                        # 텍스트와 신뢰도 함께 추출
                        data = pytesseract.image_to_data(
                            img, 
                            lang=self.languages,
                            config=self.config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # 텍스트 추출 및 신뢰도 계산
                        page_texts = []
                        page_confidence = []
                        
                        n_boxes = len(data['text'])
                        for i in range(n_boxes):
                            text = data['text'][i].strip()
                            conf = float(data['conf'][i])
                            
                            # 신뢰도 임계값 이상인 텍스트만 추출
                            if text and conf > self.confidence_threshold:
                                page_texts.append(text)
                                page_confidence.append(conf)
                        
                        if page_texts:
                            page_text = ' '.join(page_texts)
                            all_texts.append(page_text)
                            
                            avg_conf = sum(page_confidence) / len(page_confidence) if page_confidence else 0.0
                            total_confidence += sum(page_confidence)
                            total_count += len(page_texts)
                            
                            result['pages'].append({
                                'page_num': page_num + 1,
                                'text': page_text,
                                'text_count': len(page_texts),
                                'avg_confidence': avg_conf / 100.0  # Tesseract는 0-100 범위
                            })
                            
                    
                    except Exception as ocr_e:
                        # 폴백: 단순 텍스트 추출
                        logger.warning(f"Detailed OCR failed, using simple extraction: {ocr_e}")
                        text = pytesseract.image_to_string(
                            img,
                            lang=self.languages,
                            config=self.config
                        )
                        if text.strip():
                            all_texts.append(text.strip())
                            result['pages'].append({
                                'page_num': page_num + 1,
                                'text': text.strip(),
                                'text_count': 1,
                                'avg_confidence': 0.5  # 기본 신뢰도
                            })
                            total_count += 1
                            total_confidence += 50  # 기본 신뢰도
                        
                except Exception as page_e:
                    logger.error(f"Page {page_num + 1} processing failed: {page_e}")
                    # 페이지 처리 실패 시에도 계속 진행
                    result['metadata']['processed_pages'] = page_num
                    continue
            
            # 최종 결과
            result['text'] = '\n'.join(all_texts)
            result['metadata']['total_texts'] = total_count
            result['metadata']['avg_confidence'] = (total_confidence / total_count / 100.0) if total_count > 0 else 0.0
            result['metadata']['processed_pages'] = len(images)
            
            # 부분 처리 성공 여부 확인
            if result['text'].strip():
                pages_processed = len(result['pages'])
                success_rate = pages_processed / len(images) * 100
                log_extraction_result(
                    self.name, 
                    True, 
                    f"{total_count} text blocks from {pages_processed}/{len(images)} pages ({success_rate:.0f}%)"
                )
                logger.info(f"✅ Tesseract: {len(result['text'])} chars extracted from {pages_processed} pages")
            else:
                log_extraction_result(self.name, False, "No text extracted")
                logger.warning("⚠️ Tesseract: No text extracted")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tesseract processing failed: {e}")
            log_extraction_result(self.name, False, str(e))
            
            # 부분 결과가 있으면 반환
            if 'result' in locals() and result.get('text'):
                logger.info(f"📝 Returning partial results: {len(result['text'])} chars")
                result['error'] = f'Partial extraction: {str(e)}'
                return result
            
            return {'text': '', 'error': str(e)}
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        return False