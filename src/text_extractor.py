"""
Text Extractor Module
Handles OCR text extraction from images using Tesseract
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from images using OCR"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize TextExtractor
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
    def preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Apply different preprocessing techniques to improve OCR accuracy
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of preprocessing method names and processed images
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            processed_images = {}
            
            # Method 1: Simple threshold
            _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            processed_images["simple_threshold"] = thresh1
            
            # Method 2: Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images["adaptive_threshold"] = thresh2
            
            # Method 3: Otsu's thresholding
            _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images["otsu_threshold"] = thresh3
            
            # Method 4: Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images["gaussian_otsu"] = thresh4
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> Tuple[str, Dict]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply preprocessing techniques
            
        Returns:
            Tuple of (extracted_text, extraction_details)
        """
        try:
            if preprocess:
                # Try multiple preprocessing methods
                processed_images = self.preprocess_image(image_path)
                extraction_results = {}
                
                for method_name, processed_img in processed_images.items():
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(pil_img)
                    extraction_results[method_name] = {
                        "text": text,
                        "length": len(text.strip())
                    }
                
                # Find the best method (most text extracted)
                best_method = max(
                    extraction_results.keys(),
                    key=lambda x: extraction_results[x]["length"]
                )
                best_text = extraction_results[best_method]["text"]
                
                return best_text, {
                    "best_method": best_method,
                    "all_methods": extraction_results,
                    "preprocessed": True
                }
            else:
                # Simple extraction without preprocessing
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)
                
                return text, {
                    "best_method": "simple",
                    "all_methods": {"simple": {"text": text, "length": len(text.strip())}},
                    "preprocessed": False
                }
                
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\-:;\'"()]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


if __name__ == "__main__":
    # Example usage
    extractor = TextExtractor()
    text, details = extractor.extract_text("sample_image.jpg")
    print(f"Extracted {len(text)} characters")
    print(f"Best method: {details['best_method']}")
