import easyocr
import pytesseract
from PIL import Image
import io

class OCREngine:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'])
    
    def extract_text_easyocr(self, image_path):
        result = self.easyocr_reader.readtext(image_path, detail=0)
        return " ".join(result)
    
    def extract_text_tesseract(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
