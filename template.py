import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "DocumentClassification_extraction"

list_of_files = [
    # Directory structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    
    # Model files
    "model/__init__.py",
    "model/classifier_model.h5",
    "model/field_extraction.py",
    
    "ocr/__init__.py",
    "ocr/ocr_engine.py",
    
    "api/__init__.py",
    "api/main.py",
    "api/utils.py",
    
    "utils/__init__.py",
    "utils/preprocessing.py",
    "utils/image_utils.py",
    "create_labels_csv.py",
    
    # Data directories
    "data/invoices/.gitkeep",
    "data/id_cards/.gitkeep",
    "data/certificates/.gitkeep",
    "data/resumes/.gitkeep",
    
    # Logs directory
    "logs/__init__.py",
    "logs/predictions.db",
    
    # Configuration files
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            # Add basic starter code for key files
            if filepath.name == "ocr_engine.py":
                f.write("""import easyocr
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
""")
            elif filepath.name == "field_extraction.py":
                f.write("""import re

class FieldExtractor:
    def extract_invoice_fields(self, text):
        name = re.search(r"Name[:\\-]?\\s*([A-Za-z ]+)", text)
        date = re.search(r"Date[:\\-]?\\s*([\\d\\-\\/]+)", text)
        amount = re.search(r"Amount[:\\-]?\\s*([\\d,\\.]+)", text)
        return {
            "name": name.group(1) if name else "",
            "date": date.group(1) if date else "",
            "amount": amount.group(1) if amount else ""
        }
    
    def extract_id_fields(self, text):
        name = re.search(r"Name[:\\-]?\\s*([A-Za-z ]+)", text)
        dob = re.search(r"DOB[:\\-]?\\s*([\\d\\-\\/]+)", text)
        id_number = re.search(r"ID[:\\-]?\\s*([A-Z0-9]+)", text)
        return {
            "name": name.group(1) if name else "",
            "dob": dob.group(1) if dob else "",
            "id_number": id_number.group(1) if id_number else ""
        }
""")
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("Complete project structure created successfully!")
