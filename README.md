
# Document Classification & Information Extraction System

## Overview

This project is an end-to-end system for automatic **document type classification** (Certificate, ID Card, Invoice, Resume) and **information extraction** from scanned images or PDFs.  
It uses deep learning (CNN), OCR (Tesseract), regex-based field extraction, and a Flask API.  
Every prediction is logged in a SQLite database.



## Folder Structure


DOCUMENTCLASSIFICATION&EXTRACTION/
│
├── src/
│   └── DocumentClassification_extraction/
│       ├── components/
│       │   └── model_trainer.py
│       ├── pipelines/
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       └── ...
│
├── utils/
│   ├── preprocessing.py
│   ├── logger.py
│   └── ...
│
├── ocr/
│   └── ocr_engine.py
│
├── model/
│   ├── classifier_model.h5
│   ├── label_encoder.pkl
│   └── field_extraction.py
│
├── api/
│   └── main.py
│
├── data/
│   └── ... (images, labels.csv)
│
├── logs.db
├── requirements.txt
└── README.md


## Setup

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Includes: tensorflow, opencv-python, pytesseract, flask, joblib, sqlite3, etc.)*

3. **Install Tesseract OCR:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Set the path in `ocr/ocr_engine.py`:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```



## Usage

### 1. Model Training

```bash
python src/DocumentClassification_extraction/pipelines/training_pipeline.py
```
- Trains the model and saves weights and label encoder in the `model/` folder.

### 2. Prediction Pipeline

```bash
python src/DocumentClassification_extraction/pipelines/prediction_pipeline.py
```
- Predicts document type, extracts text & fields, and logs the result in `logs.db`.

### 3. Run Flask API

```bash
python api/main.py
```
- Health check: `GET /health`
- Prediction: `POST /predict` (form-data, key: `file`, value: image/pdf)

---

## Logging (SQLite)

- Every prediction is logged in `logs.db` (fields: filename, predicted_type, extracted_fields, timestamp)
- To view logs:
  ```bash
  sqlite3 logs.db
  .tables
  SELECT * FROM predictions;
  .exit
  ```



## cURL  (Image Prediction)

```bash
curl -X POST -F "file=@"C:\Users\singh\OneDrive\Documents\DocumentClassification&extraction\data\resumes\resume1 (10).jpg"" http://127.0.0.1:5000/predict
```

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- pytesseract
- Flask
- SQLite
- Regex
- easyocr
- scikit-learn

---

## Customization

- **Field Extraction:
  Edit regex rules in `model/field_extraction.py`.

- **Preprocessing:
  Change image size or normalization in `utils/preprocessing.py`.
  
- **API: 
  Customize endpoints in `api/main.py`.

name='DocumentClassification_extraction',
    version='0.0.1',
    author='suman jaiswal',
    author_email='jsonusuman351@gmail.com',
    