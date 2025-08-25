# 📄 Document Classification & Information Extraction System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Flask](https://img.shields.io/badge/Flask-API-brightgreen.svg) ![OCR](https://img.shields.io/badge/OCR-Tesseract-blueviolet.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This is an end-to-end Machine Learning project that automatically classifies documents from images and extracts key information. The system can identify document types like **Certificates, ID Cards, Invoices, and Resumes**, and then pull relevant text fields using OCR and Regex.

The entire workflow is served via a Flask API, and every prediction is logged into a SQLite database for tracking.

---

### ✨ Features

-   **Multi-class Document Classification**: A Deep Learning (CNN) model to classify images into 4 document categories.
-   **Optical Character Recognition (OCR)**: Uses Tesseract OCR to extract all text from the document image.
-   **Key Information Extraction**: Employs Regex to parse and extract specific fields like names, dates, amounts, and ID numbers from the OCR text.
-   **REST API**: A Flask-based API to serve the model and provide predictions on-the-fly.
-   **Prediction Logging**: Automatically logs every prediction request and result into a SQLite database.
-   **Structured MLOps Pipelines**: Separate, modular pipelines for model training and prediction.

---

### 📸 Demo / Screenshot

Here is a sample response from the prediction API when an ID card image is sent:

![Screenshot of the project's API JSON response]
*(Pro Tip: You can take a screenshot of Postman or a terminal showing the API's JSON output and add it here to make your repository look more impressive.)*

---

### 🛠️ Tech Stack

-   **Backend & API**: Python, Flask
-   **ML/DL Framework**: TensorFlow, Keras, Scikit-learn
-   **Data Processing**: Pandas, NumPy
-   **Image Processing**: OpenCV
-   **OCR Engine**: Pytesseract
-   **Database**: SQLite
-   **Packaging**: Setuptools

---

### ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the repository:**
    ```bash
    # Replace with your repository URL
    git clone [https://github.com/your-username/DocumentClassification_extraction.git](https://github.com/your-username/DocumentClassification_extraction.git)
    cd DocumentClassification_extraction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # It is recommended to use Python 3.10 or higher
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Tesseract OCR:**
    This is a crucial external dependency.
    -   Download and install it from the [official Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).
    -   After installation, you **must update the Tesseract path** in the `ocr/ocr_engine.py` file:
        ```python
        # ocr/ocr_engine.py
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <-- Update this path
        ```

4.  **Install the required Python packages:**
    The `setup.py` file is configured to install all dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

### 🚀 Usage

The project has three main functionalities: training the model, running local predictions, and serving the API.

1.  **Train the Model:**
    This script handles data loading, preprocessing, model training, and saves the final model artifacts (`classifier_model.h5` and `label_encoder.pkl`) into the `model/` directory.
    ```bash
    python src/DocumentClassification_extraction/pipelines/training_pipeline.py
    ```

2.  **Run a Prediction Locally:**
    This script uses the trained model to predict a single document's type and extracts its information, then logs the result to `logs.db`.
    ```bash
    python src/DocumentClassification_extraction/pipelines/prediction_pipeline.py
    ```

3.  **Run the Flask API:**
    This command starts a local server to handle prediction requests via HTTP.
    ```bash
    python api/main.py
    ```
    The API will be available at `http://127.0.0.1:5000`.
    -   **Health Check**: `GET /health`
    -   **Prediction**: `POST /predict` (sends an image file)

---

### 🌐 API Usage Example (cURL)

You can use a tool like Postman or `cURL` to send a `POST` request with an image to the `/predict` endpoint.

```bash
# Replace the file path with the actual path to your image
curl -X POST -F "file=@C:\path\to\your\document\image.jpg" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
```

**Sample JSON Response:**
```json
{
  "document_type": "ID_Card",
  "extracted_fields": {
    "id_number": "ABC12345",
    "name": "Suman Jaiswal"
  },
  "text": "Suman Jaiswal\nID: ABC12345\n..."
}
```

---

### 🗄️ Logging

Every prediction made through the `prediction_pipeline.py` is logged in the `logs.db` SQLite database.

To view the logs:
1.  Open a terminal in the project's root directory.
2.  Run the following commands:
    ```bash
    sqlite3 logs.db
    .tables
    SELECT * FROM predictions;
    .exit
    ```

---

### 📂 Project Structure

<details>
<summary>Click to view the folder structure</summary>

```
DOCUMENTCLASSIFICATION&EXTRACTION/
│
├── api/
│   └── main.py
├── data/
│   ├── certificates/
│   ├── id_cards/
│   ├── invoices/
│   └── resumes/
├── model/
│   ├── classifier_model.h5
│   ├── field_extraction.py
│   └── label_encoder.pkl
├── ocr/
│   └── ocr_engine.py
├── src/
│   └── DocumentClassification_extraction/
│       ├── components/
│       ├── pipelines/
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       ├── exception.py
│       └── logger.py
├── utils/
│   ├── preprocessing.py
│   └── logger.py
│
├── requirements.txt
├── setup.py
└── README.md
```
</details>

---

### 📄 License

This project is licensed under the MIT License.

---