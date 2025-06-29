from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import os

from utils.preprocessing import preprocess_image
from ocr.ocr_engine import extract_text
from model.field_extraction import extract_fields
from utils.logger import init_db, log_prediction

app = Flask(__name__)

# Load the pre-trained model and label encoder
model = tf.keras.models.load_model('model/classifier_model.h5')
label_encoder = joblib.load('model/label_encoder.pkl')

# Initialize database for logging
init_db()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Document Classification API is running!',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST with file)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is actually selected
        if not file or file.filename == '':
            return jsonify({'error': 'No file provided'}), 400
        
        # Save uploaded file temporarily
        file_path = 'temp_upload.png'
        file.save(file_path)
        
        # Preprocess and predict
        img = preprocess_image(file_path)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        doc_type = label_encoder.inverse_transform([np.argmax(pred)])[0]
        
        # Extract text and fields
        text = extract_text(file_path)
        fields = extract_fields(text)
        
        # Log the prediction
        log_prediction(
            filename=file.filename,
            predicted_type=doc_type,
            extracted_fields=fields
        )
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            "document_type": doc_type,
            "extracted_fields": fields,
            "text": text
        })
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists('temp_upload.png'):
            os.remove('temp_upload.png')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
