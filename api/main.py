from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

from utils.preprocessing import preprocess_image
from ocr.ocr_engine import extract_text
from model.field_extraction import extract_fields

app = Flask(__name__)

# Model और label encoder load करें (startup पर)
model = tf.keras.models.load_model('model/classifier_model.h5')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    file_path = 'temp_upload.png'
    file.save(file_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Document Classification API is running!',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST with file)'
        }
    })
    if not file:
        return jsonify({'error': 'No file provided'}), 400  
    # Preprocess and predict
    img = preprocess_image(file_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    doc_type = label_encoder.inverse_transform([np.argmax(pred)])[0]

    text = extract_text(file_path)
    fields = extract_fields(text)

    return jsonify({
        "document_type": doc_type,
        "extracted_fields": fields,
        "text": text
    })

if __name__ == '__main__':
    app.run(debug=True)
