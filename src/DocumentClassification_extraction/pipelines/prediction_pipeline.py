import os
import numpy as np
import tensorflow as tf
import joblib

from utils.preprocessing import preprocess_image
from ocr.ocr_engine import extract_text
from model.field_extraction import extract_fields

def predict_document(image_path):
    # Load the pre-trained model and label encoder
    model = tf.keras.models.load_model('model/classifier_model.h5')
    label_encoder = joblib.load('model/label_encoder.pkl')

    # Image preprocessing
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    # Prediction
    pred = model.predict(img)
    doc_type = label_encoder.inverse_transform([np.argmax(pred)])[0]

    # Extract text and fields from the image
    text = extract_text(image_path)
    fields = extract_fields(text)

    return {
        "document_type": doc_type,
        "extracted_fields": fields,
        "text": text
    }

if __name__ == "__main__":
    
    test_image = "data/id_cards/idcrd1 (1).jpg"
    result = predict_document(test_image)
    print("Prediction Result:")
    print(result)
