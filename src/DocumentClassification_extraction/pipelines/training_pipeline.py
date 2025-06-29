import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from src.DocumentClassification_extraction.components.model_trainer import build_model
from utils.preprocessing import preprocess_image   

# Step 1: Data Load
labels_file = os.path.join("data", "labels.csv")
df = pd.read_csv(labels_file)

# Step 2: Label Encode
le = LabelEncoder()
df['label'] = le.fit_transform(df['document_type'])

# Step 3: Image Preprocessing
X = []
for idx, row in df.iterrows():
    doc_type = row['document_type']
    if doc_type == "Certificate":
        subfolder = "certificates"
    elif doc_type == "ID_Card":
        subfolder = "id_cards"
    elif doc_type == "Invoice":
        subfolder = "invoices"
    elif doc_type == "Resume":
        subfolder = "resumes"
    else:
        raise ValueError(f"Unknown document type: {doc_type}")
    img_path = os.path.join("data", subfolder, row['filename'])
    img = preprocess_image(img_path)
    X.append(img)
X = np.array(X)
y = df['label'].values

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Model Build
model = build_model(num_classes=len(le.classes_))

# Step 6: Model Training
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Step 7: Model Save
os.makedirs("model", exist_ok=True)
model.save("model/classifier_model.h5")
joblib.dump(le, "model/label_encoder.pkl")

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nModel Evaluation on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("Classification Report:\n", classification_report(y_test, y_pred_labels, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))

print("\nModel training complete. Model and label encoder saved in 'model' folder.")
