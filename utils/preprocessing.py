import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img

if __name__ == "__main__":
    
    test_path = os.path.join("data","certificates","certificate1 (1).jpg")  
    print("Testing path:", test_path)
    print("File exists?", os.path.exists(test_path))
    img_arr = preprocess_image(test_path)
    print("Shape:", img_arr.shape)
    print("Min value:", img_arr.min(), "Max value:", img_arr.max())
