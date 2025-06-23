import tensorflow as tf
import logging
from src.DocumentClassification_extraction.exception import CustomException
import sys

class DataTransformationConfig:
    def __init__(self):
        self.img_size = (224, 224)
        self.augmentation_enabled = True

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_augmentation(self):
        """
        Creates data augmentation layers for training
        """
        try:
            logging.info("Creating data augmentation layers")
            
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomBrightness(0.1)
            ])
            
            logging.info("Data augmentation layers created successfully")
            return data_augmentation
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_preprocessing_layers(self):
        """
        Creates preprocessing layers (normalization)
        """
        try:
            logging.info("Creating preprocessing layers")
            
            # Rescaling layer to normalize pixel values to [0,1]
            normalization = tf.keras.layers.Rescaling(1./255)
            
            logging.info("Preprocessing layers created successfully")
            return normalization
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self):
        """
        Returns both augmentation and preprocessing layers
        """
        try:
            logging.info("Initiating data transformation")
            
            augmentation = self.get_data_augmentation()
            preprocessing = self.get_preprocessing_layers()
            
            logging.info("Data transformation completed successfully")
            return augmentation, preprocessing
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    aug, prep = obj.initiate_data_transformation()
