
print("Script starting...")
try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"TensorFlow import error: {e}")




import tensorflow as tf
import os
import logging
from pathlib import Path
from src.DocumentClassification_extraction.exception import CustomException
from src.DocumentClassification_extraction.logger import logging
import sys

class DataIngestionConfig:
    def __init__(self):
        self.data_dir = "data"
        self.img_size = (224, 224)
        self.batch_size = 32
        self.val_split = 0.2
        self.seed = 42

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        This method loads images from folder structure and creates train/validation datasets
        """
        try:
            logging.info("Entered the data ingestion method")
            
            # Check if data directory exists
            if not os.path.exists(self.ingestion_config.data_dir):
                raise CustomException(f"Data directory {self.ingestion_config.data_dir} does not exist", sys)
            
            # Check if all class folders exist
            class_folders = ["invoices", "id_cards", "certificates", "resumes"]
            for folder in class_folders:
                folder_path = os.path.join(self.ingestion_config.data_dir, folder)
                if not os.path.exists(folder_path):
                    raise CustomException(f"Class folder {folder_path} does not exist", sys)
                    
                # Check if folder has images
                image_count = len([f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                logging.info(f"Found {image_count} images in {folder}")
                
                if image_count == 0:
                    logging.warning(f"No images found in {folder} folder")
            
            # Create train dataset
            logging.info("Creating training dataset")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                self.ingestion_config.data_dir,
                validation_split=self.ingestion_config.val_split,
                subset="training",
                seed=self.ingestion_config.seed,
                image_size=self.ingestion_config.img_size,
                batch_size=self.ingestion_config.batch_size,
                label_mode="categorical"
            )
            
            # Create validation dataset
            logging.info("Creating validation dataset")
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.ingestion_config.data_dir,
                validation_split=self.ingestion_config.val_split,
                subset="validation",
                seed=self.ingestion_config.seed,
                image_size=self.ingestion_config.img_size,
                batch_size=self.ingestion_config.batch_size,
                label_mode="categorical"
            )
            
            # Get class names
            class_names = train_ds.class_names
            logging.info(f"Found classes: {class_names}")
            
            # Optimize datasets for performance
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            
            logging.info("Data ingestion completed successfully")
            logging.info(f"Training batches: {len(train_ds)}")
            logging.info(f"Validation batches: {len(val_ds)}")
            
            return train_ds, val_ds, class_names
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_dataset, val_dataset, classes = obj.initiate_data_ingestion()
