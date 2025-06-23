import sys
import logging
from src.DocumentClassification_extraction.exception import CustomException
from src.DocumentClassification_extraction.components.data_ingestion import DataIngestion
from src.DocumentClassification_extraction.components.data_transformation import DataTransformation
from src.DocumentClassification_extraction.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass
    
    def start_training(self):
        """
        Complete training pipeline from data ingestion to model training
        """
        try:
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_ds, val_ds, class_names = data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            augmentation, preprocessing = data_transformation.initiate_data_transformation()
            
            # Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            trained_model, history = model_trainer.initiate_model_trainer(
                train_ds, val_ds, augmentation, preprocessing, len(class_names)
            )
            
            logging.info("Training pipeline completed successfully")
            logging.info(f"Trained model saved")
            logging.info(f"Classes: {class_names}")
            
            return trained_model, class_names
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    model, classes = training_pipeline.start_training()
