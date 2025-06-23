import tensorflow as tf
import os
import logging
from src.DocumentClassification_extraction.exception import CustomException
import sys

class ModelTrainerConfig:
    def __init__(self):
        self.model_save_path = "model/classifier_model.h5"
        self.base_model_name = "MobileNetV2"
        self.num_classes = 4
        self.img_size = (224, 224)
        self.epochs = 10
        self.fine_tune_epochs = 5
        self.learning_rate = 0.0001
        self.fine_tune_at = 100

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def build_model(self, num_classes):
        """
        Builds and compiles the CNN model using transfer learning
        """
        try:
            logging.info(f"Building model with {num_classes} classes")
            
            # Load pre-trained MobileNetV2
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.model_trainer_config.img_size + (3,),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Add custom classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_trainer_config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logging.info("Model built and compiled successfully")
            logging.info(f"Model summary: {model.summary()}")
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_model(self, model, train_ds, val_ds, augmentation, preprocessing):
        """
        Trains the model with the given datasets
        """
        try:
            logging.info("Starting model training")
            
            # Add preprocessing and augmentation to model
            model_with_preprocessing = tf.keras.Sequential([
                preprocessing,
                augmentation,
                model
            ])
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=2,
                    min_lr=0.0001
                )
            ]
            
            # Train the model
            history = model_with_preprocessing.fit(
                train_ds,
                epochs=self.model_trainer_config.epochs,
                validation_data=val_ds,
                callbacks=callbacks
            )
            
            logging.info("Model training completed successfully")
            return model_with_preprocessing, history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_model(self, model):
        """
        Saves the trained model
        """
        try:
            logging.info(f"Saving model to {self.model_trainer_config.model_save_path}")
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_trainer_config.model_save_path), exist_ok=True)
            
            # Save the model
            model.save(self.model_trainer_config.model_save_path)
            
            logging.info("Model saved successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self, train_ds, val_ds, augmentation, preprocessing, num_classes):
        """
        Complete model training pipeline
        """
        try:
            logging.info("Initiating model training pipeline")
            
            # Build model
            model = self.build_model(num_classes)
            
            # Train model
            trained_model, history = self.train_model(model, train_ds, val_ds, augmentation, preprocessing)
            
            # Save model
            self.save_model(trained_model)
            
            logging.info("Model training pipeline completed successfully")
            return trained_model, history
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This will be called from training pipeline
    pass
