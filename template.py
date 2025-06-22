import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "DocumentClassification_extraction"

list_of_files = [
    # src structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    # DATA folders (with .gitkeep to ensure folder creation)
    "data/invoices/.gitkeep",
    "data/id_cards/.gitkeep",
    "data/certificates/.gitkeep",
    "data/resumes/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
        if os.path.getsize(filepath) == 0:
            logging.info(f"File {filename} is empty, skipping creation.")
        else:
            logging.info(f"File {filename} already exists and is not empty, skipping creation.")    