import os
import pandas as pd

class DataIngestion:
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        self.labels_file = labels_file

    def load_data(self):
        
        df = pd.read_csv(self.labels_file)
       
        def get_full_path(row):
            
            subfolder = None
            if row['document_type'] == "Certificate":
                subfolder = "certificates"
            elif row['document_type'] == "ID_Card":
                subfolder = "id_cards"
            elif row['document_type'] == "Invoice":
                subfolder = "invoices"
            elif row['document_type'] == "Resume":
                subfolder = "resumes"
            return os.path.join(self.data_dir, subfolder, row['filename'])
        df['filepath'] = df.apply(get_full_path, axis=1)
        return df

if __name__ == "__main__":
    
    data_dir = "data"
    labels_file = os.path.join(data_dir, "labels.csv")
    ingestion = DataIngestion(data_dir, labels_file)
    df = ingestion.load_data()
    print(df.head())  
    print("Total records:", len(df))
