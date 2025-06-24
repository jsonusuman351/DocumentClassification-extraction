import os
import csv

folders = {
    "certificates": "Certificate",
    "id_cards": "ID_Card",
    "invoices": "Invoice",
    "resumes": "Resume"
}

data_dir = "data"
output_csv = os.path.join(data_dir, "labels.csv")
print("Output CSV Path:", output_csv)

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "document_type"])
    for folder, doc_type in folders.items():
        folder_path = os.path.join(data_dir, folder)
        print("Checking folder:", folder_path)
        if not os.path.isdir(folder_path):
            print("Folder missing:", folder_path)
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                print("Writing:", fname, doc_type)
                writer.writerow([fname, doc_type])

print(f"labels.csv created at {output_csv}")
