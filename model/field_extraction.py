import re

class FieldExtractor:
    def extract_invoice_fields(self, text):
        name = re.search(r"Name[:\-]?\s*([A-Za-z ]+)", text)
        date = re.search(r"Date[:\-]?\s*([\d\-\/]+)", text)
        amount = re.search(r"Amount[:\-]?\s*([\d,\.]+)", text)
        return {
            "name": name.group(1) if name else "",
            "date": date.group(1) if date else "",
            "amount": amount.group(1) if amount else ""
        }
    
    def extract_id_fields(self, text):
        name = re.search(r"Name[:\-]?\s*([A-Za-z ]+)", text)
        dob = re.search(r"DOB[:\-]?\s*([\d\-\/]+)", text)
        id_number = re.search(r"ID[:\-]?\s*([A-Z0-9]+)", text)
        return {
            "name": name.group(1) if name else "",
            "dob": dob.group(1) if dob else "",
            "id_number": id_number.group(1) if id_number else ""
        }
