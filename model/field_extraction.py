import re

def extract_fields(text):
    fields = {}
    name_match = re.search(r'Name[:\s]+([A-Za-z ]+)', text)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    amount_match = re.search(r'Amount[:\s]+([₹$]?\d+[,.\d]*)', text)
    id_match = re.search(r'ID[:\s]+(\w+)', text)
    if name_match:
        fields['name'] = name_match.group(1)
    if date_match:
        fields['date'] = date_match.group(1)
    if amount_match:
        fields['amount'] = amount_match.group(1)
    if id_match:
        fields['id_number'] = id_match.group(1)
    return fields