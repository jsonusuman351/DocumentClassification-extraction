import sqlite3
import json
from datetime import datetime

DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_type TEXT,
            extracted_fields TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(filename, predicted_type, extracted_fields):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (filename, predicted_type, extracted_fields, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (
        filename,
        predicted_type,
        json.dumps(extracted_fields, ensure_ascii=False),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()
