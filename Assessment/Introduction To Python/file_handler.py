"""
File Handler Module
Handles all file operations including data persistence and transaction logging
"""
import json
import os
from datetime import datetime

class FileHandler:
    def __init__(self):
        self.data_file = "fruit_stock.json"
        self.log_file = "transaction_log.txt"
    
    def load_data(self):
        """Load fruit stock data from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as file:
                    data = json.load(file)
                    # Convert quantity to float (JSON stores as string for float keys)
                    converted_data = {}
                    for fruit, details in data.items():
                        converted_data[fruit] = {
                            'quantity': float(details['quantity']),
                            'price': float(details['price'])
                        }
                    return converted_data
            return {}
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}
    
    def save_data(self, fruit_stock):
        """Save fruit stock data to JSON file"""
        try:
            with open(self.data_file, 'w') as file:
                json.dump(fruit_stock, file, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def log_transaction(self, message):
        """Log transaction details to log file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            with open(self.log_file, 'a') as file:
                file.write(log_entry)
        except Exception as e:
            print(f"Error logging transaction: {e}")