import pandas as pd
import os

def load_compas_data():
    """Load COMPAS dataset for justice fairness testing"""
    file_path = os.path.join(os.path.dirname(__file__), '../data/real_datasets/compas_processed.csv')
    return pd.read_csv(file_path)

def get_compas_info():
    """Get COMPAS dataset information"""
    info_path = os.path.join(os.path.dirname(__file__), '../data/real_datasets/dataset_info.json')
    with open(info_path, 'r') as f:
        import json
        return json.load(f)
