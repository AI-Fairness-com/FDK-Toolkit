# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

@pytest.fixture
def sample_justice_data():
    """Fixture providing sample justice data for tests"""
    return pd.DataFrame({
        'group': ['Urban', 'Rural', 'Suburban'] * 100,
        'y_true': np.random.randint(0, 2, 300),
        'y_pred': np.random.randint(0, 2, 300),
        'y_prob': np.random.random(300)
    })

@pytest.fixture
def biased_justice_data():
    """Fixture providing intentionally biased justice data"""
    return pd.DataFrame({
        'group': ['Majority'] * 200 + ['Minority'] * 100,
        'y_true': [1] * 100 + [0] * 100 + [1] * 25 + [0] * 75,
        'y_pred': [1] * 120 + [0] * 80 + [1] * 15 + [0] * 85,
    })

@pytest.fixture
def temp_json_file():
    """Fixture for temporary JSON file creation"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'test': 'data'}, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)
