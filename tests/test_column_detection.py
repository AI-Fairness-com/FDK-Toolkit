# tests/test_column_detection.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Justice.fdk_justice import detect_justice_column_mappings

class TestColumnDetection:
    """Tests for automatic column detection in justice domain"""
    
    def test_justice_specific_detection(self):
        """Test detection of justice-specific column names"""
        df = pd.DataFrame({
            'defendant_race': ['Black', 'White', 'Hispanic'] * 10,
            'recidivism_outcome': [1, 0, 1] * 10,
            'risk_prediction': [1, 0, 0] * 10,
            'probability_score': np.random.random(30)
        })
        
        suggestions, reasoning = detect_justice_column_mappings(df, df.columns.tolist())
        
        # Should detect justice-specific columns
        assert suggestions['group'] == 'defendant_race'
        assert suggestions['y_true'] == 'recidivism_outcome'
        assert suggestions['y_pred'] == 'risk_prediction'
        assert suggestions['y_prob'] == 'probability_score'
    
    def test_fallback_detection(self):
        """Test fallback detection for generic column names"""
        df = pd.DataFrame({
            'category': ['A', 'B'] * 15,
            'outcome': [1, 0] * 15,
            'prediction': [1, 1, 0, 0] * 7 + [1, 1],  # Ensure even length
            'other_column': range(30)
        })
        
        suggestions, reasoning = detect_justice_column_mappings(df, df.columns.tolist())
        
        # Should use fallback detection
        assert suggestions['group'] == 'category'
        assert suggestions['y_true'] == 'outcome'
        assert suggestions['y_pred'] == 'prediction'
    
    def test_insufficient_columns(self):
        """Test handling of datasets with insufficient columns"""
        df = pd.DataFrame({
            'group': ['A', 'B'],
            'outcome': [1, 0]
        })
        
        suggestions, reasoning = detect_justice_column_mappings(df, df.columns.tolist())
        
        # Should detect what it can
        assert suggestions['group'] == 'group'
        assert suggestions['y_true'] == 'outcome'
        assert suggestions['y_pred'] is None  # Can't detect prediction column
