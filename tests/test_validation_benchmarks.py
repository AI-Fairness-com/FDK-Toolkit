# tests/test_validation_benchmarks.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Justice.fdk_justice_pipeline import JusticeFairnessPipeline

class TestBenchmarkValidation:
    """Validation tests against known fairness benchmarks"""
    
    def setup_method(self):
        self.pipeline = JusticeFairnessPipeline()
    
    def test_compas_benchmark_validation(self):
        """Test validation against COMPAS dataset patterns"""
        # Simulate COMPAS-like data structure
        compas_like_data = pd.DataFrame({
            'group': ['African-American'] * 300 + ['Caucasian'] * 300,
            'y_true': [1] * 150 + [0] * 150 + [1] * 100 + [0] * 200,  # Different base rates
            'y_pred': [1] * 180 + [0] * 120 + [1] * 80 + [0] * 220,   # Biased predictions
        })
        
        results = self.pipeline.run_pipeline(compas_like_data)
        
        # COMPAS-like data should show bias patterns
        metrics = results['fairness_metrics']
        
        # Check key fairness metrics
        assert 'statistical_parity_difference' in metrics
        assert 'fpr_difference' in metrics
        assert 'equal_opportunity_difference' in metrics
        
        # With the bias pattern, we expect some measurable bias
        composite_score = results['summary']['composite_bias_score']
        assert composite_score > 0.01, "COMPAS-like data should show some bias"
    
    def test_consistency_across_runs(self):
        """Test that pipeline produces consistent results"""
        test_data = pd.DataFrame({
            'group': ['A', 'B'] * 100,
            'y_true': np.random.randint(0, 2, 200),
            'y_pred': np.random.randint(0, 2, 200),
        })
        
        # Run pipeline multiple times
        results_1 = self.pipeline.run_pipeline(test_data)
        results_2 = self.pipeline.run_pipeline(test_data)
        
        # Composite scores should be very close (allowing for floating point differences)
        score_1 = results_1['summary']['composite_bias_score']
        score_2 = results_2['summary']['composite_bias_score']
        
        assert abs(score_1 - score_2) < 0.001, f"Scores not consistent: {score_1} vs {score_2}"
    
    def test_metric_boundaries(self):
        """Test that all metrics stay within reasonable boundaries"""
        test_data = pd.DataFrame({
            'group': ['X', 'Y', 'Z'] * 50,
            'y_true': np.random.randint(0, 2, 150),
            'y_pred': np.random.randint(0, 2, 150),
        })
        
        results = self.pipeline.run_pipeline(test_data)
        metrics = results['fairness_metrics']
        
        # Check boundary conditions for key metrics
        boundary_checks = {
            'statistical_parity_difference': (0, 1),
            'fpr_difference': (0, 1),
            'fnr_difference': (0, 1),
            'composite_bias_score': (0, 1)
        }
        
        for metric, (min_val, max_val) in boundary_checks.items():
            if metric in metrics:
                value = metrics[metric]
                assert min_val <= value <= max_val, f"Metric {metric} out of bounds: {value}"
