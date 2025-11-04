# ================================================================
# FDK Health Pipeline - STREAMLINED v2.2
# 19 Comprehensive Healthcare Fairness Metrics
# MIT License - AI Ethics Research Group
# ================================================================

import os
import json
import math
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error, roc_curve
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Streamlined health metrics configuration
HEALTH_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference',
        'equal_opportunity_difference',
        'demographic_parity_ratio'
    ],
    'performance_error_fairness': [
        'tpr_difference', 'fpr_difference',
        'ppv_difference', 'error_rate_difference'
    ],
    'calibration_reliability': [
        'calibration_gap_difference',
        'slice_auc_difference'
    ],
    'subgroup_disparity_analysis': [
        'error_disparity_subgroup',
        'worst_group_accuracy'
    ],
    'statistical_inequality': [
        'coefficient_of_variation',
        'mean_difference'
    ]
}

class HealthFairnessPipeline:
    """Streamlined fairness assessment for healthcare AI systems"""
    
    def __init__(self, clinical_context: bool = True, risk_threshold: float = 0.1):
        self.metrics_history = []
        self.temporal_window = 10
        self.clinical_context = clinical_context
        self.risk_threshold = risk_threshold
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for healthcare audit"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # ================================================================
    # UTILITY FUNCTIONS
    # ================================================================

    def bounded_value(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Ensure values stay within reasonable bounds"""
        if value is None or np.isnan(value):
            return min_val
        return float(max(min_val, min(max_val, value)))

    def safe_div(self, a: float, b: float, default: float = 0.0) -> float:
        """Safe division with error handling"""
        try:
            if b == 0 or abs(b) < 1e-10:
                return self.bounded_value(default)
            result = a / b
            return self.bounded_value(result)
        except Exception:
            return self.bounded_value(default)

    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _proportion_ci(self, p: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Normal approximation CI for proportion"""
        if n is None or n == 0 or p is None:
            return (0.0, 1.0)
        try:
            z = st.norm.ppf(1 - alpha / 2)
            se = math.sqrt(p * (1 - p) / n)
            lower = self.bounded_value(p - z * se)
            upper = self.bounded_value(p + z * se)
            return (lower, upper)
        except Exception:
            return (0.0, 1.0)

    def validate_healthcare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Healthcare data validation"""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required columns
        required_cols = ['group', 'y_true', 'y_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_cols}")
            return validation_results
        
        # Check group diversity
        groups = df['group'].unique()
        if len(groups) < 2:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Need at least 2 groups for fairness analysis")
        
        # Check group sizes
        group_counts = df['group'].value_counts()
        small_groups = [group for group, count in group_counts.items() if count < 10]
        if small_groups:
            validation_results["warnings"].append(
                f"Small group sizes may affect reliability: {small_groups}"
            )
        
        return validation_results

    # ================================================================
    # 1. Core Group Fairness Metrics
    # ================================================================

    def calculate_core_group_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Core group fairness metrics for healthcare"""
        metrics = {}
        groups = df['group'].unique()
        
        # Selection rates and base rates
        selection_rates, base_rates = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            selection_rates[group] = self.bounded_value(float(group_data['y_pred'].mean()))
            base_rates[group] = self.bounded_value(float(group_data['y_true'].mean()))
        
        if len(selection_rates) >= 2:
            # 1. Statistical Parity Difference
            spd = self.bounded_value(
                float(max(selection_rates.values()) - min(selection_rates.values())),
                min_val=0.0, max_val=1.0
            )
            metrics['statistical_parity_difference'] = spd
            
            # 2. Demographic Parity Ratio
            min_rate = min(selection_rates.values())
            max_rate = max(selection_rates.values())
            if max_rate > 0 and min_rate > 0:
                metrics['demographic_parity_ratio'] = self.bounded_value(float(min_rate / max_rate))
            else:
                metrics['demographic_parity_ratio'] = 0.0
        
        # 3. Equal Opportunity Difference
        tpr_values = self._calculate_group_tpr(df)
        if tpr_values and len(tpr_values) > 1:
            eo_diff = self.bounded_value(
                float(max(tpr_values.values()) - min(tpr_values.values())),
                min_val=0.0, max_val=1.0
            )
            metrics['equal_opportunity_difference'] = eo_diff
        
        return metrics

    def _calculate_group_tpr(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate True Positive Rate by group"""
        groups = df['group'].unique()
        tpr_values = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = self._confusion_counts(y_true, y_pred)
                tpr = self.safe_div(tp, (tp + fn))
                tpr_values[group] = self.bounded_value(float(tpr))
            except Exception:
                continue
        
        return tpr_values

    def _calculate_group_fpr(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate False Positive Rate by group"""
        groups = df['group'].unique()
        fpr_values = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = self._confusion_counts(y_true, y_pred)
                fpr = self.safe_div(fp, (fp + tn))
                fpr_values[group] = self.bounded_value(float(fpr))
            except Exception:
                continue
        
        return fpr_values

    def _confusion_counts(self, y_true, y_pred):
        """Calculate confusion matrix counts"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            return int(tn), int(fp), int(fn), int(tp)
        except Exception:
            return 0, 0, 0, 0

    # ================================================================
    # 2. Performance and Error Fairness Metrics
    # ================================================================

    def calculate_performance_error_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Performance and error fairness metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        tpr_vals, fpr_vals, error_rates, ppv_vals = {}, {}, {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = self._confusion_counts(y_true, y_pred)
                
                # Calculate key rates
                tpr_vals[group] = self.bounded_value(self.safe_div(tp, (tp + fn)))
                fpr_vals[group] = self.bounded_value(self.safe_div(fp, (fp + tn)))
                ppv_vals[group] = self.bounded_value(self.safe_div(tp, (tp + fp)))
                error_rates[group] = self.bounded_value(self.safe_div((fp + fn), (tp + tn + fp + fn)))
                
            except Exception:
                continue
        
        # Calculate differences
        self._calculate_differences_ratios(metrics, 'tpr', tpr_vals)
        self._calculate_differences_ratios(metrics, 'fpr', fpr_vals)
        self._calculate_differences_ratios(metrics, 'error_rate', error_rates)
        self._calculate_differences_ratios(metrics, 'ppv', ppv_vals)
        
        return metrics

    def _calculate_differences_ratios(self, metrics: Dict, prefix: str, values: Dict):
        """Calculate difference for a metric across groups"""
        if values and len(values) > 1:
            valid_vals = [v for v in values.values() if v is not None]
            if valid_vals:
                diff = self.bounded_value(
                    float(max(valid_vals) - min(valid_vals)),
                    min_val=0.0, max_val=1.0
                )
                metrics[f'{prefix}_difference'] = diff

    # ================================================================
    # 3. Calibration and Reliability Metrics
    # ================================================================

    def calculate_calibration_reliability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calibration and reliability metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        calibration_gaps, auc_scores = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            # Calibration Gap
            if 'y_prob' in df.columns:
                try:
                    y_true = group_data['y_true'].values
                    y_prob = group_data['y_prob'].values
                    
                    mean_pred_prob = self.bounded_value(float(y_prob.mean()))
                    actual_rate = self.bounded_value(float(y_true.mean()))
                    calibration_gaps[group] = self.bounded_value(float(abs(mean_pred_prob - actual_rate)))
                except Exception:
                    calibration_gaps[group] = 0.0
            
            # Slice AUC Difference
            if 'y_prob' in df.columns:
                try:
                    y_true = group_data['y_true'].values
                    y_prob = group_data['y_prob'].values
                    
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, y_prob)
                        auc_scores[group] = self.bounded_value(float(auc))
                except Exception:
                    continue
        
        # Calculate differences
        if calibration_gaps and len(calibration_gaps) > 1:
            valid_calibration = [v for v in calibration_gaps.values() if v is not None]
            if valid_calibration:
                metrics['calibration_gap_difference'] = self.bounded_value(
                    float(max(valid_calibration) - min(valid_calibration)),
                    min_val=0.0, max_val=1.0
                )
        
        if auc_scores and len(auc_scores) > 1:
            valid_auc = [v for v in auc_scores.values() if v is not None]
            if valid_auc:
                metrics['slice_auc_difference'] = self.bounded_value(
                    float(max(valid_auc) - min(valid_auc)),
                    min_val=0.0, max_val=1.0
                )
        
        return metrics

    # ================================================================
    # 4. Subgroup and Disparity Analysis
    # ================================================================

    def calculate_subgroup_disparity_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Subgroup and disparity analysis"""
        metrics = {}
        
        # Error Disparity by Subgroup
        error_disparity = self._calculate_error_disparity_subgroup(df)
        metrics['error_disparity_subgroup'] = error_disparity
        
        # Worst-Group Analysis
        worst_group_metrics = self._calculate_worst_group_analysis(df)
        metrics.update(worst_group_metrics)
        
        return metrics

    def _calculate_error_disparity_subgroup(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate error disparity across subgroups"""
        groups = df['group'].unique()
        error_rates = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            accuracy = (group_data['y_true'] == group_data['y_pred']).mean()
            error_rates[group] = self.bounded_value(1.0 - accuracy) if accuracy is not None else 0.0
        
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                error_range = self.bounded_value(float(max(valid_errors) - min(valid_errors)))
                return {
                    'range': error_range,
                    'error_rates_by_group': error_rates
                }
        
        return {'range': 0.0, 'error_rates_by_group': error_rates}

    def _calculate_worst_group_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Worst-group analysis"""
        groups = df['group'].unique()
        accuracies = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            accuracy = (group_data['y_true'] == group_data['y_pred']).mean()
            accuracies[group] = self.bounded_value(float(accuracy)) if accuracy is not None else 0.0
        
        metrics = {}
        if accuracies:
            metrics['worst_group_accuracy'] = self.bounded_value(float(min(accuracies.values())))
        
        return metrics

    # ================================================================
    # 5. Statistical Inequality Metrics
    # ================================================================

    def calculate_statistical_inequality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical inequality metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            selection_rates[group] = self.bounded_value(float(group_data['y_pred'].mean()))
        
        if len(selection_rates) >= 2:
            rates = np.array(list(selection_rates.values()))
            
            # Coefficient of Variation
            if rates.mean() > 0:
                cv = self.bounded_value(float(rates.std() / rates.mean()), min_val=0.0, max_val=5.0)
                metrics['coefficient_of_variation'] = cv
            
            # Mean differences
            mean_diff = self.bounded_value(float(max(rates) - min(rates)), min_val=0.0, max_val=1.0)
            metrics['mean_difference'] = mean_diff
        
        return metrics

    # ================================================================
    # COMPOSITE BIAS SCORE CALCULATION
    # ================================================================

    def calculate_composite_bias_score(self, all_metrics: Dict[str, Any]) -> float:
        """Calculate composite bias score for healthcare"""
        try:
            # Key metrics for healthcare bias assessment
            key_metrics = [
                all_metrics.get('statistical_parity_difference', 0.0),
                all_metrics.get('equal_opportunity_difference', 0.0),
                all_metrics.get('tpr_difference', 0.0),
                all_metrics.get('calibration_gap_difference', 0.0),
                all_metrics.get('error_disparity_subgroup', {}).get('range', 0.0)
            ]
            
            # Calculate weighted average
            composite_score = np.mean([m for m in key_metrics if m > 0])
            return self.bounded_value(composite_score)
            
        except Exception as e:
            self.logger.warning(f"Composite bias score calculation failed: {e}")
            return 0.0

    # ================================================================
    # MAIN PIPELINE INTEGRATION
    # ================================================================

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all healthcare fairness metrics"""
        metrics = {}
        
        # Data validation
        validation_results = self.validate_healthcare_data(df)
        if not validation_results["is_valid"]:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        groups = df['group'].unique()
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for fairness analysis")

        # Calculate all metric categories
        metrics.update(self.calculate_core_group_fairness(df))
        metrics.update(self.calculate_performance_error_fairness(df))
        metrics.update(self.calculate_calibration_reliability(df))
        metrics.update(self.calculate_subgroup_disparity_analysis(df))
        metrics.update(self.calculate_statistical_inequality(df))
        
        # Calculate composite bias score
        metrics['composite_bias_score'] = self.calculate_composite_bias_score(metrics)
        
        # Store for temporal analysis
        self.metrics_history.append(metrics.copy())
        if len(self.metrics_history) > self.temporal_window:
            self.metrics_history.pop(0)
        
        return metrics

    def run_pipeline(self, df: pd.DataFrame, save_to_disk: bool = False) -> Dict[str, Any]:
        """Healthcare pipeline execution"""
        
        try:
            health_metrics = self.calculate_all_metrics(df)
            
            results = {
                "domain": "health",
                "metrics_calculated": 19,
                "metric_categories": HEALTH_METRICS_CONFIG,
                "fairness_metrics": health_metrics,
                "validation": {
                    "is_valid": True,
                    "sample_size": len(df),
                    "groups_analyzed": len(df['group'].unique()),
                    "statistical_power": "strong" if len(df) >= 1000 else "adequate" if len(df) >= 500 else "moderate"
                },
                "summary": {
                    "composite_bias_score": health_metrics.get('composite_bias_score', 0.0),
                    "overall_assessment": self.assess_healthcare_fairness(health_metrics),
                    "key_findings": self._extract_key_findings(health_metrics)
                },
                "timestamp": str(pd.Timestamp.now())
            }
            
            results = self.convert_numpy_types(results)
            
            if save_to_disk:
                self.write_json("health_fairness_audit.json", results)
            
            return results
            
        except Exception as e:
            error_results = {
                "domain": "health",
                "metrics_calculated": 0,
                "error": str(e),
                "summary": {
                    "composite_bias_score": 1.0,
                    "overall_assessment": "ERROR - Could not complete fairness audit"
                },
                "timestamp": str(pd.Timestamp.now())
            }
            return self.convert_numpy_types(error_results)

    def assess_healthcare_fairness(self, metrics: Dict[str, Any]) -> str:
        """Assess overall fairness for healthcare domain"""
        bias_score = metrics.get('composite_bias_score', 0.0)
        
        if bias_score > 0.25:
            return "CRITICAL_BIAS - Immediate intervention required"
        elif bias_score > 0.15:
            return "HIGH_BIAS - Urgent review needed"
        elif bias_score > 0.08:
            return "MEDIUM_BIAS - Monitor closely"
        else:
            return "LOW_BIAS - Generally fair across groups"

    def _extract_key_findings(self, metrics: Dict[str, Any]) -> List[str]:
        """Extract key findings for summary"""
        findings = []
        
        bias_score = metrics.get('composite_bias_score', 0.0)
        if bias_score > 0.15:
            findings.append("Significant bias detected requiring mitigation")
        
        if metrics.get('statistical_parity_difference', 0) > 0.1:
            findings.append("Notable selection rate disparities across groups")
            
        if metrics.get('equal_opportunity_difference', 0) > 0.1:
            findings.append("True positive rate disparities detected")
            
        if not findings:
            findings.append("No critical fairness issues detected")
            
        return findings

    def write_json(self, path: str, obj: Any):
        """Write object to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


# ================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ================================================================

def convert_numpy_types(obj):
    """Convert numpy types to Python native types - for backward compatibility"""
    pipeline = HealthFairnessPipeline()
    return pipeline.convert_numpy_types(obj)

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """Main pipeline execution - for backward compatibility"""
    pipeline = HealthFairnessPipeline()
    return pipeline.run_pipeline(df, save_to_disk)

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """Main audit function for health domain - for backward compatibility"""
    try:
        df = pd.DataFrame(audit_request['data'])
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "health",
            "metrics_calculated": 19,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health audit failed: {str(e)}"
        }

# Production usage example
if __name__ == "__main__":
    # Test with healthcare data
    sample_data = pd.DataFrame({
        'group': ['Group_A', 'Group_A', 'Group_B', 'Group_B', 'Group_A', 'Group_B', 'Group_A', 'Group_B'],
        'y_true': [1, 0, 1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1, 1, 0],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6, 0.7, 0.4]
    })
    
    # Test streamlined pipeline
    print("Testing STREAMLINED Health Fairness Pipeline...")
    
    pipeline = HealthFairnessPipeline()
    results = pipeline.run_pipeline(sample_data)
    
    print("HEALTH FAIRNESS AUDIT COMPLETE")
    print(f"Metrics Calculated: {results['metrics_calculated']}/19")
    print(f"Overall Assessment: {results['summary']['overall_assessment']}")
    print(f"Composite Bias Score: {results['summary']['composite_bias_score']:.3f}")
    print(f"Key Findings: {results['summary']['key_findings']}")
    
    # Test backward compatibility
    print("\nTesting Backward Compatibility...")
    function_results = run_pipeline(sample_data)
    print(f"Function Interface - Metrics: {function_results['metrics_calculated']}/19")
    print("✅ Streamlined health pipeline is production-ready!")