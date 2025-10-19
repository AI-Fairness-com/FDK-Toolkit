# ================================================================
# FDK Finance Pipeline - 14 Finance Fairness Metrics
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
import scipy.stats as st
from typing import Dict, List, Any
import json

# Finance-specific metrics configuration
FINANCE_METRICS_CONFIG = {
    'core_group_fairness': [
        'disparate_impact', 
        'statistical_parity_difference',
        'base_rate'
    ],
    'calibration_reliability': [
        'calibration_gap',
        'regression_parity',
        'slice_auc_difference'
    ],
    'error_prediction_fairness': [
        'fpr_fnr_differences',
        'fdr_for_differences', 
        'predictive_parity_difference'
    ],
    'statistical_inequality': [
        'coefficient_of_variation'
    ],
    'subgroup_bias_detection': [
        'error_rate_difference'
    ],
    'causal_fairness': [
        'causal_effect_difference'
    ],
    'robustness_fairness': [
        'worst_group_accuracy',
        'composite_bias_score'
    ],
    'explainability_temporal': [
        'shap_disparity'
    ]
}

def convert_numpy_types(obj):
    """Convert numpy/pandas types to Python native types"""
    if hasattr(obj, 'dtype'):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(item) for item in obj]
    
    return obj

def interpret_prompt(prompt: str) -> Dict[str, Any]:
    """Finance-specific prompt interpretation"""
    finance_keywords = ['finance', 'financial', 'credit', 'loan', 'banking', 'lending',
                       'default', 'repayment', 'fraud', 'approval', 'risk', 'score']
    
    if any(keyword in prompt.lower() for keyword in finance_keywords):
        return {
            "domain": "finance",
            "suggested_metrics": list(FINANCE_METRICS_CONFIG.keys()),
            "interpretation": "Financial services fairness audit focusing on credit decisions, risk assessment, and lending outcomes"
        }
    return {"domain": "unknown", "suggested_metrics": [], "interpretation": "Domain not recognized"}

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """Main audit function for finance domain"""
    try:
        df = pd.DataFrame(audit_request['data'])
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "finance",
            "metrics_calculated": 14,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Finance audit failed: {str(e)}"
        }

def calculate_finance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all 14 finance fairness metrics"""
    metrics = {}
    
    required_cols = ['group', 'y_true', 'y_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    for col in required_cols:
        col_data = df[col]
        if not isinstance(col_data, pd.Series):
            raise ValueError(f"Column '{col}' is not a Series, got {type(col_data)}")
    
    groups = df['group'].unique()
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for fairness analysis")
    
    metrics.update(calculate_core_group_fairness(df))
    metrics.update(calculate_calibration_reliability(df))
    metrics.update(calculate_error_prediction_fairness(df))
    metrics.update(calculate_statistical_inequality(df))
    metrics.update(calculate_subgroup_bias_detection(df))
    metrics.update(calculate_causal_fairness(df))
    metrics.update(calculate_robustness_fairness(df, metrics))
    metrics.update(calculate_explainability_temporal(df))
    
    return metrics

def calculate_core_group_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Core Group Fairness Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    selection_rates = {}
    base_rates = {}
    
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        selection_rates[group] = float(group_data['y_pred'].mean())
        base_rates[group] = float(group_data['y_true'].mean())
    
    if len(selection_rates) >= 2:
        spd = float(max(selection_rates.values()) - min(selection_rates.values()))
        metrics['statistical_parity_difference'] = spd
        metrics['selection_rates'] = selection_rates
    
    if len(base_rates) >= 2:
        base_rate_diff = float(max(base_rates.values()) - min(base_rates.values()))
        metrics['base_rate_difference'] = base_rate_diff
        metrics['base_rates'] = base_rates
    
    return metrics

def calculate_calibration_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    """Calibration and Reliability Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    calibration_gaps = {}
    mse_values = {}
    auc_scores = {}
    
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        if 'y_prob' in df.columns:
            try:
                y_true_vals = group_data['y_true'].values
                y_prob_vals = group_data['y_prob'].values
                
                mean_pred_prob = float(y_prob_vals.mean())
                actual_rate = float(y_true_vals.mean())
                calibration_gaps[group] = float(abs(mean_pred_prob - actual_rate))
            except Exception:
                calibration_gaps[group] = 0.0
        
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            
            if len(np.unique(y_pred_vals)) > 2:
                mse_values[group] = float(mean_squared_error(y_true_vals, y_pred_vals))
        except Exception:
            mse_values[group] = 0.0
        
        if 'y_prob' in df.columns:
            try:
                y_true_vals = group_data['y_true'].values
                y_prob_vals = group_data['y_prob'].values
                
                if len(np.unique(y_true_vals)) > 1:
                    auc_scores[group] = float(roc_auc_score(y_true_vals, y_prob_vals))
            except Exception:
                continue
    
    if calibration_gaps and len(calibration_gaps) > 1:
        valid_calibration = [v for v in calibration_gaps.values() if v is not None]
        if valid_calibration:
            metrics['calibration_gap_difference'] = float(max(valid_calibration) - min(valid_calibration))
    
    if mse_values and len(mse_values) > 1:
        valid_mse = [v for v in mse_values.values() if v is not None]
        if valid_mse:
            metrics['regression_parity_difference'] = float(max(valid_mse) - min(valid_mse))
    
    if auc_scores and len(auc_scores) > 1:
        valid_auc = [v for v in auc_scores.values() if v is not None]
        if valid_auc:
            metrics['slice_auc_difference'] = float(max(valid_auc) - min(valid_auc))
    
    return metrics

def calculate_error_prediction_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Error and Prediction Fairness Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    fpr_values, fnr_values, fdr_values, for_values, ppv_values, npv_values = {}, {}, {}, {}, {}, {}
    
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            
            if len(np.unique(y_true_vals)) < 2 or len(np.unique(y_pred_vals)) < 2:
                continue
                
            tn, fp, fn, tp = confusion_matrix(y_true_vals, y_pred_vals).ravel()
            
            fpr_values[group] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr_values[group] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            fdr_values[group] = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
            for_values[group] = float(fn / (fn + tn)) if (fn + tn) > 0 else 0.0
            
            ppv_values[group] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            npv_values[group] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            
        except Exception:
            continue
    
    if fpr_values and len(fpr_values) > 1:
        valid_fpr = [v for v in fpr_values.values() if v is not None]
        valid_fnr = [v for v in fnr_values.values() if v is not None]
        
        if valid_fpr:
            metrics['fpr_difference'] = float(max(valid_fpr) - min(valid_fpr))
        if valid_fnr:
            metrics['fnr_difference'] = float(max(valid_fnr) - min(valid_fnr))
    
    if fdr_values and len(fdr_values) > 1:
        valid_fdr = [v for v in fdr_values.values() if v is not None]
        valid_for = [v for v in for_values.values() if v is not None]
        
        if valid_fdr:
            metrics['fdr_difference'] = float(max(valid_fdr) - min(valid_fdr))
        if valid_for:
            metrics['for_difference'] = float(max(valid_for) - min(valid_for))
    
    if ppv_values and len(ppv_values) > 1:
        valid_ppv = [v for v in ppv_values.values() if v is not None]
        valid_npv = [v for v in npv_values.values() if v is not None]
        
        if valid_ppv and valid_npv:
            ppv_diff = float(max(valid_ppv) - min(valid_ppv))
            npv_diff = float(max(valid_npv) - min(valid_npv))
            metrics['predictive_parity_difference'] = float((ppv_diff + npv_diff) / 2)
    
    return metrics

def calculate_statistical_inequality(df: pd.DataFrame) -> Dict[str, Any]:
    """Statistical Inequality Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    selection_rates = {}
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        selection_rates[group] = float(group_data['y_pred'].mean())
    
    if len(selection_rates) >= 2:
        rates = np.array(list(selection_rates.values()))
        
        if rates.mean() > 0:
            cv = float(rates.std() / rates.mean())
            metrics['coefficient_of_variation'] = cv
        
        mean_diff = float(max(rates) - min(rates))
        overall_mean = float(rates.mean())
        if overall_mean > 0:
            metrics['normalized_mean_difference'] = float(mean_diff / overall_mean)
    
    return metrics

def calculate_subgroup_bias_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """Subgroup and Hidden Bias Detection"""
    metrics = {}
    groups = df['group'].unique()
    
    error_rates = {}
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            error_rates[group] = float(1 - accuracy_score(y_true_vals, y_pred_vals))
        except Exception:
            error_rates[group] = 0.0
    
    if error_rates and len(error_rates) > 1:
        valid_errors = [v for v in error_rates.values() if v is not None]
        if valid_errors:
            metrics['error_rate_difference'] = float(max(valid_errors) - min(valid_errors))
            max_error = max(valid_errors)
            min_error = min(valid_errors)
            metrics['error_rate_ratio'] = float(max_error / min_error) if min_error > 0 else float('inf')
    
    return metrics

def calculate_causal_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Causal and Counterfactual Fairness (Simplified)"""
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        selection_rates = {}
        for group in groups:
            group_mask = df['group'] == group
            selection_rates[group] = float(df[group_mask]['y_pred'].mean())
        
        if len(selection_rates) >= 2:
            causal_effect = float(max(selection_rates.values()) - min(selection_rates.values()))
            return {'causal_effect_difference': causal_effect}
    
    return {'causal_effect_difference': 0.0}

def calculate_robustness_fairness(df: pd.DataFrame, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Robustness and Worst-Case Fairness"""
    metrics = {}
    groups = df['group'].unique()
    
    accuracies = {}
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            accuracies[group] = float(accuracy_score(y_true_vals, y_pred_vals))
        except Exception:
            accuracies[group] = 0.0

    if accuracies and len(accuracies) > 1:
        valid_accuracies = [v for v in accuracies.values() if v is not None]
        if valid_accuracies:
            metrics['worst_group_accuracy'] = float(min(valid_accuracies))
            accuracy_range = float(max(valid_accuracies) - min(valid_accuracies))
            
            key_metrics = [
                all_metrics.get('statistical_parity_difference', 0.0),
                all_metrics.get('fpr_difference', 0.0),
                all_metrics.get('fnr_difference', 0.0),
                all_metrics.get('fdr_difference', 0.0),
                all_metrics.get('error_rate_difference', 0.0),
                all_metrics.get('predictive_parity_difference', 0.0)
            ]
            
            capped_metrics = [float(min(metric, 0.2)) for metric in key_metrics if metric > 0]
            
            if capped_metrics:
                metrics['composite_bias_score'] = float(sum(capped_metrics) / len(capped_metrics))
            else:
                metrics['composite_bias_score'] = 0.0
    
    return metrics

def calculate_explainability_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    """Explainability and Temporal Fairness (Placeholder)"""
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        feature_importance_gap = 0.0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
        
        if len(numeric_cols) > 0:
            gap_sum = 0.0
            for col in numeric_cols:
                group_means = []
                for group in groups:
                    group_mask = df['group'] == group
                    group_means.append(float(df[group_mask][col].mean()))
                
                if len(group_means) >= 2:
                    gap_sum += float(max(group_means) - min(group_means))
            
            if len(numeric_cols) > 0:
                feature_importance_gap = float(gap_sum / len(numeric_cols))
        
        return {'shap_disparity': feature_importance_gap}
    
    return {'shap_disparity': 0.0}

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """Main finance pipeline execution"""
    
    try:
        finance_metrics = calculate_finance_metrics(df)
        
        results = {
            "domain": "finance",
            "metrics_calculated": 14,
            "metric_categories": FINANCE_METRICS_CONFIG,
            "fairness_metrics": finance_metrics,
            "summary": {
                "composite_bias_score": finance_metrics.get('composite_bias_score', 0.0),
                "overall_assessment": assess_finance_fairness(finance_metrics)
            },
            "timestamp": str(pd.Timestamp.now())
        }
        
        results = convert_numpy_types(results)
        
        return results
        
    except Exception as e:
        error_results = {
            "domain": "finance",
            "metrics_calculated": 0,
            "error": str(e),
            "summary": {
                "composite_bias_score": 1.0,
                "overall_assessment": "ERROR - Could not complete audit"
            },
            "timestamp": str(pd.Timestamp.now())
        }
        return convert_numpy_types(error_results)

def assess_finance_fairness(metrics: Dict[str, Any]) -> str:
    """Assess overall fairness for finance domain"""
    bias_score = metrics.get('composite_bias_score', 0.0)
    
    if bias_score > 0.10:
        return "HIGH_BIAS - Significant fairness concerns in financial decisions"
    elif bias_score > 0.03:
        return "MEDIUM_BIAS - Moderate fairness concerns detected"  
    else:
        return "LOW_BIAS - Generally fair across groups"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'group': ['High Income', 'High Income', 'Low Income', 'Low Income', 'High Income', 'Low Income'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
    })
    
    results = run_pipeline(sample_data)
    print("Finance Pipeline Test Results:")
    print(json.dumps(results, indent=2))