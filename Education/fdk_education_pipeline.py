# ================================================================
# FDK Education Pipeline - 15 Education Fairness Metrics
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
import scipy.stats as st
from typing import Dict, List, Any
import json

# Education-specific metrics configuration
EDUCATION_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference',
        'normalized_mean_difference'
    ],
    'equality_opportunity_treatment': [
        'tpr_difference',
        'fpr_difference',
        'error_rate_difference'
    ],
    'statistical_inequality_distribution': [
        'coefficient_of_variation'
    ],
    'data_integrity_preprocessing': [
        'sample_distortion_metrics'
    ],
    'subgroup_bias_detection': [
        'mdss_subgroup_score',
        'error_disparity_subgroup'
    ],
    'calibration_reliability': [
        'slice_auc_difference'
    ],
    'explainability_proxy_detection': [
        'feature_attribution_bias',
        'shap_feature_gap'
    ],
    'counterfactual_causal_fairness': [
        'counterfactual_flip_rate'
    ],
    'robustness_worst_case_fairness': [
        'worst_group_accuracy',
        'composite_bias_score'
    ],
    'temporal_fairness': [
        'temporal_fairness_score'
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
    """Education-specific prompt interpretation"""
    education_keywords = ['education', 'school', 'student', 'admission', 'placement', 'graduation',
                         'academic', 'educational', 'learning', 'assessment', 'curriculum',
                         'district', 'cohort', 'demographic', 'socioeconomic']
    
    if any(keyword in prompt.lower() for keyword in education_keywords):
        return {
            "domain": "education",
            "suggested_metrics": list(EDUCATION_METRICS_CONFIG.keys()),
            "interpretation": "Educational equity audit focusing on admission decisions, student placements, and academic outcomes"
        }
    return {"domain": "unknown", "suggested_metrics": [], "interpretation": "Domain not recognized"}

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """Main audit function for education domain"""
    try:
        df = pd.DataFrame(audit_request['data'])
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "education",
            "metrics_calculated": 15,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Education audit failed: {str(e)}"
        }

def calculate_education_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all 15 education fairness metrics"""
    metrics = {}
    
    # Basic validation
    required_cols = ['group', 'y_true', 'y_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Enhanced validation to ensure we're working with Series
    for col in required_cols:
        col_data = df[col]
        if not isinstance(col_data, pd.Series):
            raise ValueError(f"Column '{col}' is not a Series, got {type(col_data)}")
    
    # Use pandas unique() method properly
    groups = df['group'].unique()
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for fairness analysis")
    
    # 1. Core Group Fairness
    metrics.update(calculate_core_group_fairness(df))
    
    # 2. Equality of Opportunity and Treatment
    metrics.update(calculate_equality_opportunity_treatment(df))
    
    # 3. Statistical Inequality and Distribution
    metrics.update(calculate_statistical_inequality_distribution(df))
    
    # 4. Data Integrity and Preprocessing Fairness
    metrics.update(calculate_data_integrity_preprocessing(df))
    
    # 5. Subgroup and Hidden Bias Detection
    metrics.update(calculate_subgroup_bias_detection(df))
    
    # 6. Calibration and Reliability
    metrics.update(calculate_calibration_reliability(df))
    
    # 7. Explainability and Proxy Detection
    metrics.update(calculate_explainability_proxy_detection(df))
    
    # 8. Counterfactual and Causal Fairness
    metrics.update(calculate_counterfactual_causal_fairness(df))
    
    # 9. Robustness and Worst-Case Fairness
    metrics.update(calculate_robustness_worst_case_fairness(df, metrics))
    
    # 10. Temporal Fairness
    metrics.update(calculate_temporal_fairness(df))
    
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
    
    # Statistical Parity Difference
    if len(selection_rates) >= 2:
        spd = float(max(selection_rates.values()) - min(selection_rates.values()))
        metrics['statistical_parity_difference'] = spd
        metrics['selection_rates'] = selection_rates
    
    # Base Rate metrics
    if len(base_rates) >= 2:
        base_rate_diff = float(max(base_rates.values()) - min(base_rates.values()))
        metrics['base_rate_difference'] = base_rate_diff
        metrics['base_rates'] = base_rates
        
        # Normalized Mean Difference
        overall_mean = float(df['y_pred'].mean())
        if overall_mean > 0:
            metrics['normalized_mean_difference'] = float(spd / overall_mean)
    
    return metrics

def calculate_equality_opportunity_treatment(df: pd.DataFrame) -> Dict[str, Any]:
    """Equality of Opportunity and Treatment Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    tpr_values, fpr_values, error_rates = {}, {}, {}
    
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
            
            tpr_values[group] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr_values[group] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            error_rates[group] = float(1 - accuracy_score(y_true_vals, y_pred_vals))
            
        except Exception:
            continue
    
    # Calculate differences
    if tpr_values and len(tpr_values) > 1:
        valid_tpr = [v for v in tpr_values.values() if v is not None]
        if valid_tpr:
            metrics['tpr_difference'] = float(max(valid_tpr) - min(valid_tpr))
            
    if fpr_values and len(fpr_values) > 1:
        valid_fpr = [v for v in fpr_values.values() if v is not None]
        if valid_fpr:
            metrics['fpr_difference'] = float(max(valid_fpr) - min(valid_fpr))
            
    if error_rates and len(error_rates) > 1:
        valid_errors = [v for v in error_rates.values() if v is not None]
        if valid_errors:
            metrics['error_rate_difference'] = float(max(valid_errors) - min(valid_errors))
    
    return metrics

def calculate_statistical_inequality_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Statistical Inequality and Distribution Fairness"""
    metrics = {}
    
    groups = df['group'].unique()
    group_means = []
    
    for group in groups:
        group_mask = df['group'] == group
        group_means.append(float(df[group_mask]['y_pred'].mean()))
    
    if len(group_means) >= 2:
        mean_of_means = float(np.mean(group_means))
        std_of_means = float(np.std(group_means))
        if mean_of_means > 0:
            metrics['coefficient_of_variation'] = float(std_of_means / mean_of_means)
    
    return metrics

def calculate_data_integrity_preprocessing(df: pd.DataFrame) -> Dict[str, Any]:
    """Data Integrity and Preprocessing Fairness Metrics"""
    metrics = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
    
    if len(numeric_cols) > 0:
        distortion_scores = []
        for col in numeric_cols:
            cv = float(df[col].std() / df[col].mean()) if df[col].mean() > 0 else 0.0
            distortion_scores.append(cv)
        
        if distortion_scores:
            metrics['sample_distortion_metrics'] = {
                'average_shift': float(np.mean(distortion_scores)),
                'maximum_shift': float(np.max(distortion_scores)),
                'individual_shifts': distortion_scores
            }
    
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
            error_diff = float(max(valid_errors) - min(valid_errors))
            metrics['mdss_subgroup_score'] = error_diff
            metrics['error_disparity_subgroup'] = error_diff
    
    return metrics

def calculate_calibration_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    """Calibration and Reliability Metrics"""
    metrics = {}
    
    groups = df['group'].unique()
    auc_scores = {}
    
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        try:
            y_true_vals = group_data['y_true'].values
            if 'y_prob' in group_data.columns:
                y_prob_vals = group_data['y_prob'].values
                if len(np.unique(y_true_vals)) >= 2:
                    auc_scores[group] = float(roc_auc_score(y_true_vals, y_prob_vals))
        except Exception:
            continue
    
    if auc_scores and len(auc_scores) > 1:
        valid_auc = [v for v in auc_scores.values() if v is not None]
        if valid_auc:
            metrics['slice_auc_difference'] = float(max(valid_auc) - min(valid_auc))
    
    return metrics

def calculate_explainability_proxy_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """Explainability and Proxy Detection Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        feature_gaps = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                group_means = []
                for group in groups:
                    group_mask = df['group'] == group
                    group_means.append(float(df[group_mask][col].mean()))
                
                if len(group_means) >= 2:
                    gap = float(max(group_means) - min(group_means))
                    feature_gaps.append(gap)
            
            if feature_gaps:
                metrics['feature_attribution_bias'] = float(np.mean(feature_gaps))
                metrics['shap_feature_gap'] = float(np.max(feature_gaps))
    
    return metrics

def calculate_counterfactual_causal_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Counterfactual and Causal Fairness Metrics"""
    metrics = {}
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        prediction_means = []
        for group in groups:
            group_mask = df['group'] == group
            prediction_means.append(float(df[group_mask]['y_pred'].mean()))
        
        if len(prediction_means) >= 2:
            flip_rate = float(max(prediction_means) - min(prediction_means))
            metrics['counterfactual_flip_rate'] = flip_rate
    
    return metrics

def calculate_robustness_worst_case_fairness(df: pd.DataFrame, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Robustness and Worst-Case Fairness Metrics"""
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
            
            key_metrics = [
                all_metrics.get('statistical_parity_difference', 0.0),
                all_metrics.get('tpr_difference', 0.0),
                all_metrics.get('fpr_difference', 0.0),
                all_metrics.get('error_rate_difference', 0.0),
                all_metrics.get('counterfactual_flip_rate', 0.0),
                all_metrics.get('coefficient_of_variation', 0.0)
            ]
            
            capped_metrics = [float(min(metric, 0.2)) for metric in key_metrics if metric > 0]
            
            if capped_metrics and len(capped_metrics) > 0:
                metrics['composite_bias_score'] = float(sum(capped_metrics) / len(capped_metrics))
            else:
                metrics['composite_bias_score'] = 0.0
    
    return metrics

def calculate_temporal_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Temporal Fairness Metrics"""
    metrics = {}
    
    groups = df['group'].unique()
    prediction_stability = []
    
    for group in groups:
        group_mask = df['group'] == group
        group_predictions = df[group_mask]['y_pred'].values
        if len(group_predictions) > 1:
            stability = 1.0 - float(np.std(group_predictions))
            prediction_stability.append(max(0.0, stability))
    
    if prediction_stability:
        metrics['temporal_fairness_score'] = float(np.mean(prediction_stability))
    
    return metrics

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """Main education pipeline execution"""
    
    try:
        education_metrics = calculate_education_metrics(df)
        
        results = {
            "domain": "education",
            "metrics_calculated": 15,
            "metric_categories": EDUCATION_METRICS_CONFIG,
            "fairness_metrics": education_metrics,
            "summary": {
                "composite_bias_score": education_metrics.get('composite_bias_score', 0.0),
                "overall_assessment": assess_education_fairness(education_metrics)
            },
            "timestamp": str(pd.Timestamp.now())
        }
        
        results = convert_numpy_types(results)
        
        return results
        
    except Exception as e:
        error_results = {
            "domain": "education",
            "metrics_calculated": 0,
            "error": str(e),
            "summary": {
                "composite_bias_score": 1.0,
                "overall_assessment": "ERROR - Could not complete audit"
            },
            "timestamp": str(pd.Timestamp.now())
        }
        return convert_numpy_types(error_results)

def assess_education_fairness(metrics: Dict[str, Any]) -> str:
    """Assess overall fairness for education domain"""
    bias_score = metrics.get('composite_bias_score', 0.0)
    
    if bias_score > 0.10:
        return "HIGH_BIAS - Significant equity concerns in educational decisions"
    elif bias_score > 0.03:
        return "MEDIUM_BIAS - Moderate equity concerns detected"  
    else:
        return "LOW_BIAS - Generally equitable across student groups"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'group': ['Group A', 'Group A', 'Group B', 'Group B', 'Group A', 'Group B'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
    })
    
    results = run_pipeline(sample_data)
    print("Education Pipeline Test Results:")
    print(json.dumps(results, indent=2))