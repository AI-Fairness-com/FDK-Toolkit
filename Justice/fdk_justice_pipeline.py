# ================================================================
# FDK Justice Pipeline - Justice Fairness Metrics
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
import scipy.stats as st
from typing import Dict, List, Any
import json

# ================================================================
# JUSTICE METRICS CONFIGURATION
# ================================================================

JUSTICE_METRICS_CONFIG = {
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

# ================================================================
# CORE PIPELINE FUNCTIONS
# ================================================================

def interpret_prompt(prompt: str) -> Dict[str, Any]:
    """
    Justice-specific prompt interpretation for domain detection.
    
    Args:
        prompt: User input text to analyze for justice domain keywords
        
    Returns:
        Dictionary containing domain detection results and suggested metrics
    """
    justice_keywords = ['justice', 'legal', 'court', 'sentencing', 'recidivism', 
                       'bail', 'parole', 'defendant', 'offender', 'criminal',
                       'arrest', 'conviction', 'judicial', 'law']
    
    if any(keyword in prompt.lower() for keyword in justice_keywords):
        return {
            "domain": "justice",
            "suggested_metrics": list(JUSTICE_METRICS_CONFIG.keys()),
            "interpretation": "Justice system fairness audit focusing on legal decisions, sentencing outcomes, and recidivism predictions"
        }
    return {"domain": "unknown", "suggested_metrics": [], "interpretation": "Domain not recognized"}

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main audit function for justice domain requests.
    
    Args:
        audit_request: Dictionary containing audit request data
        
    Returns:
        Dictionary with audit results or error status
    """
    try:
        # Load data from request
        df = pd.DataFrame(audit_request['data'])
        
        # Execute justice-specific pipeline
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "justice",
            "metrics_calculated": 14,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Justice audit failed: {str(e)}"
        }

# ================================================================
# METRICS CALCULATION FUNCTIONS
# ================================================================

def calculate_justice_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all justice fairness metrics through pipeline stages.
    
    Args:
        df: Pandas DataFrame containing justice data with required columns
        
    Returns:
        Dictionary containing all calculated fairness metrics
        
    Raises:
        ValueError: If required columns are missing or insufficient groups
    """
    metrics = {}
    
    # Basic validation for required columns
    required_cols = ['group', 'y_true', 'y_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Enhanced validation to ensure we're working with Series objects
    for col in required_cols:
        col_data = df[col]
        if not isinstance(col_data, pd.Series):
            raise ValueError(f"Column '{col}' is not a Series, got {type(col_data)}")
    
    # Validate group diversity for fairness analysis
    groups = df['group'].unique()
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for justice fairness analysis")
    
    # Execute all justice metric calculation stages
    metrics.update(calculate_core_group_fairness(df))          # Stage 1
    metrics.update(calculate_calibration_reliability(df))      # Stage 2
    metrics.update(calculate_error_prediction_fairness(df))    # Stage 3
    metrics.update(calculate_statistical_inequality(df))       # Stage 4
    metrics.update(calculate_subgroup_bias_detection(df))      # Stage 5
    metrics.update(calculate_causal_fairness(df))              # Stage 6
    metrics.update(calculate_robustness_fairness(df))          # Stage 7
    metrics.update(calculate_explainability_temporal(df))      # Stage 8
    
    return metrics

def calculate_core_group_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate core group fairness metrics for justice domain.
    
    Includes statistical parity difference and base rate analysis
    across defendant/offender groups.
    
    Args:
        df: DataFrame with group, y_true, and y_pred columns
        
    Returns:
        Dictionary containing core fairness metrics
    """
    metrics = {}
    groups = df['group'].unique()
    
    selection_rates = {}
    base_rates = {}
    
    for group in groups:
        # Use proper pandas boolean indexing for group filtering
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        # Selection Rate (Predicted Positive Outcomes)
        selection_rates[group] = group_data['y_pred'].mean()
        
        # Base Rate (Actual Positive Outcomes)
        base_rates[group] = group_data['y_true'].mean()
    
    # Statistical Parity Difference - Maximum difference in selection rates
    if len(selection_rates) >= 2:
        spd = max(selection_rates.values()) - min(selection_rates.values())
        metrics['statistical_parity_difference'] = spd
        metrics['selection_rates'] = selection_rates
    
    # Base Rate metrics - Differences in actual outcome rates
    if len(base_rates) >= 2:
        base_rate_diff = max(base_rates.values()) - min(base_rates.values())
        metrics['base_rate_difference'] = base_rate_diff
        metrics['base_rates'] = base_rates
    
    return metrics

def calculate_calibration_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate calibration and reliability metrics for justice predictions.
    
    Assesses how well predicted probabilities match actual outcomes
    across different demographic groups.
    
    Args:
        df: DataFrame with justice prediction data
        
    Returns:
        Dictionary containing calibration and reliability metrics
    """
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
            
        # Calibration Gap Analysis (if probability scores available)
        if 'y_prob' in df.columns:
            try:
                y_true_vals = group_data['y_true'].values
                y_prob_vals = group_data['y_prob'].values
                
                # Simple calibration: mean predicted probability vs actual outcome rate
                mean_pred_prob = y_prob_vals.mean()
                actual_rate = y_true_vals.mean()
                calibration_gaps[group] = abs(mean_pred_prob - actual_rate)
            except Exception as e:
                calibration_gaps[group] = 0  # Default on error
        
        # Regression Parity (MSE for continuous predictions)
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            
            # Check if this is a regression task (more than 2 unique values)
            if len(np.unique(y_pred_vals)) > 2:
                mse_values[group] = mean_squared_error(y_true_vals, y_pred_vals)
        except Exception as e:
            mse_values[group] = 0  # Default on error
        
        # Slice AUC Difference - Model performance across subgroups
        if 'y_prob' in df.columns:
            try:
                y_true_vals = group_data['y_true'].values
                y_prob_vals = group_data['y_prob'].values
                
                if len(np.unique(y_true_vals)) > 1:  # Need both classes for AUC
                    auc_scores[group] = roc_auc_score(y_true_vals, y_prob_vals)
            except Exception as e:
                continue  # Skip if AUC calculation fails
    
    # Calculate differences across groups for each metric type
    
    # Calibration Gap Differences
    if calibration_gaps and len(calibration_gaps) > 1:
        valid_calibration = [v for v in calibration_gaps.values() if v is not None]
        if valid_calibration:
            metrics['calibration_gap_difference'] = max(valid_calibration) - min(valid_calibration)
    
    # Regression Parity Differences
    if mse_values and len(mse_values) > 1:
        valid_mse = [v for v in mse_values.values() if v is not None]
        if valid_mse:
            metrics['regression_parity_difference'] = max(valid_mse) - min(valid_mse)
    
    # AUC Score Differences
    if auc_scores and len(auc_scores) > 1:
        valid_auc = [v for v in auc_scores.values() if v is not None]
        if valid_auc:
            metrics['slice_auc_difference'] = max(valid_auc) - min(valid_auc)
    
    return metrics

def calculate_error_prediction_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate error and prediction fairness metrics for justice decisions.
    
    Analyzes false positive/negative rates and predictive parity
    across different defendant groups.
    
    Args:
        df: DataFrame with justice prediction outcomes
        
    Returns:
        Dictionary containing error rate and prediction fairness metrics
    """
    metrics = {}
    groups = df['group'].unique()
    
    # Initialize dictionaries for error rate storage
    fpr_values, fnr_values, fdr_values, for_values, ppv_values, npv_values = {}, {}, {}, {}, {}, {}
    
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        
        if len(group_data) == 0:
            continue
            
        try:
            y_true_vals = group_data['y_true'].values
            y_pred_vals = group_data['y_pred'].values
            
            # Skip if insufficient class diversity for confusion matrix
            if len(np.unique(y_true_vals)) < 2 or len(np.unique(y_pred_vals)) < 2:
                continue
                
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true_vals, y_pred_vals).ravel()
            
            # False Positive Rate (FPR) - False accusations/convictions
            fpr_values[group] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False Negative Rate (FNR) - Missed detections
            fnr_values[group] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # False Discovery Rate (FDR) - False positive proportion
            fdr_values[group] = fp / (fp + tp) if (fp + tp) > 0 else 0
            
            # False Omission Rate (FOR) - False negative proportion  
            for_values[group] = fn / (fn + tn) if (fn + tn) > 0 else 0
            
            # Positive Predictive Value (PPV) - Precision
            ppv_values[group] = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Negative Predictive Value (NPV)
            npv_values[group] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
        except Exception as e:
            continue  # Skip group on calculation error
    
    # Calculate differences across groups for each error metric
    
    # False Positive/Negative Rate Differences
    if fpr_values and len(fpr_values) > 1:
        valid_fpr = [v for v in fpr_values.values() if v is not None]
        valid_fnr = [v for v in fnr_values.values() if v is not None]
        
        if valid_fpr:
            metrics['fpr_difference'] = max(valid_fpr) - min(valid_fpr)
        if valid_fnr:
            metrics['fnr_difference'] = max(valid_fnr) - min(valid_fnr)
    
    # False Discovery/Omission Rate Differences
    if fdr_values and len(fdr_values) > 1:
        valid_fdr = [v for v in fdr_values.values() if v is not None]
        valid_for = [v for v in for_values.values() if v is not None]
        
        if valid_fdr:
            metrics['fdr_difference'] = max(valid_fdr) - min(valid_fdr)
        if valid_for:
            metrics['for_difference'] = max(valid_for) - min(valid_for)
    
    # Predictive Parity Differences (PPV and NPV combined)
    if ppv_values and len(ppv_values) > 1:
        valid_ppv = [v for v in ppv_values.values() if v is not None]
        valid_npv = [v for v in npv_values.values() if v is not None]
        
        if valid_ppv and valid_npv:
            ppv_diff = max(valid_ppv) - min(valid_ppv)
            npv_diff = max(valid_npv) - min(valid_npv)
            metrics['predictive_parity_difference'] = (ppv_diff + npv_diff) / 2
    
    return metrics

def calculate_statistical_inequality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistical inequality metrics for justice outcomes.
    
    Measures variation and disparity in selection rates across groups
    using coefficient of variation and normalized differences.
    
    Args:
        df: DataFrame with justice prediction data
        
    Returns:
        Dictionary containing statistical inequality metrics
    """
    metrics = {}
    groups = df['group'].unique()
    
    selection_rates = {}
    for group in groups:
        group_mask = df['group'] == group
        group_data = df[group_mask]
        selection_rates[group] = group_data['y_pred'].mean()
    
    if len(selection_rates) >= 2:
        rates = np.array(list(selection_rates.values()))
        
        # Coefficient of Variation - Relative standard deviation
        if rates.mean() > 0:
            cv = rates.std() / rates.mean()
            metrics['coefficient_of_variation'] = cv
        
        # Normalized Mean Difference - Scaled by overall mean
        mean_diff = max(rates) - min(rates)
        overall_mean = rates.mean()
        if overall_mean > 0:
            metrics['normalized_mean_difference'] = mean_diff / overall_mean
    
    return metrics

def calculate_subgroup_bias_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect subgroup bias through error rate analysis.
    
    Identifies disparities in prediction accuracy across different
    defendant/offender subgroups in justice system.
    
    Args:
        df: DataFrame with justice prediction outcomes
        
    Returns:
        Dictionary containing subgroup bias detection metrics
    """
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
            error_rates[group] = 1 - accuracy_score(y_true_vals, y_pred_vals)
        except Exception as e:
            error_rates[group] = 0  # Default on calculation error
    
    # Calculate error rate disparities across groups
    if error_rates and len(error_rates) > 1:
        valid_errors = [v for v in error_rates.values() if v is not None]
        if valid_errors:
            metrics['error_rate_difference'] = max(valid_errors) - min(valid_errors)
            metrics['error_rate_ratio'] = max(valid_errors) / min(valid_errors) if min(valid_errors) > 0 else float('inf')
    
    return metrics

def calculate_causal_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate simplified causal fairness metrics for justice domain.
    
    Note: In production, this would require proper causal models.
    This implementation uses selection rates as a proxy.
    
    Args:
        df: DataFrame with justice prediction data
        
    Returns:
        Dictionary containing causal fairness metrics
    """
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        # Simplified causal effect difference using selection rates as proxy
        selection_rates = {}
        for group in groups:
            group_mask = df['group'] == group
            selection_rates[group] = df[group_mask]['y_pred'].mean()
        
        if len(selection_rates) >= 2:
            causal_effect = max(selection_rates.values()) - min(selection_rates.values())
            return {'causal_effect_difference': causal_effect}
    
    return {'causal_effect_difference': 0}

def calculate_robustness_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate robustness and worst-case fairness metrics.
    
    Includes worst-group accuracy and composite bias score
    based on maximum disparity across all metrics.
    
    Args:
        df: DataFrame with justice prediction data
        
    Returns:
        Dictionary containing robustness and composite fairness metrics
    """
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
            accuracies[group] = accuracy_score(y_true_vals, y_pred_vals)
        except Exception as e:
            accuracies[group] = 0  # Default on calculation error
    
    if accuracies and len(accuracies) > 1:
        valid_accuracies = [v for v in accuracies.values() if v is not None]
        if valid_accuracies:
            # Worst-group accuracy - minimum performance across groups
            metrics['worst_group_accuracy'] = min(valid_accuracies)
            accuracy_range = max(valid_accuracies) - min(valid_accuracies)
            
            # COMPOSITE BIAS SCORE: Use maximum severity across all bias metrics
            bias_metrics = [
                accuracy_range,
                metrics.get('statistical_parity_difference', 0),
                metrics.get('fpr_difference', 0),
                metrics.get('fnr_difference', 0),
                metrics.get('fdr_difference', 0),
                metrics.get('for_difference', 0),
                metrics.get('predictive_parity_difference', 0),
                metrics.get('error_rate_difference', 0),
                metrics.get('causal_effect_difference', 0)
            ]
            
            # Remove zero values to avoid dilution, then take maximum disparity
            non_zero_biases = [b for b in bias_metrics if b > 0]
            if non_zero_biases:
                metrics['composite_bias_score'] = max(non_zero_biases)
            else:
                metrics['composite_bias_score'] = 0  # No detectable bias
    
    return metrics

def calculate_explainability_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate explainability and temporal fairness metrics (simplified).
    
    Placeholder implementation - in production would use SHAP values
    and temporal analysis for justice decision patterns.
    
    Args:
        df: DataFrame with justice prediction data
        
    Returns:
        Dictionary containing explainability and temporal metrics
    """
    groups = df['group'].unique()
    
    if len(groups) >= 2:
        # Simplified feature importance disparity using group means
        feature_importance_gap = 0
        
        # Calculate mean differences across groups for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
        
        if len(numeric_cols) > 0:
            gap_sum = 0
            for col in numeric_cols:
                group_means = []
                for group in groups:
                    group_mask = df['group'] == group
                    group_means.append(df[group_mask][col].mean())
                
                if len(group_means) >= 2:
                    gap_sum += max(group_means) - min(group_means)
            
            if len(numeric_cols) > 0:
                feature_importance_gap = gap_sum / len(numeric_cols)
        
        return {'shap_disparity': feature_importance_gap}
    
    return {'shap_disparity': 0}

# ================================================================
# MAIN PIPELINE EXECUTION
# ================================================================

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """
    Main justice pipeline execution with comprehensive error handling.
    
    Args:
        df: Pandas DataFrame containing justice data to audit
        save_to_disk: Whether to save results to disk (default: True)
        
    Returns:
        Dictionary containing complete justice fairness audit results
    """
    
    try:
        # Calculate all justice fairness metrics
        justice_metrics = calculate_justice_metrics(df)
        
        # Build comprehensive results structure
        results = {
            "domain": "justice",
            "metrics_calculated": 14,
            "metric_categories": JUSTICE_METRICS_CONFIG,
            "fairness_metrics": justice_metrics,
            "summary": {
                "composite_bias_score": justice_metrics.get('composite_bias_score', 0),
                "overall_assessment": assess_justice_fairness(justice_metrics)
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        # Graceful error handling with structured error response
        return {
            "domain": "justice",
            "metrics_calculated": 0,
            "error": str(e),
            "summary": {
                "composite_bias_score": 1.0,  # Maximum bias score on error
                "overall_assessment": "ERROR - Could not complete justice audit"
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }

def assess_justice_fairness(metrics: Dict[str, Any]) -> str:
    """
    Assess overall fairness for justice domain based on composite bias score.
    
    Args:
        metrics: Dictionary containing calculated fairness metrics
        
    Returns:
        String assessment of fairness severity level
    """
    bias_score = metrics.get('composite_bias_score', 0)
    
    if bias_score > 0.15:
        return "HIGH_BIAS - Significant fairness concerns in justice decisions"
    elif bias_score > 0.05:
        return "MEDIUM_BIAS - Moderate fairness concerns detected"  
    else:
        return "LOW_BIAS - Generally fair across defendant groups"

# ================================================================
# TESTING AND COMPATIBILITY
# ================================================================

if __name__ == "__main__":
    # Test with sample justice data
    sample_data = pd.DataFrame({
        'group': ['Group_A', 'Group_A', 'Group_B', 'Group_B', 'Group_A', 'Group_B'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
    })
    
    # Execute pipeline and display results
    results = run_pipeline(sample_data)
    print("Justice Pipeline Test Results:")
    print(json.dumps(results, indent=2))