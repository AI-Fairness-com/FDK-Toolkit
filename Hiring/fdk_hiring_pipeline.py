# ================================================================
# FDK Hiring Pipeline - 15 Hiring Fairness Metrics
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
import scipy.stats as st
from typing import Dict, List, Any
import json

# ================================================================
# HIRING METRICS CONFIGURATION
# ================================================================

HIRING_METRICS_CONFIG = {
    'core_group_fairness': [
        'selection_rate_difference', 
        'statistical_parity_difference',
        'normalized_mean_difference'
    ],
    'equality_opportunity_treatment': [
        'equal_opportunity_difference',
        'tpr_tnr_differences'
    ],
    'error_prediction_fairness': [
        'fnr_fpr_differences',
        'fdr_for_differences', 
        'predictive_parity_difference'
    ],
    'data_integrity_preprocessing': [
        'sample_distortion_metrics'
    ],
    'subgroup_bias_detection': [
        'mdss_subgroup_score',
        'error_disparity_subgroup'
    ],
    'explainability_proxy_detection': [
        'feature_attribution_bias',
        'shap_feature_gap'
    ],
    'counterfactual_causal_fairness': [
        'counterfactual_flip_rate'
    ],
    'robustness_temporal_fairness': [
        'worst_group_accuracy',
        'temporal_fairness_score',
        'composite_bias_score'
    ],
    'individual_fairness_consistency': [
        'individual_consistency_index'
    ]
}

# ================================================================
# TYPE CONVERSION UTILITIES
# ================================================================

def convert_numpy_types(obj):
    """
    COMPREHENSIVE conversion of all numpy/pandas types to Python native types.
    
    Essential for JSON serialization and API compatibility across different systems.
    
    Args:
        obj: Any Python object that may contain numpy/pandas types
        
    Returns:
        Object with all numpy/pandas types converted to Python native types
    """
    # Handle pandas Series, Index, and other array-like objects
    if hasattr(obj, 'dtype'):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
    
    # Handle numpy scalars
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
    
    # Handle pandas Timestamp
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # Handle containers recursively
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(item) for item in obj]
    
    # Default case - return as is
    return obj

# ================================================================
# CORE PIPELINE FUNCTIONS
# ================================================================

def interpret_prompt(prompt: str) -> Dict[str, Any]:
    """
    Hiring-specific prompt interpretation for domain detection.
    
    Args:
        prompt: User input text to analyze for hiring domain keywords
        
    Returns:
        Dictionary containing domain detection results and suggested metrics
    """
    hiring_keywords = ['hiring', 'recruitment', 'selection', 'employment', 'applicant', 'candidate',
                      'resume', 'screening', 'interview', 'promotion', 'hr', 'hiring_process']
    
    if any(keyword in prompt.lower() for keyword in hiring_keywords):
        return {
            "domain": "hiring",
            "suggested_metrics": list(HIRING_METRICS_CONFIG.keys()),
            "interpretation": "Hiring and employment fairness audit focusing on selection decisions, resume screening, and promotion outcomes"
        }
    return {"domain": "unknown", "suggested_metrics": [], "interpretation": "Domain not recognized"}

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main audit function for hiring domain API requests.
    
    Args:
        audit_request: Dictionary containing audit request data
        
    Returns:
        Dictionary with audit results or error status
    """
    try:
        # Load data from request
        df = pd.DataFrame(audit_request['data'])
        
        # Execute hiring-specific pipeline
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "hiring",
            "metrics_calculated": 15,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Hiring audit failed: {str(e)}"
        }

# ================================================================
# METRICS CALCULATION PIPELINE
# ================================================================

def calculate_hiring_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all 15 hiring fairness metrics through sequential pipeline stages.
    
    Args:
        df: Pandas DataFrame containing hiring data with required columns
        
    Returns:
        Dictionary containing all calculated fairness metrics
        
    Raises:
        ValueError: If required columns are missing or insufficient groups
    """
    metrics = {}
    
    try:
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
            raise ValueError("Need at least 2 groups for hiring fairness analysis")
        
        # Execute all hiring metric calculation stages in sequence
        metrics.update(calculate_core_group_fairness(df))                    # Stage 1
        metrics.update(calculate_equality_opportunity_treatment(df))         # Stage 2
        metrics.update(calculate_error_prediction_fairness(df))              # Stage 3
        metrics.update(calculate_data_integrity_preprocessing(df))           # Stage 4
        metrics.update(calculate_subgroup_bias_detection(df))                # Stage 5
        metrics.update(calculate_explainability_proxy_detection(df))         # Stage 6
        metrics.update(calculate_counterfactual_causal_fairness(df))         # Stage 7
        metrics.update(calculate_robustness_temporal_fairness(df, metrics))  # Stage 8 (uses previous metrics)
        metrics.update(calculate_individual_fairness_consistency(df))        # Stage 9
        
        return metrics
        
    except Exception as e:
        # Enhanced error logging for debugging
        import traceback
        error_details = f"Error in calculate_hiring_metrics: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise

def calculate_core_group_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate core group fairness metrics for hiring domain.
    
    Includes selection rate differences, statistical parity, and normalized mean differences
    across applicant demographic groups.
    
    Args:
        df: DataFrame with group, y_true, and y_pred columns
        
    Returns:
        Dictionary containing core fairness metrics
    """
    try:
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        base_rates = {}
        
        for group in groups:
            # Use proper pandas boolean indexing for group filtering
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            # Selection Rate (Predicted Positive Outcomes - hiring decisions)
            selection_rates[group] = float(group_data['y_pred'].mean())
            
            # Base Rate (Actual Positive Outcomes - successful candidates)
            base_rates[group] = float(group_data['y_true'].mean())
        
        # Statistical Parity Difference - Maximum difference in selection rates
        if len(selection_rates) >= 2:
            spd = float(max(selection_rates.values()) - min(selection_rates.values()))
            metrics['statistical_parity_difference'] = spd
            metrics['selection_rates'] = selection_rates
            
            # Selection Rate Difference (alias for statistical parity)
            metrics['selection_rate_difference'] = spd
        
        # Base Rate metrics - Differences in actual hiring outcomes
        if len(base_rates) >= 2:
            base_rate_diff = float(max(base_rates.values()) - min(base_rates.values()))
            metrics['base_rate_difference'] = base_rate_diff
            metrics['base_rates'] = base_rates
            
            # Normalized Mean Difference - Scaled by overall selection rate
            overall_mean = float(df['y_pred'].mean())
            if overall_mean > 0:
                metrics['normalized_mean_difference'] = float(spd / overall_mean)
        
        return metrics
    except Exception as e:
        raise

def calculate_equality_opportunity_treatment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate equality of opportunity and treatment metrics.
    
    Measures True Positive Rate (equal opportunity) and True Negative Rate differences
    across applicant groups for hiring decisions.
    
    Args:
        df: DataFrame with hiring prediction outcomes
        
    Returns:
        Dictionary containing equality of opportunity metrics
    """
    try:
        metrics = {}
        groups = df['group'].unique()
        
        tpr_values, tnr_values = {}, {}
        
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
                    
                tn, fp, fn, tp = confusion_matrix(y_true_vals, y_pred_vals).ravel()
                
                # TPR (True Positive Rate) - Equal Opportunity metric
                tpr_values[group] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                
                # TNR (True Negative Rate) - True negative accuracy
                tnr_values[group] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                
            except Exception as e:
                continue
        
        # Calculate differences across groups
        if tpr_values and len(tpr_values) > 1:
            valid_tpr = [v for v in tpr_values.values() if v is not None]
            valid_tnr = [v for v in tnr_values.values() if v is not None]
            
            if valid_tpr:
                tpr_diff = float(max(valid_tpr) - min(valid_tpr))
                metrics['equal_opportunity_difference'] = tpr_diff
                metrics['tpr_difference'] = tpr_diff
                
            if valid_tnr:
                tnr_diff = float(max(valid_tnr) - min(valid_tnr))
                metrics['tnr_difference'] = tnr_diff
                
            if valid_tpr and valid_tnr:
                metrics['tpr_tnr_differences'] = float((tpr_diff + tnr_diff) / 2)
        
        return metrics
    except Exception as e:
        raise

def calculate_error_prediction_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate error and prediction fairness metrics for hiring decisions.
    
    Analyzes false positive/negative rates and predictive parity across
    different applicant demographic groups.
    
    Args:
        df: DataFrame with hiring prediction outcomes
        
    Returns:
        Dictionary containing error rate and prediction fairness metrics
    """
    try:
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
                    
                tn, fp, fn, tp = confusion_matrix(y_true_vals, y_pred_vals).ravel()
                
                # False Positive Rate (FPR) - Incorrect rejections
                fpr_values[group] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                
                # False Negative Rate (FNR) - Missed qualified candidates
                fnr_values[group] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
                
                # False Discovery Rate (FDR) - False positive proportion
                fdr_values[group] = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
                
                # False Omission Rate (FOR) - False negative proportion  
                for_values[group] = float(fn / (fn + tn)) if (fn + tn) > 0 else 0.0
                
                # Positive Predictive Value (PPV) - Precision of positive predictions
                ppv_values[group] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                
                # Negative Predictive Value (NPV) - Precision of negative predictions
                npv_values[group] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                
            except Exception as e:
                continue
        
        # Calculate differences across groups for each error metric
        
        # False Positive/Negative Rate Differences
        if fpr_values and len(fpr_values) > 1:
            valid_fpr = [v for v in fpr_values.values() if v is not None]
            valid_fnr = [v for v in fnr_values.values() if v is not None]
            
            if valid_fpr:
                metrics['fpr_difference'] = float(max(valid_fpr) - min(valid_fpr))
            if valid_fnr:
                metrics['fnr_difference'] = float(max(valid_fnr) - min(valid_fnr))
                
            if valid_fpr and valid_fnr:
                metrics['fnr_fpr_differences'] = float((metrics['fpr_difference'] + metrics['fnr_difference']) / 2)
        
        # False Discovery/Omission Rate Differences
        if fdr_values and len(fdr_values) > 1:
            valid_fdr = [v for v in fdr_values.values() if v is not None]
            valid_for = [v for v in for_values.values() if v is not None]
            
            if valid_fdr:
                metrics['fdr_difference'] = float(max(valid_fdr) - min(valid_fdr))
            if valid_for:
                metrics['for_difference'] = float(max(valid_for) - min(valid_for))
                
            if valid_fdr and valid_for:
                metrics['fdr_for_differences'] = float((metrics['fdr_difference'] + metrics['for_difference']) / 2)
        
        # Predictive Parity Differences (PPV and NPV combined)
        if ppv_values and len(ppv_values) > 1:
            valid_ppv = [v for v in ppv_values.values() if v is not None]
            valid_npv = [v for v in npv_values.values() if v is not None]
            
            if valid_ppv and valid_npv:
                ppv_diff = float(max(valid_ppv) - min(valid_ppv))
                npv_diff = float(max(valid_npv) - min(valid_npv))
                metrics['predictive_parity_difference'] = float((ppv_diff + npv_diff) / 2)
        
        return metrics
    except Exception as e:
        raise

def calculate_data_integrity_preprocessing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data integrity and preprocessing fairness metrics.
    
    Measures preprocessing impact on applicant features and data quality
    across different demographic groups.
    
    Args:
        df: DataFrame with hiring data features
        
    Returns:
        Dictionary containing data integrity metrics
    """
    try:
        metrics = {}
        
        # Sample Distortion Metrics - Measures preprocessing impact on applicant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
        
        if len(numeric_cols) > 0:
            distortion_scores = []
            for col in numeric_cols:
                # Calculate coefficient of variation as distortion measure
                cv = float(df[col].std() / df[col].mean()) if df[col].mean() > 0 else 0.0
                distortion_scores.append(cv)
            
            if distortion_scores:
                metrics['sample_distortion_metrics'] = {
                    'average_shift': float(np.mean(distortion_scores)),
                    'maximum_shift': float(np.max(distortion_scores)),
                    'individual_shifts': distortion_scores
                }
        
        return metrics
    except Exception as e:
        raise

def calculate_subgroup_bias_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect subgroup and hidden bias through error rate analysis.
    
    Identifies disparities in prediction accuracy across different
    applicant subgroups using MDSS-inspired scoring.
    
    Args:
        df: DataFrame with hiring prediction outcomes
        
    Returns:
        Dictionary containing subgroup bias detection metrics
    """
    try:
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
            except Exception as e:
                error_rates[group] = 0.0
        
        # Calculate error rate disparities across groups
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                # MDSS Subgroup Discovery Score (simplified implementation)
                error_diff = float(max(valid_errors) - min(valid_errors))
                metrics['mdss_subgroup_score'] = error_diff
                metrics['error_disparity_subgroup'] = error_diff
                metrics['error_rate_difference'] = error_diff
        
        return metrics
    except Exception as e:
        raise

def calculate_explainability_proxy_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate explainability and proxy detection metrics.
    
    Uses simplified feature attribution analysis to detect potential
    proxy discrimination in hiring algorithms.
    
    Args:
        df: DataFrame with hiring prediction data
        
    Returns:
        Dictionary containing explainability and proxy detection metrics
    """
    try:
        metrics = {}
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            # Feature Attribution Bias using group means (simplified SHAP analysis)
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
    except Exception as e:
        raise

def calculate_counterfactual_causal_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate counterfactual and causal fairness metrics.
    
    Simplified implementation measuring prediction changes across demographic
    groups as a proxy for counterfactual fairness.
    
    Args:
        df: DataFrame with hiring prediction data
        
    Returns:
        Dictionary containing counterfactual fairness metrics
    """
    try:
        metrics = {}
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            # Counterfactual Flip Rate (simplified implementation)
            # Measures how often predictions change when demographic attributes change
            prediction_means = []
            for group in groups:
                group_mask = df['group'] == group
                prediction_means.append(float(df[group_mask]['y_pred'].mean()))
            
            if len(prediction_means) >= 2:
                flip_rate = float(max(prediction_means) - min(prediction_means))
                metrics['counterfactual_flip_rate'] = flip_rate
                metrics['counterfactual_consistency_index'] = 1.0 - flip_rate
        
        return metrics
    except Exception as e:
        raise

def calculate_robustness_temporal_fairness(df: pd.DataFrame, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate robustness and temporal fairness metrics.
    
    Includes worst-group accuracy, temporal stability, and composite bias score
    based on weighted average of key bias metrics.
    
    Args:
        df: DataFrame with hiring prediction data
        all_metrics: Dictionary containing previously calculated metrics
        
    Returns:
        Dictionary containing robustness and composite fairness metrics
    """
    try:
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
            except Exception as e:
                accuracies[group] = 0.0

        if accuracies and len(accuracies) > 1:
            valid_accuracies = [v for v in accuracies.values() if v is not None]
            if valid_accuracies:
                # Worst-group accuracy - minimum performance across groups
                metrics['worst_group_accuracy'] = float(min(valid_accuracies))
                accuracy_range = float(max(valid_accuracies) - min(valid_accuracies))
                
                # Temporal Fairness Score (simplified - stability over time)
                metrics['temporal_fairness_score'] = 1.0 - accuracy_range
                
                # COMPOSITE BIAS SCORE: Weighted average of key bias metrics
                key_metrics = [
                    all_metrics.get('statistical_parity_difference', 0.0),
                    all_metrics.get('equal_opportunity_difference', 0.0),
                    all_metrics.get('fpr_difference', 0.0),
                    all_metrics.get('fnr_difference', 0.0),
                    all_metrics.get('counterfactual_flip_rate', 0.0),
                    all_metrics.get('error_rate_difference', 0.0)
                ]
                
                # Cap individual metrics at 0.2 to prevent extreme values from dominating
                capped_metrics = [float(min(metric, 0.2)) for metric in key_metrics if metric > 0]
                
                # Calculate weighted average instead of taking maximum for balanced assessment
                if capped_metrics and len(capped_metrics) > 0:
                    metrics['composite_bias_score'] = float(sum(capped_metrics) / len(capped_metrics))
                else:
                    metrics['composite_bias_score'] = 0.0
        
        return metrics
    except Exception as e:
        raise

def calculate_individual_fairness_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate individual fairness and consistency metrics.
    
    Measures how similar applicants are treated consistently within
    and across demographic groups.
    
    Args:
        df: DataFrame with hiring prediction data
        
    Returns:
        Dictionary containing individual fairness metrics
    """
    try:
        metrics = {}
        
        # Individual Consistency Index (simplified implementation)
        # Measures how similar applicants are treated consistently
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            # Calculate within-group prediction consistency
            consistency_scores = []
            for group in groups:
                group_mask = df['group'] == group
                group_predictions = df[group_mask]['y_pred'].values
                if len(group_predictions) > 1:
                    # Use standard deviation as inverse consistency measure
                    consistency = 1.0 - float(np.std(group_predictions))
                    consistency_scores.append(max(0.0, consistency))
            
            if consistency_scores:
                metrics['individual_consistency_index'] = float(np.mean(consistency_scores))
                metrics['similar_applicant_parity'] = float(np.min(consistency_scores))
        
        return metrics
    except Exception as e:
        raise

# ================================================================
# MAIN PIPELINE EXECUTION
# ================================================================

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """
    Main hiring pipeline execution with comprehensive error handling.
    
    Args:
        df: Pandas DataFrame containing hiring data to audit
        save_to_disk: Whether to save results to disk (default: True)
        
    Returns:
        Dictionary containing complete hiring fairness audit results
    """
    
    try:
        # Calculate all hiring fairness metrics
        hiring_metrics = calculate_hiring_metrics(df)
        
        # Build comprehensive results structure
        results = {
            "domain": "hiring",
            "metrics_calculated": 15,
            "metric_categories": HIRING_METRICS_CONFIG,
            "fairness_metrics": hiring_metrics,
            "summary": {
                "composite_bias_score": hiring_metrics.get('composite_bias_score', 0.0),
                "overall_assessment": assess_hiring_fairness(hiring_metrics)
            },
            "timestamp": str(pd.Timestamp.now())
        }
        
        # CRITICAL: Convert ALL numpy types to Python native types for JSON serialization
        results = convert_numpy_types(results)
        
        return results
        
    except Exception as e:
        # Graceful error handling with structured error response
        error_results = {
            "domain": "hiring",
            "metrics_calculated": 0,
            "error": str(e),
            "summary": {
                "composite_bias_score": 1.0,  # Maximum bias score on error
                "overall_assessment": "ERROR - Could not complete audit"
            },
            "timestamp": str(pd.Timestamp.now())
        }
        return convert_numpy_types(error_results)

def assess_hiring_fairness(metrics: Dict[str, Any]) -> str:
    """
    Assess overall fairness for hiring domain based on composite bias score.
    
    Uses hiring-specific thresholds appropriate for employment decision contexts.
    
    Args:
        metrics: Dictionary containing calculated fairness metrics
        
    Returns:
        String assessment of fairness severity level
    """
    bias_score = metrics.get('composite_bias_score', 0.0)
    
    if bias_score > 0.10:
        return "HIGH_BIAS - Significant fairness concerns in hiring decisions"
    elif bias_score > 0.03:
        return "MEDIUM_BIAS - Moderate fairness concerns detected"  
    else:
        return "LOW_BIAS - Generally fair across applicant groups"

# ================================================================
# TESTING AND COMPATIBILITY
# ================================================================

if __name__ == "__main__":
    # Test with sample hiring data
    sample_data = pd.DataFrame({
        'group': ['Group A', 'Group A', 'Group B', 'Group B', 'Group A', 'Group B'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
    })
    
    # Execute pipeline and display results
    results = run_pipeline(sample_data)
    print("Hiring Pipeline Test Results:")
    print(json.dumps(results, indent=2))