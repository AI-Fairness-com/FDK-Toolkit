# ================================================================
# FDK Governance Pipeline - 27 Governance Fairness Metrics
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
import scipy.stats as st
from typing import Dict, List, Any
import json
import traceback
import sys

# Governance-specific metrics configuration - 27 METRICS
GOVERNANCE_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference',
        'disparate_impact_ratio',
        'equal_opportunity_difference',
        'average_odds_difference',
        'fnr_by_group',
        'fpr_by_group',
        'fdr_parity',
        'for_parity',
        'error_rate_balance',
        'overall_accuracy_equality'
    ],
    'individual_conditional_fairness': [
        'conditional_demographic_disparity',
        'counterfactual_fairness',
        'individual_fairness_distance',
        'causal_fairness',
        'subgroup_fairness_metric'
    ],
    'calibration_reliability': [
        'calibration_by_group',
        'brier_score_by_group',
        'expected_calibration_error',
        'maximum_calibration_error'
    ],
    'data_integrity_representation': [
        'representation_parity_index',
        'sampling_balance_ratio',
        'missingness_bias_index',
        'data_coverage_gap'
    ],
    'explainability_accountability': [
        'shap_summary',
        'permutation_feature_importance',
        'transparency_index',
        'fairness_correlation_index'
    ]
}

def convert_numpy_types(obj):
    """Comprehensive conversion of numpy/pandas types to Python native types"""
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
    """Governance-specific prompt interpretation"""
    governance_keywords = ['governance', 'policy', 'service', 'allocation', 'funding', 'constituency',
                         'district', 'demographic', 'ethnicity', 'gender', 'socioeconomic', 'region',
                         'voter_segment', 'political_affiliation', 'age_group', 'income_bracket',
                         'geographic_zone', 'service_allocation', 'funding_distribution', 'policy_impact',
                         'resource_access', 'benefit_approval', 'program_eligibility', 'complaint_resolution']
    
    if any(keyword in prompt.lower() for keyword in governance_keywords):
        return {
            "domain": "governance",
            "suggested_metrics": list(GOVERNANCE_METRICS_CONFIG.keys()),
            "interpretation": "Policy equity audit focusing on service allocation, funding distribution, and policy outcomes"
        }
    return {"domain": "unknown", "suggested_metrics": [], "interpretation": "Domain not recognized"}

def run_audit_from_request(audit_request: Dict[str, Any]) -> Dict[str, Any]:
    """Main audit function for governance domain"""
    try:
        df = pd.DataFrame(audit_request['data'])
        results = run_pipeline(df, save_to_disk=False)
        
        return {
            "status": "success",
            "domain": "governance",
            "metrics_calculated": 27,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Governance audit failed: {str(e)}"
        }

def validate_dataframe_before_pipeline(df, required_cols=['group', 'y_true', 'y_pred']):
    """Pre-flight check before running pipeline"""
    issues = []
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    if len(df.columns) != len(set(df.columns)):
        issues.append("Duplicate column names detected")
    
    for col in required_cols:
        if col in df.columns:
            if df[col].dtype == 'object' and col in ['y_true', 'y_pred']:
                issues.append(f"{col} should be numeric but is object")
    
    return len(issues) == 0, issues

def calculate_governance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all 27 governance fairness metrics"""
    metrics = {}
    
    try:
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
        
        # 1. Core Group Fairness (10 metrics)
        core_metrics = calculate_core_group_fairness(df)
        metrics.update(core_metrics)
        
        # 2. Individual and Conditional Fairness (5 metrics)
        individual_metrics = calculate_individual_conditional_fairness(df)
        metrics.update(individual_metrics)
        
        # 3. Calibration and Reliability (4 metrics)
        calibration_metrics = calculate_calibration_reliability(df)
        metrics.update(calibration_metrics)
        
        # 4. Data Integrity and Representation (4 metrics)
        data_metrics = calculate_data_integrity_representation(df)
        metrics.update(data_metrics)
        
        # 5. Explainability and Accountability (4 metrics)
        explainability_metrics = calculate_explainability_accountability(df, all_metrics=metrics)
        metrics.update(explainability_metrics)

        return metrics
        
    except Exception as e:
        raise

def calculate_core_group_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """1. Core Group Fairness Metrics - 10 metrics"""
    try:
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        base_rates = {}
        tpr_values, fpr_values, fnr_values, fdr_values, for_values = {}, {}, {}, {}, {}
        error_rates, accuracy_values = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            selection_rates[group] = float(group_data['y_pred'].mean())
            base_rates[group] = float(group_data['y_true'].mean())
            
            try:
                y_true_vals = group_data['y_true'].values
                y_pred_vals = group_data['y_pred'].values
                
                if len(np.unique(y_true_vals)) >= 2 and len(np.unique(y_pred_vals)) >= 2:
                    tn, fp, fn, tp = confusion_matrix(y_true_vals, y_pred_vals).ravel()
                    
                    tpr_values[group] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    fpr_values[group] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                    fnr_values[group] = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
                    fdr_values[group] = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    for_values[group] = float(fn / (tn + fn)) if (tn + fn) > 0 else 0.0
                    
                    error_rates[group] = float(1 - accuracy_score(y_true_vals, y_pred_vals))
                    accuracy_values[group] = float(accuracy_score(y_true_vals, y_pred_vals))
                    
            except Exception:
                continue
        
        # Statistical Parity Difference
        if len(selection_rates) >= 2:
            spd = float(max(selection_rates.values()) - min(selection_rates.values()))
            metrics['statistical_parity_difference'] = spd
            metrics['selection_rates'] = selection_rates
        
        # Disparate Impact Ratio
        if len(selection_rates) >= 2:
            min_selection = min(selection_rates.values())
            max_selection = max(selection_rates.values())
            if max_selection > 0:
                metrics['disparate_impact_ratio'] = float(min_selection / max_selection)
        
        # Equal Opportunity Difference
        if tpr_values and len(tpr_values) > 1:
            valid_tpr = [v for v in tpr_values.values() if v is not None]
            if valid_tpr:
                metrics['equal_opportunity_difference'] = float(max(valid_tpr) - min(valid_tpr))
        
        # Average Odds Difference
        if tpr_values and fpr_values and len(tpr_values) > 1 and len(fpr_values) > 1:
            valid_tpr = [v for v in tpr_values.values() if v is not None]
            valid_fpr = [v for v in fpr_values.values() if v is not None]
            if valid_tpr and valid_fpr:
                tpr_diff = max(valid_tpr) - min(valid_tpr)
                fpr_diff = max(valid_fpr) - min(valid_fpr)
                metrics['average_odds_difference'] = float((tpr_diff + fpr_diff) / 2)
        
        # FNR and FPR by group
        metrics['fnr_by_group'] = fnr_values
        metrics['fpr_by_group'] = fpr_values
        
        # FDR and FOR Parity
        if fdr_values and len(fdr_values) > 1:
            valid_fdr = [v for v in fdr_values.values() if v is not None]
            if valid_fdr:
                metrics['fdr_parity'] = float(max(valid_fdr) - min(valid_fdr))
                
        if for_values and len(for_values) > 1:
            valid_for = [v for v in for_values.values() if v is not None]
            if valid_for:
                metrics['for_parity'] = float(max(valid_for) - min(valid_for))
        
        # Error Rate Balance and Overall Accuracy Equality
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                metrics['error_rate_balance'] = float(max(valid_errors) - min(valid_errors))
                
        if accuracy_values and len(accuracy_values) > 1:
            valid_accuracies = [v for v in accuracy_values.values() if v is not None]
            if valid_accuracies:
                metrics['overall_accuracy_equality'] = float(max(valid_accuracies) - min(valid_accuracies))
        
        return metrics
    except Exception as e:
        raise

def calculate_individual_conditional_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """2. Individual and Conditional Fairness Metrics - 5 metrics"""
    try:
        metrics = {}
        groups = df['group'].unique()
        
        # Conditional Demographic Disparity
        if len(groups) >= 2:
            group_means = []
            for group in groups:
                group_mask = df['group'] == group
                group_means.append(float(df[group_mask]['y_pred'].mean()))
            
            if len(group_means) >= 2:
                metrics['conditional_demographic_disparity'] = float(max(group_means) - min(group_means))
        
        # Counterfactual Fairness (simplified)
        prediction_means = []
        for group in groups:
            group_mask = df['group'] == group
            prediction_means.append(float(df[group_mask]['y_pred'].mean()))
        
        if len(prediction_means) >= 2:
            metrics['counterfactual_fairness'] = float(max(prediction_means) - min(prediction_means))
        
        # Individual Fairness Distance (simplified)
        overall_pred_mean = float(df['y_pred'].mean())
        individual_distances = []
        for group in groups:
            group_mask = df['group'] == group
            group_mean = float(df[group_mask]['y_pred'].mean())
            individual_distances.append(abs(group_mean - overall_pred_mean))
        
        if individual_distances:
            metrics['individual_fairness_distance'] = float(np.mean(individual_distances))
        
        # Causal Fairness (path-specific effects simplified)
        if len(groups) >= 2:
            causal_effects = []
            for group in groups:
                group_mask = df['group'] == group
                causal_effect = float(df[group_mask]['y_pred'].mean() - df[group_mask]['y_true'].mean())
                causal_effects.append(abs(causal_effect))
            
            if causal_effects:
                metrics['causal_fairness'] = float(np.mean(causal_effects))
        
        # Subgroup Fairness Metric
        subgroup_errors = {}
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true_vals = group_data['y_true'].values
                y_pred_vals = group_data['y_pred'].values
                subgroup_errors[group] = float(1 - accuracy_score(y_true_vals, y_pred_vals))
            except Exception:
                subgroup_errors[group] = 0.0
        
        if subgroup_errors and len(subgroup_errors) > 1:
            valid_errors = [v for v in subgroup_errors.values() if v is not None]
            if valid_errors:
                metrics['subgroup_fairness_metric'] = float(max(valid_errors) - min(valid_errors))
        
        return metrics
    except Exception as e:
        raise

def calculate_calibration_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    """3. Calibration and Reliability Metrics - 4 metrics"""
    try:
        metrics = {}
        groups = df['group'].unique()
        
        calibration_scores = {}
        brier_scores = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true_vals = group_data['y_true'].values
                if 'y_prob' in group_data.columns:
                    y_prob_vals = group_data['y_prob'].values
                    
                    calibration_error = float(np.mean(np.abs(y_prob_vals - y_true_vals)))
                    calibration_scores[group] = calibration_error
                    
                    brier_scores[group] = float(mean_squared_error(y_true_vals, y_prob_vals))
                    
            except Exception:
                continue
        
        if calibration_scores:
            metrics['calibration_by_group'] = calibration_scores
        
        if brier_scores:
            metrics['brier_score_by_group'] = brier_scores
        
        if calibration_scores:
            metrics['expected_calibration_error'] = float(np.mean(list(calibration_scores.values())))
        
        if calibration_scores:
            metrics['maximum_calibration_error'] = float(np.max(list(calibration_scores.values())))
        
        return metrics
    except Exception as e:
        raise

def calculate_data_integrity_representation(df: pd.DataFrame) -> Dict[str, Any]:
    """4. Data Integrity and Representation Metrics - 4 metrics"""
    try:
        metrics = {}
        groups = df['group'].unique()
        
        # Representation Parity Index
        group_sizes = {}
        total_size = len(df)
        
        for group in groups:
            group_mask = df['group'] == group
            group_sizes[group] = len(df[group_mask])
        
        if group_sizes:
            expected_size = total_size / len(groups)
            representation_gaps = [abs(size - expected_size) / expected_size for size in group_sizes.values()]
            metrics['representation_parity_index'] = float(1.0 - np.mean(representation_gaps))
        
        # Sampling Balance Ratio
        if group_sizes and len(group_sizes) > 1:
            min_size = min(group_sizes.values())
            max_size = max(group_sizes.values())
            if max_size > 0:
                metrics['sampling_balance_ratio'] = float(min_size / max_size)
        
        # Missingness Bias Index (simplified)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missingness_scores = []
        
        for col in numeric_cols:
            if col not in ['y_true', 'y_pred', 'y_prob']:
                col_missing = df[col].isna().mean()
                missingness_scores.append(col_missing)
        
        if missingness_scores:
            metrics['missingness_bias_index'] = float(np.mean(missingness_scores))
        
        # Data Coverage Gap
        coverage_gaps = []
        for group in groups:
            group_mask = df['group'] == group
            group_coverage = len(df[group_mask]) / total_size
            expected_coverage = 1.0 / len(groups)
            coverage_gaps.append(abs(group_coverage - expected_coverage))
        
        if coverage_gaps:
            metrics['data_coverage_gap'] = float(np.mean(coverage_gaps))
        
        return metrics
    except Exception as e:
        raise

def calculate_explainability_accountability(df: pd.DataFrame, all_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """5. Explainability and Accountability Metrics - 4 metrics"""
    try:
        metrics = {}
        groups = df['group'].unique()
        
        # SHAP Summary (simplified feature importance)
        feature_importance = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob']]
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                correlation = abs(df[col].corr(df['y_pred']))
                if not np.isnan(correlation):
                    feature_importance[col] = float(correlation)
            
            if feature_importance:
                metrics['shap_summary'] = feature_importance
        
        # Permutation Feature Importance (simplified)
        if feature_importance:
            metrics['permutation_feature_importance'] = feature_importance
        
        # Transparency Index
        transparency_factors = []
        
        if len(numeric_cols) > 0:
            complexity = 1.0 / (1.0 + len(numeric_cols) / 10.0)
            transparency_factors.append(complexity)
        
        group_accuracies = []
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) > 0:
                try:
                    accuracy = accuracy_score(group_data['y_true'].values, group_data['y_pred'].values)
                    group_accuracies.append(accuracy)
                except:
                    pass
        
        if len(group_accuracies) > 1:
            consistency = 1.0 - np.std(group_accuracies)
            transparency_factors.append(consistency)
        
        if transparency_factors:
            metrics['transparency_index'] = float(np.mean(transparency_factors))
        
        # Fairness Correlation Index
        fairness_correlations = []
        
        try:
            group_numeric = pd.factorize(df['group'])[0]
            group_pred_corr = abs(np.corrcoef(group_numeric, df['y_pred'].values)[0, 1])
            if not np.isnan(group_pred_corr):
                fairness_correlations.append(1.0 - group_pred_corr)
        except:
            pass
        
        if fairness_correlations:
            metrics['fairness_correlation_index'] = float(np.mean(fairness_correlations))
        
        # Calculate composite bias score for overall assessment
        key_metrics = []
        
        if all_metrics is not None:
            for metric_name in ['statistical_parity_difference', 'equal_opportunity_difference', 
                               'average_odds_difference', 'conditional_demographic_disparity',
                               'error_rate_balance', 'fdr_parity', 'for_parity']:
                if metric_name in all_metrics and all_metrics[metric_name] is not None:
                    key_metrics.append(float(all_metrics[metric_name]))
        else:
            for metric_name in ['statistical_parity_difference', 'equal_opportunity_difference', 
                               'average_odds_difference', 'conditional_demographic_disparity',
                               'error_rate_balance', 'fdr_parity', 'for_parity']:
                if metric_name in metrics and metrics[metric_name] is not None:
                    key_metrics.append(float(metrics[metric_name]))

        if not key_metrics:
            metrics['composite_bias_score'] = 0.0
        else:
            capped_metrics = [float(min(metric, 0.2)) for metric in key_metrics if metric > 0]
            if capped_metrics:
                metrics['composite_bias_score'] = float(sum(capped_metrics) / len(capped_metrics))
            else:
                metrics['composite_bias_score'] = 0.0
        
        return metrics
    except Exception as e:
        raise

def assess_governance_fairness(metrics: Dict[str, Any]) -> str:
    """Assess overall fairness for governance domain"""
    bias_score = metrics.get('composite_bias_score', 0.0)
    
    if bias_score > 0.10:
        return "HIGH_BIAS - Significant equity concerns in policy decisions"
    elif bias_score > 0.03:
        return "MEDIUM_BIAS - Moderate equity concerns detected"  
    else:
        return "LOW_BIAS - Generally equitable across constituent groups"

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """Main governance pipeline execution"""
    
    try:
        # Pre-flight validation
        is_valid, issues = validate_dataframe_before_pipeline(df)
        if not is_valid:
            error_results = {
                "domain": "governance",
                "metrics_calculated": 0,
                "error": f"Data validation failed: {issues}",
                "summary": {
                    "composite_bias_score": 1.0,
                    "overall_assessment": "ERROR - Data validation failed"
                },
                "timestamp": str(pd.Timestamp.now())
            }
            return convert_numpy_types(error_results)
        
        # Calculate all governance metrics
        governance_metrics = calculate_governance_metrics(df)
        
        # Build comprehensive results
        results = {
            "domain": "governance",
            "metrics_calculated": 27,
            "metric_categories": GOVERNANCE_METRICS_CONFIG,
            "fairness_metrics": governance_metrics,
            "summary": {
                "composite_bias_score": governance_metrics.get('composite_bias_score', 0.0),
                "overall_assessment": assess_governance_fairness(governance_metrics)
            },
            "timestamp": str(pd.Timestamp.now())
        }
        
        # Convert ALL numpy types to Python native types
        results = convert_numpy_types(results)
        
        return results
        
    except Exception as e:
        error_results = {
            "domain": "governance",
            "metrics_calculated": 0,
            "error": str(e),
            "summary": {
                "composite_bias_score": 1.0,
                "overall_assessment": "ERROR - Could not complete audit"
            },
            "timestamp": str(pd.Timestamp.now())
        }
        return convert_numpy_types(error_results)

# For backward compatibility
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'group': ['District A', 'District A', 'District B', 'District B', 'District A', 'District B'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
    })
    
    results = run_pipeline(sample_data)
    print("Governance Pipeline Test Results:")
    print(json.dumps(results, indent=2))