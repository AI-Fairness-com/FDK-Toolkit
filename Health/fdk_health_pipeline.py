# ================================================================
# FDK Health Pipeline - Comprehensive Fairness Audit for Healthcare AI
# ================================================================
# Core fairness metrics and audit pipeline for healthcare applications
# Compliant with EU AI Act and medical device regulations
# ================================================================

import os
import json
import math
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from collections import defaultdict
from typing import Dict, List, Optional

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# ================================================================
# Configuration and Constants
# ================================================================

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_ALPHA = 0.05

# ================================================================
# Core Utility Functions
# ================================================================

def safe_div(a, b):
    """Safe division with error handling"""
    try:
        return a / b if b != 0 else None
    except Exception:
        return None

def write_json(path, obj):
    """Write object to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def _proportion_ci(p, n, alpha=DEFAULT_ALPHA):
    """Normal approximation CI for proportion"""
    if n is None or n == 0 or p is None:
        return (None, None)
    z = st.norm.ppf(1 - alpha / 2)
    se = math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))

def _metric_ratio_ci(num, den, alpha=DEFAULT_ALPHA):
    """Confidence interval for ratio metrics"""
    if den in (None, 0):
        return (None, None)
    p = num / den
    return _proportion_ci(p, den, alpha=alpha)

# ================================================================
# Data Validation Functions
# ================================================================

def validate_binary_labels(series, label_col="y_true"):
    """Validate that series contains only binary values 0 and 1"""
    series = pd.Series(series)
    unique_vals = series.dropna().unique()
    non_binary = set(unique_vals) - {0, 1}
    
    if non_binary:
        # Convert common binary representations
        if set(map(str, unique_vals)) <= {'0', '1', 'True', 'False', 'Yes', 'No', 'Y', 'N'}:
            series = series.replace({
                'True': 1, 'False': 0, 'Yes': 1, 'No': 0, 'Y': 1, 'N': 0
            }).astype(int)
            return series
        raise ValueError(f"Column '{label_col}' contains non-binary values: {non_binary}")
    return series

def validate_probability_scores(series, prob_col="y_prob"):
    """Validate that probability scores are in range [0, 1]"""
    series = pd.Series(series)
    if series.isna().any():
        raise ValueError(f"Column '{prob_col}' contains NaN values")
    if (series < 0).any() or (series > 1).any():
        raise ValueError(f"Column '{prob_col}' contains values outside [0, 1] range")
    return series

def validate_dataframe_structure(df, required_cols, label_col="y_true", pred_col="y_pred", prob_col="y_prob"):
    """
    Validate dataframe structure for fairness audit.
    Expects pre-mapped columns from user input.
    """
    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate data types
    if label_col in df.columns:
        label_series = df[label_col]
        df[label_col] = validate_binary_labels(label_series, label_col)
    
    if pred_col in df.columns:
        pred_series = df[pred_col] 
        df[pred_col] = validate_binary_labels(pred_series, pred_col)
        
    if prob_col in df.columns and not df[prob_col].isna().all():
        prob_series = df[prob_col]
        df[prob_col] = validate_probability_scores(prob_series, prob_col)

    # Store column mapping info
    df.attrs['column_mapping'] = {
        'group': required_cols[0],
        'y_true': label_col,
        'y_pred': pred_col,
        'y_prob': prob_col if prob_col in df.columns else None
    }
    df.attrs['original_columns'] = list(df.columns)

    return df

# ================================================================
# Core Metric Calculation Functions
# ================================================================

def _confusion_counts(y_true, y_pred, positive=1):
    """Calculate confusion matrix counts"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

def _compute_rates_from_counts(counts):
    """
    Compute comprehensive performance metrics from confusion matrix counts.
    """
    tp = counts.get('tp', 0)
    tn = counts.get('tn', 0)
    fp = counts.get('fp', 0)
    fn = counts.get('fn', 0)

    total = tp + tn + fp + fn
    denom_pos = tp + fp
    denom_neg = tn + fn
    actual_pos = tp + fn
    actual_neg = tn + fp

    # Core rates with safe division
    tpr = safe_div(tp, actual_pos)
    tnr = safe_div(tn, actual_neg)
    fpr = safe_div(fp, actual_neg)
    fnr = safe_div(fn, actual_pos)
    ppv = safe_div(tp, denom_pos)
    npv = safe_div(tn, denom_neg)
    fdr = safe_div(fp, denom_pos)
    forate = safe_div(fn, denom_neg)

    # Accuracy metrics
    acc = safe_div(tp + tn, total)
    bal_acc = safe_div(tpr + tnr, 2) if None not in [tpr, tnr] else None

    def with_ci(metric_val, denominator, alpha=DEFAULT_ALPHA):
        if metric_val is None or denominator is None or denominator == 0:
            return {"value": None, "ci": (None, None)}
        ci = _metric_ratio_ci(int(metric_val * denominator), denominator, alpha=alpha)
        return {"value": float(metric_val), "ci": ci}

    return {
        "TPR": with_ci(tpr, actual_pos),
        "TNR": with_ci(tnr, actual_neg),
        "FPR": with_ci(fpr, actual_neg),
        "FNR": with_ci(fnr, actual_pos),
        "PPV": with_ci(ppv, denom_pos),
        "NPV": with_ci(npv, denom_neg),
        "FDR": with_ci(fdr, denom_pos),
        "FOR": with_ci(forate, denom_neg),
        "Accuracy": with_ci(acc, total),
        "BalancedAccuracy": {"value": bal_acc, "ci": (None, None)} if bal_acc is not None else {"value": None, "ci": (None, None)}
    }

def _expected_calibration_error(y_true, y_prob, bins=10):
    """
    Safe Expected Calibration Error calculation with error handling.
    """
    try:
        if y_true is None or y_prob is None:
            return None
        if len(y_true) == 0 or len(y_prob) == 0:
            return None
        if len(y_true) != len(y_prob):
            return None

        prob = np.array(y_prob)
        truth = np.array(y_true)

        # Validate probability range
        if np.any(prob < 0) or np.any(prob > 1):
            LOGGER.warning("Probabilities outside [0,1] range in ECE calculation")
            return None

        if len(prob) < bins:
            LOGGER.warning(f"Insufficient data for ECE: {len(prob)} samples for {bins} bins")
            return None

        bins_edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        total_weight = 0

        for i in range(bins):
            if i < bins - 1:
                mask = (prob >= bins_edges[i]) & (prob < bins_edges[i+1])
            else:
                mask = (prob >= bins_edges[i]) & (prob <= bins_edges[i+1])

            if not mask.any():
                continue

            bin_mean_pred = prob[mask].mean()
            bin_true = truth[mask].mean()
            bin_weight = mask.sum() / len(prob)

            ece += abs(bin_mean_pred - bin_true) * bin_weight
            total_weight += bin_weight

        return float(ece / total_weight) if total_weight > 0 else None

    except Exception as e:
        LOGGER.warning(f"ECE calculation failed: {e}")
        return None

def _auc_safe(y_true, y_score):
    """Safe AUC calculation with comprehensive error handling"""
    try:
        if y_true is None or y_score is None:
            return None
        if len(y_true) == 0 or len(y_score) == 0:
            return None
        if len(y_true) != len(y_score):
            return None

        # Check for constant predictions or single class
        if len(np.unique(y_score)) == 1 or len(np.unique(y_true)) == 1:
            return None

        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        LOGGER.warning(f"AUC calculation failed: {e}")
        return None

# ================================================================
# Fairness Metric Functions
# ================================================================

def _range_and_ratio(dct):
    """Compute range and ratio safely from dictionary of values"""
    try:
        vals = [float(v) for v in dct.values() if v is not None]
        if len(vals) < 2:
            return {"range": None, "ratio": None}
        mx, mn = max(vals), min(vals)
        rng = mx - mn
        ratio = (mx / mn) if mn != 0 else None
        return {"range": rng, "ratio": ratio}
    except Exception:
        return {"range": None, "ratio": None}

def get_reference_group(group_metrics, strategy="largest", custom_reference=None):
    """
    Unified reference group selector for fairness metrics.
    """
    valid_metrics = {g: v for g, v in group_metrics.items() if v is not None}

    if not valid_metrics:
        return None

    if strategy == "specified" and custom_reference:
        if custom_reference in group_metrics:
            return custom_reference
        else:
            raise ValueError(f"Specified reference group '{custom_reference}' not found")

    if strategy == "largest":
        return max(valid_metrics.items(), key=lambda x: x[1])[0]
    elif strategy == "smallest":
        return min(valid_metrics.items(), key=lambda x: x[1])[0]
    else:
        raise ValueError(f"Unknown reference strategy: {strategy}")

def get_reference_group_with_n(group_n_counts, strategy="majority"):
    """Reference group selector based on sample sizes"""
    valid_n = {g: n for g, n in group_n_counts.items() if n is not None and n > 0}

    if not valid_n:
        return None

    if strategy == "majority":
        return max(valid_n.items(), key=lambda x: x[1])[0]
    elif strategy == "minority":
        return min(valid_n.items(), key=lambda x: x[1])[0]
    else:
        raise ValueError(f"Unknown n-based strategy: {strategy}")

def selection_rate_by_group(df, group_col='group', pred_col='y_pred', positive=1):
    """Calculate selection rate by group"""
    sr = {}
    groups_series = df[group_col].dropna()
    if isinstance(groups_series, pd.DataFrame):
        groups_series = groups_series.iloc[:, 0]
    groups = groups_series.unique()
    
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) == 0:
            sr[g] = None
        else:
            sr[g] = float((sub[pred_col] == positive).mean())
    return sr

def statistical_parity_difference(sr_by_group, reference_strategy="largest", custom_reference=None):
    """SPD = p(pred=1 | g) - p(pred=1 | reference)"""
    reference = get_reference_group(sr_by_group, strategy=reference_strategy, custom_reference=custom_reference)
    if reference is None:
        return {}, None
    ref_val = sr_by_group.get(reference)
    return {g: (None if v is None or ref_val is None else float(v - ref_val)) for g,v in sr_by_group.items()}, reference

def group_base_rates(df, group_col='group', label_col='y_true'):
    """Calculate base rates (prevalence) by group"""
    br = {}
    groups_series = df[group_col].dropna()
    if isinstance(groups_series, pd.DataFrame):
        groups_series = groups_series.iloc[:, 0]
    groups = groups_series.unique()
    
    for g in groups:
        sub = df[df[group_col]==g]
        br[g] = float(sub[label_col].mean()) if len(sub) else None
    return br

def equal_opportunity_diff(group_tpr, reference_strategy="largest", custom_reference=None):
    """TPR difference relative to reference group"""
    reference = get_reference_group(group_tpr, strategy=reference_strategy, custom_reference=custom_reference)
    if reference is None:
        return {}, None
    ref_val = group_tpr.get(reference)
    return {g: (None if v is None or ref_val is None else float(v - ref_val)) for g,v in group_tpr.items()}, reference

def error_disparity(group_error_rates):
    """Calculate error rate disparity across groups"""
    vals = [v for v in group_error_rates.values() if v is not None]
    if len(vals) < 2:
        return {"range": None, "ratio": None}
    return {"range": max(vals)-min(vals), "ratio": (max(vals)/min(vals)) if min(vals)!=0 else None}

def worst_group_metric(group_stats, metric_key='Accuracy'):
    """Find worst performing group for a given metric"""
    worst = None
    worst_val = None
    for g, gs in group_stats.items():
        v = gs.get(metric_key)
        if v is None: 
            continue
        if worst_val is None or v < worst_val:
            worst_val = v
            worst = g
    return worst, worst_val

# ================================================================
# Core Fairness Audit Function
# ================================================================

def run_fairness_audit(df: pd.DataFrame, group_col: str = "group", label_col: str = "y_true",
                       pred_col: str = "y_pred", prob_col: str = "y_prob", save_to_disk: bool = False):
    """
    Run a fairness audit on df grouped by `group_col`.
    Returns an `audit` dictionary with performance and calibration info.
    """
    # Enhanced data validation
    required_cols = [group_col, label_col, pred_col]
    if prob_col in df.columns:
        required_cols.append(prob_col)

    df_mapped = validate_dataframe_structure(df, required_cols, label_col, pred_col, prob_col)
    column_mapping = df_mapped.attrs.get('column_mapping', {})
    original_columns = df_mapped.attrs.get('original_columns', [])

    # Extract groups from validated dataframe
    groups_series = df_mapped[group_col].dropna()
    if isinstance(groups_series, pd.DataFrame):
        groups_series = groups_series.iloc[:, 0]
    groups = groups_series.unique().tolist()
    
    # Calculate group statistics
    group_stats = {}
    for g in groups:
        sub = df_mapped[df_mapped[group_col] == g]
        n = len(sub)
        if n == 0:
            continue
            
        y_true = sub[label_col].astype(int)
        y_pred = sub[pred_col].astype(int)

        counts = _confusion_counts(y_true, y_pred)
        performance_metrics = _compute_rates_from_counts(counts)

        ece = None
        if prob_col in sub.columns:
            ece = _expected_calibration_error(sub[label_col], sub[prob_col])

        group_stats[g] = {
            "n": n,
            "prevalence": float(y_true.mean()) if len(y_true) > 0 else None,
            "mean_pred": float(y_pred.mean()) if len(y_pred) > 0 else None,
            "ece": float(ece) if ece is not None else None,
            "counts": counts,
            "performance": performance_metrics
        }

    # Build audit results
    audit = {
        "groups": list(group_stats.keys()),
        "group_stats": group_stats,
        "performance": {},
        "column_mapping": column_mapping,
        "original_columns": original_columns
    }

    # Overall metrics
    try:
        auc = _auc_safe(df_mapped[label_col].astype(int), df_mapped[prob_col]) if prob_col in df_mapped.columns else None
    except Exception:
        auc = None
    audit["performance"]["auc"] = auc

    # Calibration analysis
    calibration_by_group = {}
    calib_ci_by_group = {}
    ece_by_group = {}
    
    for g, gs in group_stats.items():
        mean_pred = gs.get("mean_pred")
        prev = gs.get("prevalence")
        ece = gs.get("ece")
        
        if mean_pred is None or prev is None:
            calibration_by_group[g] = None
            calib_ci_by_group[g] = (None, None)
        else:
            calibration_by_group[g] = float(abs(mean_pred - prev))
            n = gs.get("n", 0)
            if prev is None:
                calib_ci_by_group[g] = (None, None)
            else:
                low, high = _proportion_ci(prev, n)
                calib_ci_by_group[g] = (low, high)
        ece_by_group[g] = ece
        
    audit["calibration_gap"] = {
        "by_group": calibration_by_group,
        "ece_by_group": ece_by_group,
        "calib_ci_by_group": calib_ci_by_group,
    }

    # Performance gaps
    perf_gaps = {}
    metric_keys = ["TPR", "TNR", "FPR", "FNR", "PPV", "NPV", "FDR", "FOR", "Accuracy", "BalancedAccuracy"]

    for metric in metric_keys:
        metric_values = {}
        for g, gs in group_stats.items():
            metric_val = gs.get("performance", {}).get(metric, {}).get("value")
            metric_values[g] = metric_val
        perf_gaps[metric] = _range_and_ratio(metric_values)

    audit["performance_gaps"] = perf_gaps
    audit = convert_numpy_types(audit)

    if save_to_disk:
        write_json("audit_result.json", audit)
        
    return audit

# ================================================================
# Composite Bias Score Calculation
# ================================================================

def compute_composite_bias_score(audit):
    """
    Enhanced composite bias score incorporating multiple fairness dimensions.
    """
    try:
        weights = {
            'performance_gaps': 0.4,
            'calibration_gaps': 0.3,
            'subgroup_analysis': 0.2,
            'error_disparity': 0.1
        }

        component_scores = {}

        # Performance gaps component
        perf_gaps = audit.get("performance_gaps", {})
        key_perf_metrics = ['TPR', 'FPR', 'PPV', 'Accuracy']
        perf_gap_values = []

        for metric in key_perf_metrics:
            gap_range = perf_gaps.get(metric, {}).get('range')
            if gap_range is not None:
                perf_gap_values.append(gap_range)

        component_scores['performance_gaps'] = np.mean(perf_gap_values) if perf_gap_values else 0.0

        # Calibration gaps component
        calib_gap = audit.get("calibration_gap", {}).get("by_group", {})
        calib_vals = [v for v in calib_gap.values() if v is not None]
        if len(calib_vals) > 1:
            component_scores['calibration_gaps'] = max(calib_vals) - min(calib_vals)
        else:
            component_scores['calibration_gaps'] = 0.0

        # Subgroup analysis component
        subgroup_analysis = audit.get("subgroup_analysis", {}).get("worst_group_analysis", {})
        if subgroup_analysis and subgroup_analysis.get('overall_severity_score') is not None:
            component_scores['subgroup_analysis'] = subgroup_analysis['overall_severity_score']
        else:
            component_scores['subgroup_analysis'] = 0.0

        # Error disparity component
        error_analysis = audit.get("error_analysis", {})
        if error_analysis and error_analysis.get('disparity', {}).get('range') is not None:
            component_scores['error_disparity'] = error_analysis['disparity']['range']
        else:
            component_scores['error_disparity'] = 0.0

        # Calculate weighted composite score
        composite_score = 0.0
        total_weight = 0.0

        for component, weight in weights.items():
            if component in component_scores:
                composite_score += component_scores[component] * weight
                total_weight += weight

        final_score = composite_score / total_weight if total_weight > 0 else 0.0

        return {
            "composite_bias_score": float(final_score),
            "component_scores": component_scores,
            "interpretation": f"Composite bias score: {final_score:.3f}",
            "severity_level": "HIGH" if final_score > 0.3 else "MEDIUM" if final_score > 0.1 else "LOW"
        }

    except Exception as e:
        LOGGER.warning(f"Composite bias score calculation failed: {e}")
        return {"composite_bias_score": None, "error": str(e)}

# ================================================================
# Enhanced Analysis Functions
# ================================================================

def compute_performance_matrix(df, group_col='group', label_col='y_true', pred_col='y_pred',
                             reference_strategy="largest", custom_reference=None):
    """
    Compute comprehensive performance matrix with differences and ratios for all groups.
    """
    groups_series = df[group_col].dropna()
    if isinstance(groups_series, pd.DataFrame):
        groups_series = groups_series.iloc[:, 0]
    groups = groups_series.unique().tolist()
    
    if not groups:
        return {}

    performance_matrix = {}
    
    for group in groups:
        sub = df[df[group_col] == group]
        if len(sub) == 0:
            continue
            
        y_true = sub[label_col].astype(int)
        y_pred = sub[pred_col].astype(int)
        
        counts = _confusion_counts(y_true, y_pred)
        performance_metrics = _compute_rates_from_counts(counts)
        
        performance_matrix[group] = {}
        metric_keys = ["TPR", "TNR", "FPR", "FNR", "PPV", "NPV", "FDR", "FOR", "Accuracy", "BalancedAccuracy"]
        
        for metric in metric_keys:
            performance_matrix[group][metric] = performance_metrics.get(metric, {}).get("value")

    # Get reference group
    group_sizes = {g: len(df[df[group_col] == g]) for g in groups}
    reference_group = get_reference_group_with_n(group_sizes, strategy="majority")

    if reference_group is None or reference_group not in performance_matrix:
        return {"performance_matrix": performance_matrix, "differences": {}, "ratios": {}}

    # Compute differences and ratios
    differences = {}
    ratios = {}

    for metric in metric_keys:
        differences[metric] = {}
        ratios[metric] = {}

        ref_value = performance_matrix[reference_group].get(metric)
        if ref_value is None:
            continue

        for group in performance_matrix:
            group_value = performance_matrix[group].get(metric)
            if group_value is not None and ref_value is not None:
                differences[metric][group] = float(group_value - ref_value)
                ratios[metric][group] = float(group_value / ref_value) if ref_value != 0 else None
            else:
                differences[metric][group] = None
                ratios[metric][group] = None

    return {
        "performance_matrix": performance_matrix,
        "differences": differences,
        "ratios": ratios,
        "reference_group": reference_group,
        "metric_summary": _range_and_ratio({g: pm.get("Accuracy") for g, pm in performance_matrix.items()})
    }

def calculate_error_rate_disparity(df, group_col='group', label_col='y_true', pred_col='y_pred'):
    """
    Calculate error rate disparity across groups.
    """
    groups_series = df[group_col].dropna()
    if isinstance(groups_series, pd.DataFrame):
        groups_series = groups_series.iloc[:, 0]
    groups = groups_series.unique().tolist()
    
    error_rates = {}
    for group in groups:
        sub = df[df[group_col] == group]
        if len(sub) == 0:
            error_rates[group] = None
            continue
            
        accuracy = (sub[label_col] == sub[pred_col]).mean()
        error_rates[group] = 1.0 - accuracy if accuracy is not None else None

    disparity = error_disparity(error_rates)
    
    # Find worst group for accuracy
    worst_group = None
    worst_accuracy = float('inf')
    for group, error_rate in error_rates.items():
        if error_rate is not None and error_rate < worst_accuracy:
            worst_accuracy = error_rate
            worst_group = group

    return {
        "error_rates_by_group": error_rates,
        "disparity": disparity,
        "worst_group": [worst_group, 1.0 - worst_accuracy] if worst_group else [None, None]
    }

def root_cause_error_slice(df, group_col='group', label_col='y_true', pred_col='y_pred',
                          min_support=0.05, max_depth=3):
    """
    Basic subgroup discovery to identify critical underperforming subgroups.
    """
    try:
        total_samples = len(df)
        min_samples = max(1, int(min_support * total_samples))

        # Calculate base error rate
        base_error = 1 - (df[label_col] == df[pred_col]).mean()

        problematic_subgroups = []

        # Analyze single-feature subgroups
        for feature in [group_col]:
            if feature not in df.columns:
                continue

            feature_series = df[feature].dropna()
            if isinstance(feature_series, pd.DataFrame):
                feature_series = feature_series.iloc[:, 0]
                
            for value in feature_series.unique():
                subgroup_mask = df[feature] == value
                subgroup_size = subgroup_mask.sum()

                if subgroup_size < min_samples:
                    continue

                subgroup_error = 1 - (df[subgroup_mask][label_col] == df[subgroup_mask][pred_col]).mean()
                error_ratio = subgroup_error / base_error if base_error > 0 else 1.0

                # Consider subgroups with high error rates
                if subgroup_error > base_error and error_ratio > 1.2:
                    problematic_subgroups.append({
                        'subgroup_description': f"{feature}={value}",
                        'subgroup_size': subgroup_size,
                        'subgroup_error_rate': float(subgroup_error),
                        'base_error_rate': float(base_error),
                        'error_ratio': float(error_ratio),
                        'support': float(subgroup_size / total_samples),
                        'mdss_score': float((subgroup_error - base_error) * np.log(subgroup_size))
                    })

        # Sort by MDSS score
        problematic_subgroups.sort(key=lambda x: x['mdss_score'], reverse=True)

        return {
            'base_error_rate': float(base_error),
            'total_samples': total_samples,
            'top_problematic_subgroups': problematic_subgroups[:10],
            'subgroup_count': len(problematic_subgroups)
        }

    except Exception as e:
        LOGGER.warning(f"Subgroup discovery failed: {e}")
        return None

def enhanced_worst_group_analysis(df, group_col='group', label_col='y_true', pred_col='y_pred'):
    """
    Enhanced worst-group analysis with multiple metrics and severity scoring.
    """
    audit = run_fairness_audit(df, group_col, label_col, pred_col)
    group_stats = audit.get("group_stats", {})

    if not group_stats:
        return None

    # Find worst groups across multiple metrics
    metrics_to_check = ['Accuracy', 'TPR', 'PPV', 'BalancedAccuracy']
    worst_groups = {}

    for metric in metrics_to_check:
        worst_group, worst_value = worst_group_metric(group_stats, metric_key=metric)
        if worst_group:
            worst_groups[metric] = {
                'group': worst_group,
                'value': worst_value,
                'severity': 1.0 - worst_value if worst_value is not None else None
            }

    # Calculate overall worst group
    group_severities = defaultdict(list)
    for metric, info in worst_groups.items():
        if info['severity'] is not None:
            group_severities[info['group']].append(info['severity'])

    overall_worst = None
    overall_severity = -1
    for group, severities in group_severities.items():
        avg_severity = np.mean(severities)
        if avg_severity > overall_severity:
            overall_severity = avg_severity
            overall_worst = group

    return {
        'worst_groups_by_metric': worst_groups,
        'overall_worst_group': overall_worst,
        'overall_severity_score': float(overall_severity) if overall_severity != -1 else None,
        'recommendation': f"Focus improvement efforts on group '{overall_worst}'" if overall_worst else "No clear worst group identified"
    }

# ================================================================
# Main Pipeline Function
# ================================================================

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = False):
    """
    Enhanced pipeline that runs comprehensive fairness audit including all metrics.
    """
    # Run core fairness audit
    audit = run_fairness_audit(df, save_to_disk=save_to_disk)

    # Recreate mapped dataframe using column mapping
    column_mapping = audit.get("column_mapping", {})
    df_mapped = df.copy()
    rename_dict = {}
    for std_name, actual_name in column_mapping.items():
        if actual_name in df.columns and actual_name != std_name:
            rename_dict[actual_name] = std_name

    if rename_dict:
        df_mapped = df_mapped.rename(columns=rename_dict)
    else:
        df_mapped = df

    # Integrate core fairness metrics
    sr = selection_rate_by_group(df_mapped, group_col='group', pred_col='y_pred', positive=1)
    spd_by_group, spd_ref = statistical_parity_difference(sr)

    audit.setdefault("fairness", {})["selection_rate"] = sr
    audit["fairness"]["SPD"] = {"by_group": spd_by_group, "reference": spd_ref}

    # Base rates and equal opportunity
    br = group_base_rates(df_mapped, 'group', 'y_true')
    audit["fairness"]["base_rate"] = br

    group_tpr = {g: gs.get("performance", {}).get("TPR", {}).get("value")
                 for g, gs in audit.get("group_stats", {}).items()}
    eo_diff, eo_ref = equal_opportunity_diff(group_tpr)
    audit["fairness"]["equal_opportunity_diff"] = {"by_group": eo_diff, "reference": eo_ref}

    # Add comprehensive performance matrix
    try:
        performance_matrix = compute_performance_matrix(df_mapped)
        audit["comprehensive_performance"] = performance_matrix
    except Exception as e:
        LOGGER.warning(f"Performance matrix calculation failed: {e}")
        audit["comprehensive_performance"] = None

    # Add error rate disparity
    try:
        error_disparity_result = calculate_error_rate_disparity(df_mapped)
        audit["error_analysis"] = error_disparity_result
    except Exception as e:
        LOGGER.warning(f"Error disparity calculation failed: {e}")
        audit["error_analysis"] = None

    # Add subgroup discovery
    try:
        subgroup_analysis = root_cause_error_slice(df_mapped)
        worst_group_analysis = enhanced_worst_group_analysis(df_mapped)
        audit["subgroup_analysis"] = {
            "problematic_subgroups": subgroup_analysis,
            "worst_group_analysis": worst_group_analysis
        }
    except Exception as e:
        LOGGER.warning(f"Subgroup analysis failed: {e}")
        audit["subgroup_analysis"] = None

    # Update composite bias score
    audit["summary"] = audit.get("summary", {})
    composite_result = compute_composite_bias_score(audit)
    if composite_result:
        audit["summary"].update(composite_result)

    # Enhanced human readable summary
    human_lines = []
    human_lines.append(f"=== COMPREHENSIVE FAIRNESS AUDIT REPORT ===")
    human_lines.append(f"Groups analyzed: {list(audit.get('groups', []))}")
    human_lines.append(f"Total samples: {len(df)}")

    # Performance summary
    if audit.get('performance', {}).get('auc'):
        human_lines.append(f"Overall AUC: {audit['performance']['auc']:.3f}")

    # Fairness gaps summary
    perf_gaps = audit.get('performance_gaps', {})
    if perf_gaps.get('TPR', {}).get('range'):
        human_lines.append(f"TPR disparity range: {perf_gaps['TPR']['range']:.3f}")
    if perf_gaps.get('Accuracy', {}).get('range'):
        human_lines.append(f"Accuracy disparity range: {perf_gaps['Accuracy']['range']:.3f}")

    # Worst group info
    if audit.get('subgroup_analysis', {}).get('worst_group_analysis', {}).get('overall_worst_group'):
        worst_info = audit['subgroup_analysis']['worst_group_analysis']
        human_lines.append(f"Overall worst-performing group: {worst_info['overall_worst_group']} "
                          f"(severity: {worst_info['overall_severity_score']:.3f})")

    # Composite score
    if audit['summary'].get('composite_bias_score'):
        score = audit['summary']['composite_bias_score']
        human_lines.append(f"Composite Bias Score: {score:.3f} "
                          f"({'Low' if score < 0.1 else 'Medium' if score < 0.3 else 'High'} bias)")

    # Recommendations
    human_lines.append("\n=== KEY RECOMMENDATIONS ===")
    if audit['summary'].get('composite_bias_score', 0) > 0.3:
        human_lines.append("🚨 HIGH PRIORITY: Significant fairness issues detected")
        human_lines.append("   - Investigate worst-performing groups")
        human_lines.append("   - Consider bias mitigation techniques")
    elif audit['summary'].get('composite_bias_score', 0) > 0.1:
        human_lines.append("⚠️  MEDIUM PRIORITY: Moderate fairness concerns")
        human_lines.append("   - Monitor performance disparities")
        human_lines.append("   - Consider targeted improvements")
    else:
        human_lines.append("✅ LOW PRIORITY: Minimal fairness concerns detected")
        human_lines.append("   - Continue monitoring")
        human_lines.append("   - Maintain current practices")

    audit["human_summary"] = "\n".join(human_lines)
    audit = convert_numpy_types(audit)

    if save_to_disk:
        write_json("comprehensive_audit_result.json", audit)

    return audit

# ================================================================
# API Compatibility Functions
# ================================================================

def interpret_prompt(prompt: str) -> dict:
    """Simplified natural-language parser for API compatibility"""
    return {
        "prompt": prompt,
        "dataset_path": None,
        "metrics": "all"
    }

def run_audit_from_request(req: dict):
    """Compatibility wrapper for Flask app"""
    import pandas as pd
    df = pd.read_csv(req["dataset_path"])
    return run_fairness_audit(df, save_to_disk=False)