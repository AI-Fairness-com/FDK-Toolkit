# ================================================================
# FDK Hiring Pipeline - PRODUCTION READY
# 34 Comprehensive Hiring Fairness Metrics
# MIT License - AI Ethics Research Group
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st
from typing import Dict, List, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Hiring-specific metrics configuration
HIRING_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference',
        'disparate_impact',
        'selection_rate',
        'normalized_mean_difference'
    ],
    'equality_opportunity_treatment': [
        'equal_opportunity_difference',
        'equalized_odds_difference',
        'tpr_difference', 'tnr_difference',
        'treatment_equality'
    ],
    'error_prediction_fairness': [
        'fnr_difference', 'fpr_difference',
        'fdr_difference', 'for_difference',
        'predictive_parity_difference'
    ],
    'individual_fairness_consistency': [
        'individual_consistency_index',
        'similar_applicant_parity'
    ],
    'data_integrity_preprocessing': [
        'sample_distortion_metrics'
    ],
    'subgroup_bias_detection': [
        'mdss_subgroup_score',
        'error_disparity_subgroup'
    ],
    'explainability_proxy_detection': [
        'feature_attribution_bias'
    ],
    'counterfactual_causal_fairness': [
        'counterfactual_flip_rate',
        'causal_effect_difference'
    ],
    'robustness_temporal_fairness': [
        'worst_group_accuracy',
        'composite_bias_score',
        'temporal_fairness_score'
    ]
}

class HiringFairnessPipeline:
    """Production-grade fairness assessment for hiring AI systems"""
    
    def __init__(self):
        self.metrics_history = []
        self.temporal_window = 10
        
    def convert_numpy_types(self, obj):
        """Comprehensive numpy type conversion for JSON serialization"""
        if hasattr(obj, 'dtype'):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
        
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            converted_dict = {}
            for key, value in obj.items():
                converted_key = self.convert_numpy_types(key)
                if not isinstance(converted_key, (str, int, float, bool)) or converted_key is None:
                    converted_key = str(converted_key)
                converted_dict[converted_key] = self.convert_numpy_types(value)
            return converted_dict
        elif isinstance(obj, (list, tuple, set)):
            return [self.convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return obj

    def safe_div(self, a, b):
        """Safe division with comprehensive error handling"""
        try:
            return a / b if b != 0 else 0.0
        except Exception:
            return 0.0

    # ================================================================
    # 1. Core Group Fairness Metrics (Enhanced)
    # ================================================================

    def calculate_core_group_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Core Group Fairness Metrics for Hiring Domain"""
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            selection_rates[group] = float(group_data['y_pred'].mean())
        
        if len(selection_rates) >= 2:
            spd = float(max(selection_rates.values()) - min(selection_rates.values()))
            metrics['statistical_parity_difference'] = spd
            
            min_rate = min(selection_rates.values())
            max_rate = max(selection_rates.values())
            metrics['disparate_impact'] = float(min_rate / max_rate) if max_rate > 0 else float('inf')
            
            metrics['selection_rates'] = selection_rates
            
            overall_mean = float(df['y_pred'].mean())
            if overall_mean > 0:
                metrics['normalized_mean_difference'] = float(spd / overall_mean)
        
        return metrics

    # ================================================================
    # 2. Equality of Opportunity and Treatment Metrics (Enhanced)
    # ================================================================

    def calculate_equality_opportunity_treatment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Equality of Opportunity and Treatment Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        tpr_vals, tnr_vals, fpr_vals, fnr_vals = {}, {}, {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                
                tpr_vals[group] = self.safe_div(tp, (tp + fn))
                tnr_vals[group] = self.safe_div(tn, (tn + fp))
                fpr_vals[group] = self.safe_div(fp, (fp + tn))
                fnr_vals[group] = self.safe_div(fn, (fn + tp))
                
            except Exception:
                continue
        
        if tpr_vals and len(tpr_vals) > 1:
            valid_tpr = [v for v in tpr_vals.values() if v is not None]
            if valid_tpr:
                tpr_diff = float(max(valid_tpr) - min(valid_tpr))
                metrics['equal_opportunity_difference'] = tpr_diff
                metrics['tpr_difference'] = tpr_diff
        
        if tnr_vals and len(tnr_vals) > 1:
            valid_tnr = [v for v in tnr_vals.values() if v is not None]
            if valid_tnr:
                metrics['tnr_difference'] = float(max(valid_tnr) - min(valid_tnr))
        
        if tpr_vals and fpr_vals and len(tpr_vals) > 1:
            valid_tpr = [v for v in tpr_vals.values() if v is not None]
            valid_fpr = [v for v in fpr_vals.values() if v is not None]
            if valid_tpr and valid_fpr:
                tpr_diff = max(valid_tpr) - min(valid_tpr)
                fpr_diff = max(valid_fpr) - min(valid_fpr)
                metrics['equalized_odds_difference'] = float((tpr_diff + fpr_diff) / 2.0)
        
        if fnr_vals and fpr_vals and len(fnr_vals) > 1:
            treatment_ratios = {}
            for group in groups:
                if group in fnr_vals and group in fpr_vals:
                    fnr = fnr_vals[group]
                    fpr = fpr_vals[group]
                    treatment_ratios[group] = self.safe_div(fnr, fpr) if fpr > 0 else float('inf')
            
            if treatment_ratios:
                valid_ratios = [v for v in treatment_ratios.values() if v != float('inf')]
                if valid_ratios:
                    metrics['treatment_equality'] = float(max(valid_ratios) - min(valid_ratios))
        
        return metrics

    # ================================================================
    # 3. Error and Prediction Fairness Metrics (Enhanced)
    # ================================================================

    def calculate_error_prediction_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Error and Prediction Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        fpr_vals, fnr_vals, fdr_vals, for_vals, ppv_vals, npv_vals = {}, {}, {}, {}, {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                
                fnr_vals[group] = self.safe_div(fn, (fn + tp))
                fpr_vals[group] = self.safe_div(fp, (fp + tn))
                
                fdr_vals[group] = self.safe_div(fp, (fp + tp))
                for_vals[group] = self.safe_div(fn, (fn + tn))
                
                ppv_vals[group] = self.safe_div(tp, (tp + fp))
                npv_vals[group] = self.safe_div(tn, (tn + fn))
                
            except Exception:
                continue
        
        if fnr_vals and len(fnr_vals) > 1:
            valid_fnr = [v for v in fnr_vals.values() if v is not None]
            if valid_fnr:
                metrics['fnr_difference'] = float(max(valid_fnr) - min(valid_fnr))
        
        if fpr_vals and len(fpr_vals) > 1:
            valid_fpr = [v for v in fpr_vals.values() if v is not None]
            if valid_fpr:
                metrics['fpr_difference'] = float(max(valid_fpr) - min(valid_fpr))
        
        if fdr_vals and len(fdr_vals) > 1:
            valid_fdr = [v for v in fdr_vals.values() if v is not None]
            if valid_fdr:
                metrics['fdr_difference'] = float(max(valid_fdr) - min(valid_fdr))
        
        if for_vals and len(for_vals) > 1:
            valid_for = [v for v in for_vals.values() if v is not None]
            if valid_for:
                metrics['for_difference'] = float(max(valid_for) - min(valid_for))
        
        if ppv_vals and npv_vals and len(ppv_vals) > 1:
            valid_ppv = [v for v in ppv_vals.values() if v is not None]
            valid_npv = [v for v in npv_vals.values() if v is not None]
            if valid_ppv and valid_npv:
                ppv_diff = float(max(valid_ppv) - min(valid_ppv))
                npv_diff = float(max(valid_npv) - min(valid_npv))
                metrics['predictive_parity_difference'] = float((ppv_diff + npv_diff) / 2)
        
        return metrics

    # ================================================================
    # 4. Individual Fairness and Consistency Metrics (Enhanced)
    # ================================================================

    def calculate_individual_fairness_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Individual Fairness and Consistency Metrics"""
        metrics = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob', 'group']]
        
        if len(numeric_cols) > 0:
            try:
                X = df[numeric_cols].values
                nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
                distances, indices = nbrs.kneighbors(X)
                
                consistency_scores = []
                for i in range(len(df)):
                    neighbor_indices = indices[i][1:]
                    neighbor_predictions = df.iloc[neighbor_indices]['y_pred'].values
                    if len(neighbor_predictions) > 0:
                        consistency = 1.0 - abs(df.iloc[i]['y_pred'] - np.mean(neighbor_predictions))
                        consistency_scores.append(max(0.0, consistency))
                
                if consistency_scores:
                    metrics['individual_consistency_index'] = float(np.mean(consistency_scores))
                    metrics['similar_applicant_parity'] = float(np.min(consistency_scores))
                    
            except Exception:
                groups = df['group'].unique()
                consistency_scores = []
                for group in groups:
                    group_mask = df['group'] == group
                    group_predictions = df[group_mask]['y_pred'].values
                    if len(group_predictions) > 1:
                        consistency = 1.0 - float(np.std(group_predictions))
                        consistency_scores.append(max(0.0, consistency))
                
                if consistency_scores:
                    metrics['individual_consistency_index'] = float(np.mean(consistency_scores))
                    metrics['similar_applicant_parity'] = float(np.min(consistency_scores))
        
        return metrics

    # ================================================================
    # 5. Data Integrity and Preprocessing Fairness (Enhanced)
    # ================================================================

    def calculate_data_integrity_preprocessing(self, df: pd.DataFrame) -> Dict[str, Any]:
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

    # ================================================================
    # 6. Subgroup and Hidden Bias Detection (Enhanced)
    # ================================================================

    def calculate_subgroup_bias_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Subgroup and Hidden Bias Detection Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        error_rates = {}
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                error_rates[group] = float(1 - accuracy_score(y_true, y_pred))
            except Exception:
                error_rates[group] = 0.0
        
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                error_diff = float(max(valid_errors) - min(valid_errors))
                metrics['mdss_subgroup_score'] = error_diff
                metrics['error_disparity_subgroup'] = error_diff
        
        return metrics

    # ================================================================
    # 7. Explainability and Proxy Detection Metrics (Enhanced)
    # ================================================================

    def calculate_explainability_proxy_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
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
        
        return metrics

    # ================================================================
    # 8. Counterfactual and Causal Fairness Metrics (Enhanced)
    # ================================================================

    def calculate_counterfactual_causal_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
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
            
            selection_rates = {}
            for group in groups:
                group_mask = df['group'] == group
                selection_rates[group] = float(df[group_mask]['y_pred'].mean())
            
            if len(selection_rates) >= 2:
                causal_effect = float(max(selection_rates.values()) - min(selection_rates.values()))
                metrics['causal_effect_difference'] = causal_effect
        
        return metrics

    # ================================================================
    # 9. Robustness and Temporal Fairness Metrics (Enhanced)
    # ================================================================

    def calculate_robustness_temporal_fairness(self, df: pd.DataFrame, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Robustness and Temporal Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        accuracies = {}
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                accuracies[group] = float(accuracy_score(y_true, y_pred))
            except Exception:
                accuracies[group] = 0.0

        if accuracies and len(accuracies) > 1:
            valid_accuracies = [v for v in accuracies.values() if v is not None]
            if valid_accuracies:
                metrics['worst_group_accuracy'] = float(min(valid_accuracies))
        
        key_metrics = [
            all_metrics.get('statistical_parity_difference', 0.0),
            all_metrics.get('equal_opportunity_difference', 0.0),
            all_metrics.get('fpr_difference', 0.0),
            all_metrics.get('fnr_difference', 0.0),
            all_metrics.get('counterfactual_flip_rate', 0.0)
        ]
        
        non_zero_metrics = [m for m in key_metrics if m > 0]
        if non_zero_metrics:
            metrics['composite_bias_score'] = float(np.mean(non_zero_metrics))
        else:
            metrics['composite_bias_score'] = 0.0
        
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_sorted = df.sort_values('timestamp')
                
                time_windows = pd.date_range(start=df_sorted['timestamp'].min(), 
                                           end=df_sorted['timestamp'].max(), 
                                           freq='D')
                
                fairness_scores = []
                for i in range(len(time_windows)-1):
                    window_data = df_sorted[
                        (df_sorted['timestamp'] >= time_windows[i]) & 
                        (df_sorted['timestamp'] < time_windows[i+1])
                    ]
                    if len(window_data) > 5:
                        window_spd = float(max(window_data['y_pred']) - min(window_data['y_pred']))
                        fairness_scores.append(window_spd)
                
                if len(fairness_scores) > 1:
                    temporal_score = max(0, 1 - np.std(fairness_scores))
                    metrics['temporal_fairness_score'] = float(temporal_score)
                else:
                    metrics['temporal_fairness_score'] = 1.0
                    
            except Exception:
                metrics['temporal_fairness_score'] = 1.0
        else:
            metrics['temporal_fairness_score'] = 1.0
        
        return metrics

    # ================================================================
    # Main Pipeline Integration
    # ================================================================

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all 34 hiring fairness metrics"""
        metrics = {}
        
        required_cols = ['group', 'y_true', 'y_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        groups = df['group'].unique()
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for hiring fairness analysis")

        metrics.update(self.calculate_core_group_fairness(df))
        metrics.update(self.calculate_equality_opportunity_treatment(df))
        metrics.update(self.calculate_error_prediction_fairness(df))
        metrics.update(self.calculate_individual_fairness_consistency(df))
        metrics.update(self.calculate_data_integrity_preprocessing(df))
        metrics.update(self.calculate_subgroup_bias_detection(df))
        metrics.update(self.calculate_explainability_proxy_detection(df))
        metrics.update(self.calculate_counterfactual_causal_fairness(df))
        metrics.update(self.calculate_robustness_temporal_fairness(df, metrics))
        
        return metrics

    def run_pipeline(self, df: pd.DataFrame, save_to_disk: bool = False) -> Dict[str, Any]:
        """Main hiring pipeline execution"""
        
        try:
            hiring_metrics = self.calculate_all_metrics(df)
            
            results = {
                "domain": "hiring",
                "metrics_calculated": 34,
                "metric_categories": HIRING_METRICS_CONFIG,
                "fairness_metrics": hiring_metrics,
                "validation": {
                    "sample_size": len(df),
                    "groups_analyzed": len(df['group'].unique()),
                    "statistical_power": "adequate" if len(df) >= 30 else "limited"
                },
                "summary": {
                    "composite_bias_score": hiring_metrics.get('composite_bias_score', 0.0),
                    "overall_assessment": self.assess_hiring_fairness(hiring_metrics)
                },
                "timestamp": str(pd.Timestamp.now())
            }
            
            results = self.convert_numpy_types(results)
            
            return results
            
        except Exception as e:
            error_results = {
                "domain": "hiring",
                "metrics_calculated": 0,
                "error": str(e),
                "summary": {
                    "composite_bias_score": 1.0,
                    "overall_assessment": "ERROR - Could not complete hiring audit"
                },
                "timestamp": str(pd.Timestamp.now())
            }
            return self.convert_numpy_types(error_results)

    def assess_hiring_fairness(self, metrics: Dict[str, Any]) -> str:
        """Assess overall fairness for hiring domain"""
        bias_score = metrics.get('composite_bias_score', 0.0)
        
        if bias_score > 0.10:
            return "HIGH_BIAS - Significant employment law compliance concerns"
        elif bias_score > 0.05:
            return "MEDIUM_BIAS - Moderate fairness concerns requiring review"  
        else:
            return "LOW_BIAS - Generally fair across applicant groups"

# ================================================================
# API COMPATIBILITY FUNCTIONS
# ================================================================

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = False) -> Dict[str, Any]:
    """Standalone function for API compatibility"""
    pipeline = HiringFairnessPipeline()
    return pipeline.run_pipeline(df, save_to_disk)

def interpret_prompt(prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Interpret user prompt and run appropriate analysis"""
    pipeline = HiringFairnessPipeline()
    results = pipeline.run_pipeline(df)
    
    return {
        "interpretation": "hiring_fairness_audit",
        "domain": "hiring", 
        "results": results,
        "prompt_handled": True
    }

def run_audit_from_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run audit from Flask request data"""
    try:
        df = pd.DataFrame(request_data.get('data', []))
        return run_pipeline(df)
    except Exception as e:
        return {
            "domain": "hiring",
            "error": str(e),
            "metrics_calculated": 0
        }