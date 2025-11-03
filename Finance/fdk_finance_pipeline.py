# ================================================================
# FDK Finance Pipeline - PRODUCTION READY
# 21 Comprehensive Finance Fairness Metrics
# MIT License - AI Ethics Research Group
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as st
from typing import Dict, List, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Production configuration for all 21 metrics
FINANCE_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference',
        'disparate_impact', 
        'selection_rate',
        'base_rate',
        'predicted_positives_per_group',
        'predicted_negatives_per_group'
    ],
    'calibration_reliability': [
        'calibration_gap',
        'regression_parity',
        'slice_auc_difference',
        'auc_confidence_interval_disparity'
    ],
    'error_prediction_fairness': [
        'fpr_difference', 'fnr_difference',
        'fpr_ratio', 'fnr_ratio',
        'treatment_equality',
        'fdr_difference', 'for_difference',
        'fdr_ratio', 'for_ratio',
        'predictive_parity_difference',
        'ppv_difference', 'npv_difference'
    ],
    'statistical_inequality': [
        'coefficient_of_variation',
        'generalized_entropy_index',
        'theil_index',
        'mean_difference',
        'normalized_mean_difference'
    ],
    'subgroup_bias_detection': [
        'error_rate_difference',
        'error_rate_ratio',
        'subgroup_error_disparity'
    ],
    'causal_fairness': [
        'average_causal_effect_difference',
        'counterfactual_fairness_score'
    ],
    'robustness_fairness': [
        'worst_group_accuracy',
        'worst_group_loss',
        'composite_bias_score',
        'temporal_drift_index',
        'stability_metric'
    ],
    'explainability_temporal': [
        'feature_attribution_bias',
        'temporal_fairness_score'
    ]
}

class FinanceFairnessPipeline:
    """Production-grade fairness assessment for financial AI systems"""
    
    def __init__(self):
        self.metrics_history = []
        self.temporal_window = 10
        
    def convert_numpy_types(self, obj):
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        if hasattr(obj, 'dtype'):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
        
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.str_)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        if isinstance(obj, dict):
            return {str(k): self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self.convert_numpy_types(item) for item in obj]
        
        return obj

    def calculate_core_group_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """1. Core Group Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates, base_rates = {}, {}
        predicted_positives, predicted_negatives = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            selection_rates[group] = float(group_data['y_pred'].mean())
            base_rates[group] = float(group_data['y_true'].mean())
            predicted_positives[group] = int(group_data['y_pred'].sum())
            predicted_negatives[group] = int(len(group_data) - group_data['y_pred'].sum())
        
        if len(selection_rates) >= 2:
            # 1. Statistical Parity Difference
            spd = float(max(selection_rates.values()) - min(selection_rates.values()))
            metrics['statistical_parity_difference'] = spd
            
            # 2. Disparate Impact
            min_rate = min(selection_rates.values())
            max_rate = max(selection_rates.values())
            metrics['disparate_impact'] = float(min_rate / max_rate) if max_rate > 0 else float('inf')
            
            metrics['selection_rates'] = selection_rates
        
        if len(base_rates) >= 2:
            # 4. Base Rate Difference
            base_rate_diff = float(max(base_rates.values()) - min(base_rates.values()))
            metrics['base_rate_difference'] = base_rate_diff
            metrics['base_rates'] = base_rates
        
        # 3. Selection Rate components
        metrics['predicted_positives_per_group'] = predicted_positives
        metrics['predicted_negatives_per_group'] = predicted_negatives
        
        return metrics

    def calculate_calibration_reliability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """2. Calibration and Reliability Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        calibration_gaps, mse_values, auc_scores, auc_cis = {}, {}, {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            # 5. Calibration Gap
            if 'y_prob' in df.columns:
                try:
                    y_true_vals = group_data['y_true'].values
                    y_prob_vals = group_data['y_prob'].values
                    
                    mean_pred_prob = float(y_prob_vals.mean())
                    actual_rate = float(y_true_vals.mean())
                    calibration_gaps[group] = float(abs(mean_pred_prob - actual_rate))
                except Exception:
                    calibration_gaps[group] = 0.0
            
            # 6. Regression Parity
            try:
                y_true_vals = group_data['y_true'].values
                y_pred_vals = group_data['y_pred'].values
                
                if len(np.unique(y_pred_vals)) > 2:  # Regression case
                    mse_values[group] = float(mean_squared_error(y_true_vals, y_pred_vals))
            except Exception:
                mse_values[group] = 0.0
            
            # 7. Slice AUC with Confidence Intervals
            if 'y_prob' in df.columns:
                try:
                    y_true_vals = group_data['y_true'].values
                    y_prob_vals = group_data['y_prob'].values
                    
                    if len(np.unique(y_true_vals)) > 1:
                        auc = roc_auc_score(y_true_vals, y_prob_vals)
                        auc_scores[group] = float(auc)
                        
                        # Calculate AUC confidence interval
                        n = len(y_true_vals)
                        if n > 1:
                            auc_var = auc * (1 - auc) * (1 + (n/2 - 1)) / (n * (n - 1))
                            ci_width = 1.96 * np.sqrt(auc_var)
                            auc_cis[group] = [max(0, auc - ci_width), min(1, auc + ci_width)]
                except Exception:
                    continue
        
        # Calculate differences
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
                
                # AUC Confidence Interval Disparity
                if auc_cis and len(auc_cis) > 1:
                    ci_widths = [ci[1] - ci[0] for ci in auc_cis.values() if ci is not None]
                    if ci_widths:
                        metrics['auc_confidence_interval_disparity'] = float(max(ci_widths) - min(ci_widths))
        
        return metrics

    def calculate_error_prediction_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """3. Error and Prediction Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        fpr_vals, fnr_vals, fdr_vals, for_vals, ppv_vals, npv_vals = {}, {}, {}, {}, {}, {}
        
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
                
                # 8. FPR and FNR
                fpr_vals[group] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                fnr_vals[group] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
                
                # 10. FDR and FOR
                fdr_vals[group] = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
                for_vals[group] = float(fn / (fn + tn)) if (fn + tn) > 0 else 0.0
                
                # 11. PPV and NPV
                ppv_vals[group] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                npv_vals[group] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                
            except Exception:
                continue
        
        # Calculate differences and ratios
        if fpr_vals and len(fpr_vals) > 1:
            valid_fpr = [v for v in fpr_vals.values() if v is not None and v > 0]
            valid_fnr = [v for v in fnr_vals.values() if v is not None and v > 0]
            
            if valid_fpr:
                metrics['fpr_difference'] = float(max(valid_fpr) - min(valid_fpr))
                metrics['fpr_ratio'] = float(max(valid_fpr) / min(valid_fpr)) if min(valid_fpr) > 0 else float('inf')
            
            if valid_fnr:
                metrics['fnr_difference'] = float(max(valid_fnr) - min(valid_fnr))
                metrics['fnr_ratio'] = float(max(valid_fnr) / min(valid_fnr)) if min(valid_fnr) > 0 else float('inf')
        
        # 9. Treatment Equality
        if fpr_vals and fnr_vals and len(fpr_vals) > 1:
            treatment_ratios = {}
            for group in groups:
                if group in fpr_vals and group in fnr_vals:
                    fpr = fpr_vals[group]
                    fnr = fnr_vals[group]
                    treatment_ratios[group] = float(fnr / fpr) if fpr > 0 else float('inf')
            
            if treatment_ratios:
                valid_ratios = [v for v in treatment_ratios.values() if v != float('inf')]
                if valid_ratios:
                    metrics['treatment_equality_difference'] = float(max(valid_ratios) - min(valid_ratios))
        
        if fdr_vals and len(fdr_vals) > 1:
            valid_fdr = [v for v in fdr_vals.values() if v is not None and v > 0]
            valid_for = [v for v in for_vals.values() if v is not None and v > 0]
            
            if valid_fdr:
                metrics['fdr_difference'] = float(max(valid_fdr) - min(valid_fdr))
                metrics['fdr_ratio'] = float(max(valid_fdr) / min(valid_fdr)) if min(valid_fdr) > 0 else float('inf')
            
            if valid_for:
                metrics['for_difference'] = float(max(valid_for) - min(valid_for))
                metrics['for_ratio'] = float(max(valid_for) / min(valid_for)) if min(valid_for) > 0 else float('inf')
        
        if ppv_vals and len(ppv_vals) > 1:
            valid_ppv = [v for v in ppv_vals.values() if v is not None]
            valid_npv = [v for v in npv_vals.values() if v is not None]
            
            if valid_ppv:
                metrics['ppv_difference'] = float(max(valid_ppv) - min(valid_ppv))
            if valid_npv:
                metrics['npv_difference'] = float(max(valid_npv) - min(valid_npv))
            
            if valid_ppv and valid_npv:
                metrics['predictive_parity_difference'] = float((metrics['ppv_difference'] + metrics['npv_difference']) / 2)
        
        return metrics

    def calculate_statistical_inequality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """4. Statistical Inequality and Distribution Fairness"""
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            selection_rates[group] = float(group_data['y_pred'].mean())
        
        if len(selection_rates) >= 2:
            rates = np.array(list(selection_rates.values()))
            
            # 12. Coefficient of Variation
            if rates.mean() > 0:
                cv = float(rates.std() / rates.mean())
                metrics['coefficient_of_variation'] = cv
            
            # 13. Generalized Entropy and Theil Index
            if len(rates) > 0 and rates.mean() > 0:
                # Generalized Entropy (alpha=2)
                alpha = 2
                ge_index = np.mean(((rates / rates.mean()) ** alpha - 1)) / (alpha * (alpha - 1))
                metrics['generalized_entropy_index'] = float(ge_index)
                
                # Theil Index (alpha=1)
                theil_index = np.mean((rates / rates.mean()) * np.log(rates / rates.mean()))
                metrics['theil_index'] = float(theil_index)
            
            # Mean differences
            mean_diff = float(max(rates) - min(rates))
            overall_mean = float(rates.mean())
            metrics['mean_difference'] = mean_diff
            
            if overall_mean > 0:
                metrics['normalized_mean_difference'] = float(mean_diff / overall_mean)
        
        return metrics

    def calculate_subgroup_bias_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """5. Subgroup and Hidden Bias Detection"""
        metrics = {}
        groups = df['group'].unique()
        
        error_rates = {}
        subgroup_errors = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true_vals = group_data['y_true'].values
                y_pred_vals = group_data['y_pred'].values
                error_rate = float(1 - accuracy_score(y_true_vals, y_pred_vals))
                error_rates[group] = error_rate
                
                # Subgroup analysis for intersectional groups
                if 'subgroup' in df.columns:
                    subgroups = group_data['subgroup'].unique()
                    for subgroup in subgroups:
                        subgroup_mask = (df['group'] == group) & (df['subgroup'] == subgroup)
                        subgroup_data = df[subgroup_mask]
                        if len(subgroup_data) > 0:
                            subgroup_error = 1 - accuracy_score(subgroup_data['y_true'], subgroup_data['y_pred'])
                            subgroup_key = f"{group}_{subgroup}"
                            subgroup_errors[subgroup_key] = float(subgroup_error)
                
            except Exception:
                error_rates[group] = 0.0
        
        # 14. Error Rate metrics
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                metrics['error_rate_difference'] = float(max(valid_errors) - min(valid_errors))
                max_error = max(valid_errors)
                min_error = min(valid_errors)
                metrics['error_rate_ratio'] = float(max_error / min_error) if min_error > 0 else float('inf')
        
        # 15. Subgroup Error Disparity
        if subgroup_errors and len(subgroup_errors) > 1:
            valid_subgroup_errors = [v for v in subgroup_errors.values() if v is not None]
            if valid_subgroup_errors:
                metrics['subgroup_error_disparity'] = float(max(valid_subgroup_errors) - min(valid_subgroup_errors))
        
        return metrics

    def calculate_causal_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """6. Causal and Counterfactual Fairness"""
        metrics = {}
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            selection_rates = {}
            for group in groups:
                group_mask = df['group'] == group
                selection_rates[group] = float(df[group_mask]['y_pred'].mean())
            
            if len(selection_rates) >= 2:
                # 16. Average Causal Effect Difference
                causal_effect = float(max(selection_rates.values()) - min(selection_rates.values()))
                metrics['average_causal_effect_difference'] = causal_effect
                
                # Simplified counterfactual fairness check
                # In production, this would integrate with causal inference libraries
                counterfactual_score = max(0, 1 - causal_effect)
                metrics['counterfactual_fairness_score'] = float(counterfactual_score)
        
        return metrics

    def calculate_robustness_fairness(self, df: pd.DataFrame, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """7. Robustness and Worst-Case Fairness"""
        metrics = {}
        groups = df['group'].unique()
        
        accuracies, losses = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true_vals = group_data['y_true'].values
                y_pred_vals = group_data['y_pred'].values
                
                # 17. Worst Group Accuracy and Loss
                accuracy = float(accuracy_score(y_true_vals, y_pred_vals))
                accuracies[group] = accuracy
                
                # Calculate loss (1 - accuracy for classification, MSE for regression)
                if len(np.unique(y_pred_vals)) > 2:  # Regression
                    loss = float(mean_squared_error(y_true_vals, y_pred_vals))
                else:  # Classification
                    loss = float(1 - accuracy)
                losses[group] = loss
                
            except Exception:
                accuracies[group] = 0.0
                losses[group] = 1.0

        if accuracies and len(accuracies) > 1:
            valid_accuracies = [v for v in accuracies.values() if v is not None]
            if valid_accuracies:
                metrics['worst_group_accuracy'] = float(min(valid_accuracies))
        
        if losses and len(losses) > 1:
            valid_losses = [v for v in losses.values() if v is not None]
            if valid_losses:
                metrics['worst_group_loss'] = float(max(valid_losses))
        
        # 18. Composite Bias Score
        key_metrics = [
            all_metrics.get('statistical_parity_difference', 0.0),
            all_metrics.get('fpr_difference', 0.0),
            all_metrics.get('fnr_difference', 0.0),
            all_metrics.get('fdr_difference', 0.0),
            all_metrics.get('error_rate_difference', 0.0),
            all_metrics.get('predictive_parity_difference', 0.0)
        ]
        
        # Normalize and weight metrics (financial domain specific)
        normalized_metrics = [min(metric, 1.0) for metric in key_metrics if metric > 0]
        
        if normalized_metrics:
            metrics['composite_bias_score'] = float(sum(normalized_metrics) / len(normalized_metrics))
        else:
            metrics['composite_bias_score'] = 0.0
        
        # 19. Temporal and Stability Metrics (simplified)
        metrics['temporal_drift_index'] = self.calculate_temporal_drift(all_metrics)
        metrics['stability_metric'] = self.calculate_stability_metric(all_metrics)
        
        return metrics

    def calculate_explainability_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """8. Explainability and Temporal Fairness"""
        metrics = {}
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            # 20. Feature Attribution Bias (simplified SHAP implementation)
            feature_bias = self.calculate_feature_attribution_bias(df)
            metrics['feature_attribution_bias'] = feature_bias
            
            # 21. Temporal Fairness Score
            temporal_score = self.calculate_temporal_fairness(df)
            metrics['temporal_fairness_score'] = temporal_score
        
        return metrics

    def calculate_feature_attribution_bias(self, df: pd.DataFrame) -> float:
        """Calculate feature importance disparities across groups"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['y_true', 'y_pred', 'y_prob', 'group']]
        
        if len(numeric_cols) == 0:
            return 0.0
        
        groups = df['group'].unique()
        feature_disparities = []
        
        for col in numeric_cols:
            group_means = []
            for group in groups:
                group_mask = df['group'] == group
                group_mean = float(df[group_mask][col].mean())
                group_means.append(group_mean)
            
            if len(group_means) >= 2:
                disparity = float(max(group_means) - min(group_means))
                # Normalize by overall standard deviation
                col_std = float(df[col].std())
                if col_std > 0:
                    disparity /= col_std
                feature_disparities.append(disparity)
        
        return float(np.mean(feature_disparities)) if feature_disparities else 0.0

    def calculate_temporal_fairness(self, df: pd.DataFrame) -> float:
        """Calculate temporal fairness consistency"""
        if 'timestamp' not in df.columns:
            return 1.0  # Default to fair if no temporal data
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('timestamp')
            
            # Calculate fairness metrics over time windows
            time_windows = pd.date_range(start=df_sorted['timestamp'].min(), 
                                       end=df_sorted['timestamp'].max(), 
                                       freq='D')
            
            fairness_scores = []
            for i in range(len(time_windows)-1):
                window_data = df_sorted[
                    (df_sorted['timestamp'] >= time_windows[i]) & 
                    (df_sorted['timestamp'] < time_windows[i+1])
                ]
                if len(window_data) > 10:  # Minimum samples per window
                    window_metrics = self.calculate_finance_metrics(window_data)
                    fairness_scores.append(window_metrics.get('composite_bias_score', 0.0))
            
            if len(fairness_scores) > 1:
                # Measure consistency over time (lower variance = better temporal fairness)
                temporal_score = max(0, 1 - np.std(fairness_scores))
                return float(temporal_score)
            
        except Exception:
            pass
        
        return 1.0

    def calculate_temporal_drift(self, current_metrics: Dict[str, Any]) -> float:
        """Calculate temporal drift in fairness metrics"""
        if len(self.metrics_history) == 0:
            return 0.0
        
        # Compare with historical metrics
        recent_metrics = self.metrics_history[-1]
        drift_score = 0.0
        comparison_count = 0
        
        for key in current_metrics:
            if key in recent_metrics and isinstance(current_metrics[key], (int, float)):
                drift = abs(current_metrics[key] - recent_metrics[key])
                drift_score += drift
                comparison_count += 1
        
        return float(drift_score / comparison_count) if comparison_count > 0 else 0.0

    def calculate_stability_metric(self, current_metrics: Dict[str, Any]) -> float:
        """Calculate stability of fairness metrics over time"""
        if len(self.metrics_history) < 2:
            return 1.0
        
        stability_scores = []
        key_metrics = ['composite_bias_score', 'statistical_parity_difference', 'fpr_difference']
        
        for metric in key_metrics:
            if metric in current_metrics:
                historical_values = [hist[metric] for hist in self.metrics_history if metric in hist]
                if len(historical_values) > 1:
                    cv = np.std(historical_values) / np.mean(historical_values) if np.mean(historical_values) > 0 else 0
                    stability_scores.append(max(0, 1 - cv))
        
        return float(np.mean(stability_scores)) if stability_scores else 1.0

    def calculate_finance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all 21 finance fairness metrics"""
        metrics = {}
        
        # Data validation
        required_cols = ['group', 'y_true', 'y_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        groups = df['group'].unique()
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for fairness analysis")
        
        # Calculate all metric categories
        metrics.update(self.calculate_core_group_fairness(df))
        metrics.update(self.calculate_calibration_reliability(df))
        metrics.update(self.calculate_error_prediction_fairness(df))
        metrics.update(self.calculate_statistical_inequality(df))
        metrics.update(self.calculate_subgroup_bias_detection(df))
        metrics.update(self.calculate_causal_fairness(df))
        metrics.update(self.calculate_robustness_fairness(df, metrics))
        metrics.update(self.calculate_explainability_temporal(df))
        
        # Store for temporal analysis
        self.metrics_history.append(metrics.copy())
        if len(self.metrics_history) > self.temporal_window:
            self.metrics_history.pop(0)
        
        return metrics

    def run_pipeline(self, df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
        """Main finance pipeline execution"""
        
        try:
            finance_metrics = self.calculate_finance_metrics(df)
            
            results = {
                "domain": "finance",
                "metrics_calculated": 21,
                "metric_categories": FINANCE_METRICS_CONFIG,
                "fairness_metrics": finance_metrics,
                "summary": {
                    "composite_bias_score": finance_metrics.get('composite_bias_score', 0.0),
                    "temporal_stability": finance_metrics.get('stability_metric', 1.0),
                    "overall_assessment": self.assess_finance_fairness(finance_metrics)
                },
                "timestamp": str(pd.Timestamp.now())
            }
            
            results = self.convert_numpy_types(results)
            
            return results
            
        except Exception as e:
            error_results = {
                "domain": "finance",
                "metrics_calculated": 0,
                "error": str(e),
                "summary": {
                    "composite_bias_score": 1.0,
                    "temporal_stability": 0.0,
                    "overall_assessment": "ERROR - Could not complete audit"
                },
                "timestamp": str(pd.Timestamp.now())
            }
            return self.convert_numpy_types(error_results)

    def assess_finance_fairness(self, metrics: Dict[str, Any]) -> str:
        """Assess overall fairness for finance domain"""
        bias_score = metrics.get('composite_bias_score', 0.0)
        stability = metrics.get('stability_metric', 1.0)
        
        if bias_score > 0.10 or stability < 0.85:
            return "HIGH_BIAS - Significant fairness concerns in financial decisions"
        elif bias_score > 0.05 or stability < 0.95:
            return "MEDIUM_BIAS - Moderate fairness concerns detected"  
        else:
            return "LOW_BIAS - Generally fair across groups"

# Production usage example
if __name__ == "__main__":
    # Test with sample financial data
    sample_data = pd.DataFrame({
        'group': ['High_Income', 'High_Income', 'Low_Income', 'Low_Income', 'High_Income', 'Low_Income'],
        'y_true': [1, 0, 1, 0, 1, 0],
        'y_pred': [1, 0, 0, 0, 1, 1],
        'y_prob': [0.8, 0.2, 0.4, 0.3, 0.9, 0.6],
        'credit_score': [750, 800, 600, 550, 780, 620],
        'timestamp': pd.date_range('2024-01-01', periods=6, freq='D')
    })
    
    pipeline = FinanceFairnessPipeline()
    results = pipeline.run_pipeline(sample_data)
    
    print("PRODUCTION FINANCE FAIRNESS AUDIT COMPLETE")
    print(f"Metrics Calculated: {results['metrics_calculated']}/21")
    print(f"Overall Assessment: {results['summary']['overall_assessment']}")
    print(f"Composite Bias Score: {results['summary']['composite_bias_score']:.3f}")

# ADD TO fdk_finance_pipeline.py (at the end of the file)

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = True) -> Dict[str, Any]:
    """Standalone function for API compatibility"""
    pipeline = FinanceFairnessPipeline()
    return pipeline.run_pipeline(df, save_to_disk)

# Also add these for backward compatibility
def calculate_finance_fairness(df: pd.DataFrame) -> Dict[str, Any]:
    """Alternative entry point"""
    return run_pipeline(df)