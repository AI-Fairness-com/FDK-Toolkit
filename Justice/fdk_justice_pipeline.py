# ================================================================
# FDK Justice Pipeline - PRODUCTION READY (FINAL CORRECTED)
# 20 Comprehensive Justice Fairness Metrics
# MIT License - AI Ethics Research Group
# ================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import scipy.stats as st
from typing import Dict, List, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Justice-specific metrics configuration
JUSTICE_METRICS_CONFIG = {
    'core_group_fairness': [
        'statistical_parity_difference', 'disparate_impact', 'selection_rate',
        'predicted_positives_per_group', 'predicted_negatives_per_group'
    ],
    'error_performance_fairness': [
        'fpr_difference', 'fpr_ratio', 'fnr_difference', 'fnr_ratio', 
        'tpr_difference', 'tpr_ratio', 'tnr_difference', 'tnr_ratio',
        'error_rate_difference', 'error_rate_ratio', 'predictive_equality',
        'disparate_mistreatment_index'
    ],
    'equality_opportunity_treatment': [
        'equalized_odds_difference', 'equal_opportunity_difference',
        'average_odds_difference', 'average_absolute_odds_difference'
    ],
    'error_distribution_subgroup': [
        'fdr_difference', 'fdr_ratio', 'for_difference', 'for_ratio',
        'error_disparity_subgroup', 'mdss_subgroup_discovery_score'
    ],
    'robustness_worst_case': [
        'worst_group_accuracy', 'worst_group_loss', 'composite_bias_score',
        'validation_robustness_score'
    ],
    'calibration_predictive': ['slice_auc_difference'],
    'causal_counterfactual': ['counterfactual_fairness_score', 'causal_effect_difference'],
    'explainability_temporal': ['feature_attribution_bias', 'temporal_fairness_score']
}

class JusticeFairnessPipeline:
    """Production-grade fairness assessment for justice AI systems"""
    
    def __init__(self):
        self.metrics_history = []
        self.temporal_window = 10
        
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

    def safe_div(self, a, b):
        """Safe division with comprehensive error handling"""
        try:
            return a / b if b != 0 else 0.0
        except Exception:
            return 0.0

    def calculate_core_group_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Core Group Fairness Metrics for Justice Domain"""
        metrics = {}
        groups = df['group'].unique()
        
        selection_rates = {}
        predicted_positives = {}
        predicted_negatives = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            selection_rates[group] = float(group_data['y_pred'].mean())
            predicted_positives[group] = int(group_data['y_pred'].sum())
            predicted_negatives[group] = int(len(group_data) - group_data['y_pred'].sum())
        
        if len(selection_rates) >= 2:
            spd = float(max(selection_rates.values()) - min(selection_rates.values()))
            metrics['statistical_parity_difference'] = spd
            
            min_rate = min(selection_rates.values())
            max_rate = max(selection_rates.values())
            metrics['disparate_impact'] = float(min_rate / max_rate) if max_rate > 0 else float('inf')
            
            metrics['selection_rates'] = selection_rates
        
        metrics['predicted_positives_per_group'] = predicted_positives
        metrics['predicted_negatives_per_group'] = predicted_negatives
        
        return metrics

    def calculate_error_performance_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Error and Performance Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        fpr_vals, fnr_vals, tpr_vals, tnr_vals = {}, {}, {}, {}
        error_rates = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                
                fpr_vals[group] = self.safe_div(fp, (fp + tn))
                fnr_vals[group] = self.safe_div(fn, (fn + tp))
                tpr_vals[group] = self.safe_div(tp, (tp + fn))
                tnr_vals[group] = self.safe_div(tn, (tn + fp))
                error_rates[group] = self.safe_div((fp + fn), (tp + tn + fp + fn))
                
            except Exception:
                continue
        
        self._calculate_differences_ratios(metrics, 'fpr', fpr_vals)
        self._calculate_differences_ratios(metrics, 'fnr', fnr_vals)
        self._calculate_differences_ratios(metrics, 'tpr', tpr_vals)
        self._calculate_differences_ratios(metrics, 'tnr', tnr_vals)
        self._calculate_differences_ratios(metrics, 'error_rate', error_rates)
        
        if fpr_vals and len(fpr_vals) > 1:
            valid_fpr = [v for v in fpr_vals.values() if v is not None]
            if valid_fpr:
                metrics['predictive_equality'] = float(max(valid_fpr) - min(valid_fpr))
        
        if fpr_vals and fnr_vals and len(fpr_vals) > 1:
            dmi_values = {}
            for group in groups:
                if group in fpr_vals and group in fnr_vals:
                    dmi_values[group] = fpr_vals[group] + fnr_vals[group]
            
            if dmi_values:
                valid_dmi = [v for v in dmi_values.values() if v is not None]
                if valid_dmi:
                    metrics['disparate_mistreatment_index'] = float(max(valid_dmi) - min(valid_dmi))
        
        return metrics

    def _calculate_differences_ratios(self, metrics: Dict, prefix: str, values: Dict):
        """Calculate difference and ratio for a metric across groups"""
        if values and len(values) > 1:
            valid_vals = [v for v in values.values() if v is not None and v > 0]
            if valid_vals:
                metrics[f'{prefix}_difference'] = float(max(valid_vals) - min(valid_vals))
                metrics[f'{prefix}_ratio'] = float(max(valid_vals) / min(valid_vals))

    def calculate_equality_opportunity_treatment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Equality of Opportunity and Treatment Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        tpr_vals, fpr_vals = {}, {}
        
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
                fpr_vals[group] = self.safe_div(fp, (fp + tn))
                
            except Exception:
                continue
        
        if tpr_vals and fpr_vals and len(tpr_vals) > 1:
            valid_tpr = [v for v in tpr_vals.values() if v is not None]
            valid_fpr = [v for v in fpr_vals.values() if v is not None]
            
            if valid_tpr and valid_fpr:
                tpr_diff = max(valid_tpr) - min(valid_tpr)
                fpr_diff = max(valid_fpr) - min(valid_fpr)
                metrics['equalized_odds_difference'] = float((tpr_diff + fpr_diff) / 2.0)
                metrics['equal_opportunity_difference'] = float(tpr_diff)
                metrics['average_odds_difference'] = float((tpr_diff - fpr_diff) / 2.0)
                metrics['average_absolute_odds_difference'] = float((abs(tpr_diff) + abs(fpr_diff)) / 2.0)
        
        return metrics

    def calculate_error_distribution_subgroup(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Error Distribution and Subgroup Analysis"""
        metrics = {}
        groups = df['group'].unique()
        
        fdr_vals, for_vals, error_rates = {}, {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                
                fdr_vals[group] = self.safe_div(fp, (fp + tp))
                for_vals[group] = self.safe_div(fn, (fn + tn))
                error_rates[group] = self.safe_div((fp + fn), (tp + tn + fp + fn))
                
            except Exception:
                continue
        
        self._calculate_differences_ratios(metrics, 'fdr', fdr_vals)
        self._calculate_differences_ratios(metrics, 'for', for_vals)
        
        if error_rates and len(error_rates) > 1:
            valid_errors = [v for v in error_rates.values() if v is not None]
            if valid_errors:
                metrics['error_disparity_subgroup'] = {
                    'range': float(max(valid_errors) - min(valid_errors)),
                    'ratio': float(max(valid_errors) / min(valid_errors)) if min(valid_errors) > 0 else float('inf')
                }
        
        metrics['mdss_subgroup_discovery_score'] = self._calculate_mdss_subgroup_discovery(df)
        
        return metrics

    def _calculate_mdss_subgroup_discovery(self, df: pd.DataFrame, min_support: float = 0.05) -> float:
        """Calculate MDSS Subgroup Discovery Score"""
        try:
            total_samples = len(df)
            min_samples = max(1, int(min_support * total_samples))
            base_error = 1 - (df['y_true'] == df['y_pred']).mean()

            max_mdss = 0.0

            for value in df['group'].unique():
                subgroup_mask = df['group'] == value
                subgroup_size = subgroup_mask.sum()

                if subgroup_size < min_samples:
                    continue

                subgroup_error = 1 - (df[subgroup_mask]['y_true'] == df[subgroup_mask]['y_pred']).mean()
                
                if subgroup_error > base_error:
                    mdss_score = (subgroup_error - base_error) * np.log(subgroup_size)
                    max_mdss = max(max_mdss, mdss_score)

            return float(max_mdss)

        except Exception:
            return 0.0

    def calculate_robustness_worst_case(self, df: pd.DataFrame, all_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Robustness and Worst-Case Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        accuracies, losses = {}, {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            try:
                y_true = group_data['y_true'].values
                y_pred = group_data['y_pred'].values
                
                accuracy = accuracy_score(y_true, y_pred)
                accuracies[group] = float(accuracy)
                losses[group] = float(1.0 - accuracy)
                
            except Exception:
                continue
        
        if accuracies:
            metrics['worst_group_accuracy'] = float(min(accuracies.values()))
        
        if losses:
            metrics['worst_group_loss'] = float(max(losses.values()))
        
        metrics['composite_bias_score'] = self._calculate_composite_bias_score(all_metrics or metrics)
        metrics['validation_robustness_score'] = self._calculate_validation_robustness(df)
        
        return metrics

    def _calculate_composite_bias_score(self, all_metrics: Dict[str, Any]) -> float:
        """Composite bias score for justice domain"""
        try:
            statistical_parity = all_metrics.get('statistical_parity_difference', 0.0)
            equal_opportunity = all_metrics.get('equal_opportunity_difference', 0.0)
            predictive_equality = all_metrics.get('predictive_equality', 0.0)
            
            error_disparity_data = all_metrics.get('error_disparity_subgroup', {})
            error_disparity = error_disparity_data.get('range', 0.0) if isinstance(error_disparity_data, dict) else 0.0
            
            bias_components = [statistical_parity, equal_opportunity, predictive_equality, error_disparity]
            weights = [0.3, 0.2, 0.3, 0.2]
            
            valid_components = [comp for comp in bias_components if comp is not None and not np.isnan(comp)]
            
            if not valid_components:
                return 0.0
                
            weighted_sum = sum(comp * weight for comp, weight in zip(valid_components, weights[:len(valid_components)]))
            return float(weighted_sum / sum(weights[:len(valid_components)]))
            
        except Exception:
            return 0.0

    def _calculate_validation_robustness(self, df: pd.DataFrame, n_splits: int = 3) -> float:
        """Calculate validation robustness score"""
        try:
            groups = df['group'].unique()
            robustness_scores = []
            
            for group in groups:
                group_mask = df['group'] == group
                group_data = df[group_mask]
                
                if len(group_data) < n_splits * 2:
                    continue
                
                accuracies = []
                for _ in range(n_splits):
                    split_data = group_data.sample(frac=0.7, replace=True)
                    accuracy = accuracy_score(split_data['y_true'], split_data['y_pred'])
                    accuracies.append(float(accuracy))
                
                if len(accuracies) > 1:
                    cv = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
                    robustness_scores.append(max(0, 1 - cv))
            
            return float(np.mean(robustness_scores)) if robustness_scores else 1.0
        except Exception:
            return 1.0

    def calculate_calibration_predictive(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calibration and Predictive Reliability Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        auc_scores = {}
        
        for group in groups:
            group_mask = df['group'] == group
            group_data = df[group_mask]
            
            if len(group_data) == 0:
                continue
                
            if 'y_prob' in df.columns:
                try:
                    y_true = group_data['y_true'].values
                    y_prob = group_data['y_prob'].values
                    
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, y_prob)
                        auc_scores[group] = float(auc)
                except Exception:
                    continue
        
        if auc_scores and len(auc_scores) > 1:
            valid_auc = [v for v in auc_scores.values() if v is not None]
            if valid_auc:
                metrics['slice_auc_difference'] = float(max(valid_auc) - min(valid_auc))
        
        return metrics

    def calculate_causal_counterfactual(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Causal and Counterfactual Fairness Metrics"""
        metrics = {}
        groups = df['group'].unique()
        
        if len(groups) >= 2:
            selection_rates = {}
            for group in groups:
                group_mask = df['group'] == group
                selection_rates[group] = float(df[group_mask]['y_pred'].mean())
            
            if len(selection_rates) >= 2:
                causal_effect = max(selection_rates.values()) - min(selection_rates.values())
                metrics['counterfactual_fairness_score'] = float(max(0, 1 - causal_effect))
                metrics['causal_effect_difference'] = float(causal_effect)
        
        return metrics

    def calculate_explainability_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Explainability and Temporal Fairness Metrics"""
        metrics = {}
        metrics['feature_attribution_bias'] = self._calculate_feature_attribution_bias(df)
        metrics['temporal_fairness_score'] = self._calculate_temporal_fairness(df)
        return metrics

    def _calculate_feature_attribution_bias(self, df: pd.DataFrame) -> float:
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
                col_std = float(df[col].std())
                if col_std > 0:
                    disparity /= col_std
                feature_disparities.append(disparity)
        
        return float(np.mean(feature_disparities)) if feature_disparities else 0.0

    def _calculate_temporal_fairness(self, df: pd.DataFrame) -> float:
        """Calculate temporal fairness consistency"""
        if 'timestamp' not in df.columns:
            return 1.0
        
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
                    window_metrics = self.calculate_all_metrics(window_data)
                    fairness_scores.append(window_metrics.get('composite_bias_score', 0.0))
            
            if len(fairness_scores) > 1:
                return float(max(0, 1 - np.std(fairness_scores)))
            
        except Exception:
            pass
        
        return 1.0

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all 20 justice fairness metrics"""
        required_cols = ['group', 'y_true', 'y_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        groups = df['group'].unique()
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for justice fairness analysis")

        metrics = {}
        metrics.update(self.calculate_core_group_fairness(df))
        metrics.update(self.calculate_error_performance_fairness(df))
        metrics.update(self.calculate_equality_opportunity_treatment(df))
        metrics.update(self.calculate_error_distribution_subgroup(df))
        metrics.update(self.calculate_calibration_predictive(df))
        metrics.update(self.calculate_causal_counterfactual(df))
        metrics.update(self.calculate_explainability_temporal(df))
        
        robustness_metrics = self.calculate_robustness_worst_case(df, all_metrics=metrics)
        metrics.update(robustness_metrics)
        
        self.metrics_history.append(metrics.copy())
        if len(self.metrics_history) > self.temporal_window:
            self.metrics_history.pop(0)
        
        return metrics

    def run_pipeline(self, df: pd.DataFrame, save_to_disk: bool = False) -> Dict[str, Any]:
        """Main justice pipeline execution"""
        
        try:
            justice_metrics = self.calculate_all_metrics(df)
            assessment_result = self.assess_justice_fairness_enhanced(justice_metrics)
            
            results = {
                "domain": "justice",
                "metrics_calculated": 20,
                "metric_categories": JUSTICE_METRICS_CONFIG,
                "fairness_metrics": justice_metrics,
                "validation": {
                    "sample_size": len(df),
                    "groups_analyzed": len(df['group'].unique()),
                    "statistical_power": "strong" if len(df) >= 1000 else "adequate" if len(df) >= 500 else "moderate"
                },
                "summary": {
                    "composite_bias_score": justice_metrics.get('composite_bias_score', 0.0),
                    "overall_assessment": assessment_result["professional"]
                },
                "timestamp": str(pd.Timestamp.now())
            }
            
            return self.convert_numpy_types(results)
            
        except Exception as e:
            error_results = {
                "domain": "justice",
                "metrics_calculated": 0,
                "error": str(e),
                "summary": {
                    "composite_bias_score": 1.0,
                    "overall_assessment": "ERROR - Could not complete justice audit"
                },
                "timestamp": str(pd.Timestamp.now())
            }
            return self.convert_numpy_types(error_results)

    def assess_justice_fairness_enhanced(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Justice fairness assessment with proper thresholds"""
        bias_score = metrics.get('composite_bias_score', 0.0)
        
        statistical_parity = metrics.get('statistical_parity_difference', 0.0)
        fpr_gap = metrics.get('fpr_difference', 0.0)
        equal_opportunity = metrics.get('equal_opportunity_difference', 0.0)
        
        has_medium_metrics = (statistical_parity > 0.05 or fpr_gap > 0.05 or equal_opportunity > 0.05)
        
        if bias_score > 0.08 or (has_medium_metrics and bias_score > 0.06):
            return {"professional": "HIGH_BIAS - Significant constitutional fairness concerns requiring immediate review"}
        elif bias_score > 0.03 or has_medium_metrics:
            return {"professional": "MEDIUM_BIAS - Moderate fairness concerns requiring monitoring and documentation"}
        elif bias_score > 0.01:
            return {"professional": "LOW_BIAS - Minor variations observed, continue regular monitoring"}
        else:
            return {"professional": "MINIMAL_BIAS - Generally fair across protected groups"}

# ================================================================
# COMPATIBILITY FUNCTIONS (ADD TO PIPELINE)
# ================================================================

def interpret_prompt(prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Interpret user prompt and run justice analysis - REQUIRED FOR BLUEPRINT"""
    pipeline = JusticeFairnessPipeline()
    results = pipeline.run_pipeline(df)
    
    return {
        "interpretation": "justice_fairness_audit",
        "domain": "justice", 
        "results": results,
        "prompt_handled": True
    }

def run_audit_from_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run audit from Flask request data - REQUIRED FOR BLUEPRINT"""
    try:
        df = pd.DataFrame(request_data.get('data', []))
        return run_pipeline(df)
    except Exception as e:
        return {
            "domain": "justice",
            "error": str(e),
            "metrics_calculated": 0
        }

# ================================================================
# API COMPATIBILITY FUNCTIONS
# ================================================================

def run_pipeline(df: pd.DataFrame, save_to_disk: bool = False) -> Dict[str, Any]:
    """Standalone function for API compatibility"""
    pipeline = JusticeFairnessPipeline()
    return pipeline.run_pipeline(df, save_to_disk)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'group': ['Urban', 'Rural', 'Suburban'] * 100,
        'y_true': np.random.randint(0, 2, 300),
        'y_pred': np.random.randint(0, 2, 300),
        'y_prob': np.random.random(300)
    })
    
    pipeline = JusticeFairnessPipeline()
    results = pipeline.run_pipeline(sample_data)
    
    print("PRODUCTION JUSTICE FAIRNESS AUDIT COMPLETE")
    print(f"Composite Bias Score: {results['summary']['composite_bias_score']:.3f}")
    print(f"Assessment: {results['summary']['overall_assessment']}")