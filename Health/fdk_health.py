# ================================================================
# FDK Health - Fairness Audit for Healthcare AI Systems
# ================================================================
# Universal API for healthcare dataset fairness auditing
# Compliant with EU AI Act and medical device regulations
# UPDATED: Unified Intelligent System Integration
# FIXED: Circular import issue resolved + improved y_pred detection
# ================================================================

import os
import json
import pandas as pd
import numpy as np
import traceback
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime

# ================================================================
# Configuration
# ================================================================

UPLOAD_FOLDER = 'uploads_health'
REPORT_FOLDER = 'reports_health'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# Health Blueprint
# ================================================================

health_bp = Blueprint('health', __name__, template_folder='templates')

# ================================================================
# Pipeline Import
# ================================================================

from .fdk_health_pipeline import run_pipeline

# ================================================================
# UNIFIED HEALTH COLUMN DETECTION WITH UNIVERSAL INTELLIGENT SYSTEM
# ================================================================

def detect_health_column_mappings(df, columns, test_type='pre_implementation', user_target=None):
    """
    Unified column detection with intelligent system integration.
    Priority: User Override > Intelligent Detection > Healthcare-specific detection
    
    Args:
        df: Pandas DataFrame containing healthcare data
        columns: List of column names in the dataset
        test_type: Type of test ('pre_implementation' or 'post_implementation')
        user_target: User-specified target column (optional override)
        
    Returns:
        tuple: (suggestions_dict, reasoning_dict, intelligent_suggestion) containing column mappings, explanations, and intelligent suggestion metadata
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None, 'timestamp': None}
    reasoning = {}
    intelligent_suggestion = None
    
    for col in columns:
        reasoning[col] = ""
    
    # STEP 1: INTELLIGENT TARGET SELECTION (using local import to avoid circular dependency)
    if test_type in ['pre_implementation', 'post_implementation']:
        try:
            # Import locally to avoid circular imports
            try:
                from FDK import intelligent_target_selection
                intelligent_suggestion = intelligent_target_selection(df, test_type, 'health')
                if intelligent_suggestion and intelligent_suggestion in df.columns:
                    suggestions['y_true'] = intelligent_suggestion
                    reasoning[intelligent_suggestion] = f"✅ FDK INTELLIGENT SELECTION (test_type: {test_type})"
                    print(f"🎯 FDK Intelligent suggests: {intelligent_suggestion} for {test_type}")
            except ImportError:
                print("⚠️ FDK intelligent selection not available")
        except Exception as e:
            print(f"⚠️ FDK intelligent selection failed: {e}")
    
    # STEP 2: USER OVERRIDE (TAKES PRIORITY)
    if user_target and user_target in df.columns:
        suggestions['y_true'] = user_target
        override_source = 'FDK' if intelligent_suggestion else 'auto-detection'
        reasoning[user_target] = f"✅ USER MANUAL SELECTION (overrides {override_source})"
        print(f"🎯 User overrides to: {user_target}")
    
    # STEP 3: HEALTHCARE-SPECIFIC DETECTION (for group, y_pred, y_prob, and fallback)
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Detect patient demographic groups
        if not suggestions['group']:
            if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.nunique() > 1):
                health_group_keywords = ['race', 'ethnic', 'gender', 'age_group', 'location', 
                                       'region', 'type', 'category', 'demographic', 'patient_group', 
                                       'population', 'background', 'ethnicity', 'age', 'sex', 'demographics']
                if any(keyword in col.lower() for keyword in health_group_keywords):
                    suggestions['group'] = col
                    reasoning[col] = "Patient groups for fairness analysis"
                    continue
                    
        # Y_TRUE COLUMN: Only if not already set by FDK or user
        if not suggestions['y_true']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                # More flexible binary detection
                if (set(unique_vals).issubset({0, 1}) or 
                    set(unique_vals).issubset({0.0, 1.0}) or
                    set(unique_vals).issubset({True, False}) or
                    set(unique_vals).issubset({1, 2})):
                    health_true_keywords = ['diagnosis', 'outcome', 'result', 'disease', 'positive', 
                                          'mortality', 'readmission', 'complication', 'adverse_event', 
                                          'survival', 'recovery', 'death', 'event']
                    if any(keyword in col.lower() for keyword in health_true_keywords):
                        suggestions['y_true'] = col
                        reasoning[col] = "Medical outcomes (binary)"
                        continue
                        
        # Y_PRED COLUMN: Detect algorithm predictions (binary)
        if not suggestions['y_pred']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                # More flexible binary detection for predictions
                if (set(unique_vals).issubset({0, 1}) or 
                    set(unique_vals).issubset({0.0, 1.0}) or
                    set(unique_vals).issubset({True, False}) or
                    set(unique_vals).issubset({1, 2})):
                    # First check for prediction-specific keywords
                    health_pred_keywords = ['prediction', 'pred', 'predicted', 'y_pred', 'yhat', 
                                          'estimate', 'forecast', 'model_output', 'algorithm']
                    if any(keyword in col.lower() for keyword in health_pred_keywords):
                        suggestions['y_pred'] = col
                        reasoning[col] = "Model predictions (binary)"
                        continue
                    # Fallback: if column name sounds like a prediction and isn't the target
                    elif col != suggestions['y_true'] and ('pred' in col.lower() or 'score' in col.lower()):
                        suggestions['y_pred'] = col
                        reasoning[col] = "Suggested model predictions (binary)"
                        continue
                        
        # Y_PROB COLUMN: Detect probability scores (continuous 0-1)
        if not suggestions['y_prob']:
            if col_data.dtype in ['float64', 'float32']:
                if len(unique_vals) > 2:
                    # Check if values are in probability range (0-1)
                    try:
                        if col_data.dropna().between(0, 1).all() or col_data.dropna().between(0.0, 1.0).all():
                            prob_keywords = ['probability', 'prob', 'score', 'risk', 'likelihood', 
                                           'propensity', 'confidence', 'calibration']
                            if any(keyword in col.lower() for keyword in prob_keywords):
                                suggestions['y_prob'] = col
                                reasoning[col] = "Probability scores (0-1 range)"
                                continue
                    except:
                        pass

        # TIMESTAMP: optional column enabling temporal fairness / model decay metrics
        if not suggestions.get('timestamp'):
            if any(keyword in col.lower() for keyword in ['timestamp', 'date', 'decision_date', 'time', 'datetime']):
                try:
                    pd.to_datetime(df[col], errors='raise')
                    suggestions['timestamp'] = col
                    reasoning[col] = "Detected as a parseable date/time column for temporal fairness metrics"
                    continue
                except Exception:
                    pass
    
    # STEP 4: FALLBACK DETECTION
    # Group fallback: first categorical column
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested patient groups (categorical)"
                break
                
    # y_true fallback: first binary column
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                suggestions['y_true'] = col
                reasoning[col] = "Suggested medical outcomes (binary)"
                break
                
    # y_pred fallback: first binary column that isn't y_true
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Suggested model predictions (binary)"
                break
        # If still not found, use y_true as y_pred (common in pre-implementation tests)
        if not suggestions['y_pred'] and suggestions['y_true']:
            suggestions['y_pred'] = suggestions['y_true']
            reasoning[suggestions['y_true']] += " (also used as predictions for baseline test)"
                
    # y_prob fallback: first float column with reasonable range
    if not suggestions['y_prob']:
        for col in columns:
            if df[col].dtype in ['float64', 'float32']:
                if col != suggestions['y_true'] and col != suggestions['y_pred']:
                    suggestions['y_prob'] = col
                    reasoning[col] = "Suggested probability scores"
                    break
    
    return suggestions, reasoning, intelligent_suggestion

# ================================================================
# Summary Generation (Preserves Risk Communication Logic)
# ================================================================

def build_human_summaries(audit_response: dict) -> list:
    """
    Generate consistent professional and patient summaries.
    CRITICAL: Preserves the risk communication consistency between summaries.
    """
    lines = []
    
    try:
        # Extract composite score from the nested structure
        summary_data = audit_response.get('summary', {})
        composite_score = summary_data.get('composite_bias_score')
        fairness_metrics = audit_response.get('fairness_metrics', {})
        
        # Professional Summary
        lines.extend(_build_professional_summary(audit_response, composite_score, fairness_metrics))
        
        # Patient/Public Summary  
        lines.extend(_build_patient_summary(composite_score, fairness_metrics))
        
        # Legal Disclaimer
        lines.extend(_build_legal_disclaimer())
        
    except Exception as e:
        lines = [
            "=== PROFESSIONAL SUMMARY ===",
            "FDK Fairness Audit — Healthcare Professional Interpretation",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 DATASET OVERVIEW:",
            "→ Dataset statistics: Information not available",
            "",
            "⚠️ SUMMARY GENERATION ERROR:",
            f"→ Could not generate detailed summary: {str(e)}"
        ]
    
    return lines

def _build_professional_summary(audit_response: dict, composite_score: float, fairness_metrics: dict) -> list:
    """Build professional healthcare summary with risk assessment"""
    lines = [
        "=== PROFESSIONAL SUMMARY ===",
        "FDK Fairness Audit — Healthcare Professional Interpretation",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # ========== STANDARDIZED DATASET OVERVIEW ==========
    lines.append("📊 DATASET OVERVIEW:")
    
    # Extract dataset statistics from the ACTUAL validation section
    validation = audit_response.get('validation', {})
    
    if validation:
        sample_size = validation.get('sample_size')
        groups_analyzed = validation.get('groups_analyzed') 
        statistical_power = validation.get('statistical_power', 'moderate')
        
        if sample_size:
            lines.append(f"→ Total Patients Analyzed: {sample_size}")
        if groups_analyzed:
            lines.append(f"→ Patient Groups: {groups_analyzed}")
        if statistical_power:
            lines.append(f"→ Statistical Power: {statistical_power.title()}")
    else:
        # Fallback to basic info
        lines.append("→ Basic dataset analysis completed")
    
    lines.append("")
    # ========== END DATASET OVERVIEW ==========
    
    # Overall Assessment
    if composite_score is not None:
        lines.append("1) OVERALL FAIRNESS ASSESSMENT:")
        lines.append(f"→ Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.25:
            lines.append("→ SEVERITY: CRITICAL - Immediate intervention required")
            lines.append("→ ACTION: IMMEDIATE CLINICAL REVIEW")
        elif composite_score > 0.15:
            lines.append("→ SEVERITY: HIGH - Significant fairness concerns detected")
            lines.append("→ ACTION: URGENT REVIEW REQUIRED")
        elif composite_score > 0.08:
            lines.append("→ SEVERITY: MEDIUM - Moderate fairness concerns")
            lines.append("→ ACTION: SCHEDULE REVIEW")
        else:
            lines.append("→ SEVERITY: LOW - Minimal fairness concerns")
            lines.append("→ ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Performance Gaps Analysis
    lines.extend(_analyze_performance_gaps(fairness_metrics))
    
    # Calibration Analysis
    lines.extend(_analyze_calibration(fairness_metrics))
    
    # Worst Group Analysis
    lines.extend(_analyze_worst_group(fairness_metrics))
    
    # Professional Recommendations
    lines.extend(_generate_recommendations(composite_score, fairness_metrics))
    
    return lines

def _analyze_performance_gaps(fairness_metrics: dict) -> list:
    """Analyze performance disparities across groups"""
    lines = []
    
    # Extract performance metrics from the actual structure
    tpr_diff = fairness_metrics.get('tpr_difference', 0)
    fpr_diff = fairness_metrics.get('fpr_difference', 0)
    ppv_diff = fairness_metrics.get('ppv_difference', 0)
    error_disp = fairness_metrics.get('error_disparity_subgroup', {})
    error_range = error_disp.get('range', 0) if error_disp else 0
    
    if any([tpr_diff > 0, fpr_diff > 0, ppv_diff > 0, error_range > 0]):
        lines.append("2) KEY PERFORMANCE DISPARITIES:")
        
        # True Positive Rate (Detection Accuracy)
        if tpr_diff > 0:
            lines.append(f"→ Detection Rate Gap (TPR): {tpr_diff:.3f}")
            if tpr_diff > 0.15:
                lines.append("🚨 HIGH: Significant variation in true positive detection")
            elif tpr_diff > 0.05:
                lines.append("⚠️  MEDIUM: Noticeable detection rate differences")
            else:
                lines.append("✅ LOW: Consistent detection across groups")
        
        # False Positive Rate (False Alarms)
        if fpr_diff > 0:
            lines.append(f"→ False Alarm Gap (FPR): {fpr_diff:.3f}")
            if fpr_diff > 0.15:
                lines.append("🚨 HIGH: Some groups experience many more false alarms")
            elif fpr_diff > 0.05:
                lines.append("⚠️  MEDIUM: Moderate variation in false alarms")
            else:
                lines.append("✅ LOW: Consistent false alarm rates")
        
        # Error Rate Disparity
        if error_range > 0:
            lines.append(f"→ Overall Error Gap: {error_range:.3f}")
            if error_range > 0.15:
                lines.append("🚨 HIGH: Significant accuracy differences between groups")
            elif error_range > 0.05:
                lines.append("⚠️  MEDIUM: Noticeable accuracy variations")
            else:
                lines.append("✅ LOW: Consistent performance across groups")
        
        # Positive Predictive Value
        if ppv_diff > 0:
            lines.append(f"→ Predictive Value Gap (PPV): {ppv_diff:.3f}")
            if ppv_diff > 0.15:
                lines.append("🚨 HIGH: Large variation in prediction reliability")
            elif ppv_diff > 0.05:
                lines.append("⚠️  MEDIUM: Moderate prediction value differences")
        lines.append("")
    
    return lines

def _analyze_calibration(audit: dict) -> list:
    """Analyze prediction reliability across groups"""
    lines = []
    calibration_gap = audit.get("calibration_gap", {})
    
    if calibration_gap.get("by_group"):
        lines.append("3) PREDICTION RELIABILITY (Calibration):")
        calib_values = [v for v in calibration_gap["by_group"].values() if v is not None]
        if calib_values:
            max_calib_gap = max(calib_values) if calib_values else 0
            lines.append(f"   → Maximum Calibration Gap: {max_calib_gap:.3f}")
            if max_calib_gap > 0.1:
                lines.append("     🚨 HIGH: Prediction scores may be unreliable for some groups")
                lines.append("     → Clinical interpretation of scores may vary by patient group")
            elif max_calib_gap > 0.05:
                lines.append("     ⚠️  MEDIUM: Moderate reliability concerns")
                lines.append("     → Consider group-specific score interpretation")
            else:
                lines.append("     ✅ LOW: Consistent score reliability across groups")
        lines.append("")
    
    return lines

def _analyze_worst_group(audit: dict) -> list:
    """Identify highest risk group"""
    lines = []
    subgroup_analysis = audit.get("subgroup_analysis", {})
    worst_group_info = subgroup_analysis.get("worst_group_analysis", {})
    
    if worst_group_info.get("overall_worst_group"):
        lines.append("4) HIGHEST RISK GROUP IDENTIFIED:")
        lines.append(f"   → Group: {worst_group_info['overall_worst_group']}")
        lines.append(f"   → Severity Score: {worst_group_info.get('overall_severity_score', 0):.3f}")
        lines.append("   → This group experiences the most significant performance issues")
        lines.append("")
    
    return lines

def _generate_recommendations(composite_score: float, fairness_metrics: dict = None) -> list:
    """Generate recommendations that match the actual risk assessment"""
    lines = []
    
    # Extract key performance metrics
    tpr_diff = fairness_metrics.get('tpr_difference', 0) if fairness_metrics else 0
    fpr_diff = fairness_metrics.get('fpr_difference', 0) if fairness_metrics else 0
    ppv_diff = fairness_metrics.get('ppv_difference', 0) if fairness_metrics else 0
    
    # Check for ANY high disparities regardless of composite score
    has_high_disparities = tpr_diff > 0.15 or ppv_diff > 0.15
    has_medium_disparities = tpr_diff > 0.05 or fpr_diff > 0.05 or ppv_diff > 0.05
    
    lines.append("5) PROFESSIONAL RECOMMENDATIONS:")
    
    if composite_score > 0.15 or has_high_disparities:
        lines.append("🚨 IMMEDIATE ACTIONS REQUIRED:")
        lines.append("• Investigate root causes of high detection rate disparities")
        lines.append("• Address predictive value inconsistencies across groups") 
        lines.append("• Implement targeted model retraining for affected populations")
        lines.append("• Conduct clinical review before further deployment")
        lines.append("• Enhance monitoring for high-risk patient groups")
        
    elif composite_score > 0.08 or has_medium_disparities:
        lines.append("⚠️  ENHANCED MONITORING NEEDED:")
        lines.append("• Schedule model performance review within 30 days")
        lines.append("• Analyze subgroup performance patterns in detail")
        lines.append("• Consider calibration adjustments for consistent performance")
        lines.append("• Document monitoring plan and review timeline")
        lines.append("• Prepare mitigation strategies if disparities worsen")
        
    else:
        lines.append("✅ MAINTENANCE ACTIONS:")
        lines.append("• Continue regular fairness monitoring")
        lines.append("• Maintain current practices")
        lines.append("• Document this positive fairness assessment")
    
    lines.append("")
    return lines

def _build_patient_summary(composite_score: float, fairness_metrics: dict = None) -> list:
    """Build patient/public-friendly summary that matches professional risk assessment"""
    lines = [
        "=== PATIENT / PUBLIC-FRIENDLY SUMMARY ===",
        "Plain-English Interpretation for App Users:",
        ""
    ]
    
    # Extract key performance metrics to inform the summary
    tpr_diff = fairness_metrics.get('tpr_difference', 0) if fairness_metrics else 0
    fpr_diff = fairness_metrics.get('fpr_difference', 0) if fairness_metrics else 0
    error_range = fairness_metrics.get('error_disparity_subgroup', {}).get('range', 0) if fairness_metrics else 0
    
    # Determine overall assessment based on ALL metrics, not just composite score
    has_high_disparities = tpr_diff > 0.15 or fpr_diff > 0.15
    has_medium_disparities = (tpr_diff > 0.05 or fpr_diff > 0.05 or error_range > 0.05)
    
    if composite_score > 0.15 or has_high_disparities:
        lines.append("🔴 FAIRNESS CONCERNS IDENTIFIED")
        lines.append("")
        lines.append("This health tool shows some performance differences across user groups.")
        lines.append("")
        lines.append("What this means for you:")
        lines.append("• Results may vary slightly depending on your background")
        lines.append("• The tool is being improved for better consistency")
        lines.append("• Overall reliability remains acceptable for most uses")
        lines.append("")
        lines.append("Recommended next steps:")
        lines.append("• Continue using the tool with awareness of these findings")
        lines.append("• Discuss any concerns with healthcare providers")
        lines.append("• Report any inconsistent experiences you notice")
        
    elif composite_score > 0.08 or has_medium_disparities:
        lines.append("🟡 MODERATE FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This health tool works reasonably well across user groups with minor variations.")
        lines.append("")
        lines.append("What this means for you:")
        lines.append("• The tool provides generally reliable results")
        lines.append("• Some small performance differences exist between groups")
        lines.append("• These variations are being monitored and addressed")
        lines.append("")
        lines.append("Recommended next steps:")
        lines.append("• Use this tool as part of your health management")
        lines.append("• Be aware that results may have slight variations")
        lines.append("• Always consult professionals for medical decisions")
        
    else:
        lines.append("🟢 GOOD FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This health tool works consistently well across all user groups.")
        lines.append("")
        lines.append("What this means for you:")
        lines.append("• The tool provides reliable results regardless of your background")
        lines.append("• You can trust the accuracy and fairness of predictions")
        lines.append("• Performance is consistent across different user groups")
        lines.append("")
        lines.append("Recommended next steps:")
        lines.append("• Use this tool with confidence")
        lines.append("• Continue following standard health guidelines")
        lines.append("• Always consult professionals for serious concerns")
    
    return lines

def _build_legal_disclaimer() -> list:
    """Build regulatory compliance disclaimer"""
    return [
        "=== IMPORTANT DISCLAIMER ===",
        "This fairness audit complies with:",
        "• EU AI Act requirements for high-risk AI systems", 
        "• UN Principles on AI ethics and non-discrimination",
        "• Medical device regulatory frameworks",
        "",
        "LEGAL NOTICE: This tool is for fairness assessment only and does not:",
        "• Provide medical advice or diagnosis",
        "• Replace professional healthcare consultation", 
        "• Guarantee specific medical outcomes",
        "",
        "For medical concerns, always consult qualified healthcare professionals.",
        "In emergency situations, contact local emergency services immediately.",
        "",
        "International Resources:",
        "• World Health Organization: www.who.int", 
        "• Local health authorities in your region",
        "• Certified medical professionals",
        ""
    ]

# ================================================================
# Flask Routes
# ================================================================

@health_bp.route('/health-upload')
def health_upload_page():
    """Health upload page - clean session start"""
    session.clear()
    return render_template('upload_health.html')

@health_bp.route('/health-audit', methods=['POST'])
def start_health_audit_process():
    """Process uploaded health dataset with unified intelligent system"""
    if 'file' not in request.files:
        return render_template("result_health.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_health.html", title="Error", 
                             message="Empty filename.", summary=None)

    # ✅ UNIFIED SYSTEM PARAMETERS
    user_selected_target = request.form.get('target_column', '').strip()
    test_type = request.form.get('test_type', 'pre_implementation')
    
    print(f"🎯 UNIFIED INTELLIGENT SYSTEM: test_type={test_type}, user_target='{user_selected_target}'")

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        # Validate dataset structure
        if len(columns) < 3:
            return render_template("result_health.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # ✅ UNIFIED DETECTION with intelligent system
        suggested_mappings, column_reasoning, intelligent_suggestion = detect_health_column_mappings(
            df, columns, test_type, user_selected_target
        )
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_health.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        # Store in session and show confirmation
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        session['test_type'] = test_type
        session['user_selected_target'] = user_selected_target
        session['intelligent_suggestion'] = intelligent_suggestion
        
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_health.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            filename=file.filename,
            detected_key_features=detected_key_features,
            test_type=test_type,
            intelligent_suggestion=intelligent_suggestion,
            user_selected=user_selected_target if user_selected_target else None
        )
        
    except Exception as e:
        return render_template("result_health.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@health_bp.route('/run-health-audit')
def run_health_audit_with_mapping():
    """Execute fairness audit with universal metadata integration"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    test_type = session.get('test_type', 'pre_implementation')
    user_selected_target = session.get('user_selected_target', '')
    intelligent_suggestion = session.get('intelligent_suggestion', None)
    
    if not dataset_path or not column_mapping:
        return render_template("result_health.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_health.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)

        # Create mapped dataframe
        df_mapped = pd.DataFrame()
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()

        # Carry through any remaining original columns as additional features,
        # excluding pure identifier columns (every value unique -- never a
        # genuine fairness-relevant feature, and can dominate scale-sensitive
        # calculations like feature attribution gaps).
        mapped_originals = set(v for v in column_mapping.values() if v)
        for col in df.columns:
            if col not in mapped_originals and col not in df_mapped.columns:
                if df[col].nunique() < len(df):
                    df_mapped[col] = df[col].copy()
        
        # Final validation
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_health.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Run fairness audit pipeline
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # ✅ UNIVERSAL METADATA ADDITION
        metadata = {
            "target_column_used": column_mapping.get('y_true'),
            "target_column_original": column_mapping.get('y_true'),
            "prediction_column_used": column_mapping.get('y_pred'),
            "group_column_used": column_mapping.get('group'),
            "probability_column_used": column_mapping.get('y_prob'),
            "test_type": test_type,
            "intelligent_suggestion": intelligent_suggestion,
            "user_override_applied": bool(user_selected_target and user_selected_target in df.columns),
            "user_selected_target": user_selected_target if user_selected_target else None,
            "timestamp": datetime.now().isoformat(),
            "dataset_filename": os.path.basename(dataset_path),
            "fdk_version": "health_1.0_unified",
            "column_mapping": column_mapping
        }
        audit_response["metadata"] = metadata
        
        # Validation info if not present
        if "validation" not in audit_response:
            group_counts = df_mapped['group'].value_counts().to_dict()
            audit_response["validation"] = {
                "sample_size": len(df_mapped),
                "groups_analyzed": len(df_mapped['group'].unique()),
                "statistical_power": "strong" if len(df_mapped) >= 1000 else "adequate" if len(df_mapped) >= 500 else "moderate",
                "group_counts": group_counts,
                "test_type": test_type
            }
        else:
            audit_response["validation"]["test_type"] = test_type
        
        # Save detailed report with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"health_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2)
        
        session['report_filename'] = report_filename
        
        # Generate human-readable summary
        summary_lines = build_human_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_health.html",
            title="Health Fairness Audit Completed",
            message=f"Your health dataset was audited successfully using 45 fairness metrics. Test Type: {test_type.replace('_', ' ').title()}",
            summary=summary_text,
            report_filename=session['report_filename'],
            test_type=test_type,
            metadata=metadata
        )
        
    except Exception as e:
        print("Audit pipeline error:")
        traceback.print_exc()
        
        return render_template("result_health.html", title="Health Audit Failed",
                              message=f"Health audit failed: {str(e)}", summary=None)

@health_bp.route('/download-health-report/<filename>')
def download_health_report(filename):
    """Serve health audit reports for download"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@health_bp.route('/')
def index():
    """Home page - redirect to health upload interface"""
    return redirect(url_for('health.health_upload_page'))