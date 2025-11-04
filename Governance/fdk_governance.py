# ================================================================
# FDK Governance App - Interactive Fairness Audit for Governance Domain
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime

# ================================================================
# BLUEPRINT CONFIGURATION
# ================================================================

governance_bp = Blueprint('governance', __name__, template_folder='templates')

# ================================================================
# FOLDER CONFIGURATION
# ================================================================

UPLOAD_FOLDER = 'uploads_governance'
REPORT_FOLDER = 'reports_governance'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# PIPELINE IMPORT
# ================================================================

from .fdk_governance_pipeline import run_pipeline

# ================================================================
# UNIVERSAL GOVERNANCE AUTO-DETECTION SYSTEM
# ================================================================

def detect_governance_column_mappings(df, columns):
    """Universal auto-detection for governance datasets"""
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {}
    
    for col in columns:
        reasoning[col] = ""
    
    # Layer 1: Direct matching for standard column names
    for col in columns:
        col_lower = col.lower()
        if col_lower in ['group', 'protected_group', 'demographic', 'category', 'segment', 'protected_attribute']:
            suggestions['group'] = col
            reasoning[col] = "Direct match: group/protected attribute column"
            continue
        elif col_lower in ['y_true', 'actual', 'true', 'outcome', 'target', 'label', 'ground_truth']:
            suggestions['y_true'] = col
            reasoning[col] = "Direct match: true outcomes/target variable"
            continue
        elif col_lower in ['y_pred', 'predicted', 'prediction', 'estimate', 'model_output']:
            suggestions['y_pred'] = col
            reasoning[col] = "Direct match: model predictions"
            continue
        elif col_lower in ['y_prob', 'probability', 'score', 'confidence', 'risk_score', 'propensity']:
            suggestions['y_prob'] = col
            reasoning[col] = "Direct match: probability/confidence scores"
            continue

    # Layer 2: Governance-specific keyword detection
    for col in columns:
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP: Governance-specific groups
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            governance_group_keywords = ['constituency', 'district', 'demographic', 'ethnicity', 'gender', 
                                       'socioeconomic', 'region', 'voter_segment', 'political_affiliation',
                                       'age_group', 'income_bracket', 'geographic_zone']
            if any(keyword in col.lower() for keyword in governance_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Governance domain: Constituent groups for fairness analysis"
                continue
                
        # Y_TRUE: Governance outcomes
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                governance_true_keywords = ['service_allocation', 'funding_distribution', 'policy_impact',
                                          'resource_access', 'benefit_approval', 'program_eligibility',
                                          'complaint_resolution', 'approval_status']
                if any(keyword in col.lower() for keyword in governance_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Governance domain: Policy outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED: Governance predictions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                governance_pred_keywords = ['risk_score', 'eligibility_prediction', 'allocation_algorithm', 
                                          'priority_ranking', 'service_classification', 'decision_model']
                if any(keyword in col.lower() for keyword in governance_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Governance domain: Policy algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB: Probability scores
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                               'estimate', 'calibration', 'confidence_score', 'rating']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Governance domain: Policy probability scores (0-1 range)"
                    continue
    
    # Layer 3: Statistical fallbacks
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 20:
                suggestions['group'] = col
                reasoning[col] = "Statistical fallback: Categorical groups (2-20 unique values)"
                break
        if not suggestions['group']:
            for col in columns:
                if df[col].dtype in ['int64', 'float64'] and 2 <= df[col].nunique() <= 10:
                    suggestions['group'] = col
                    reasoning[col] = "Statistical fallback: Numeric groups (2-10 unique values)"
                    break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                if col != suggestions['y_pred']:
                    suggestions['y_true'] = col
                    reasoning[col] = "Statistical fallback: Binary outcomes (2 unique values)"
                    break
                
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Statistical fallback: Binary predictions (2 unique values)"
                break
    
    return suggestions, reasoning

# ================================================================
# GOVERNANCE-SPECIFIC REPORT GENERATION WITH DATASET OVERVIEW
# ================================================================

def build_governance_summaries(audit: dict) -> list:
    """Governance-specific human-readable summary"""
    lines = []
    
    # PROFESSIONAL SUMMARY
    lines.append("=== GOVERNANCE PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Policy Equity & Access Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # STANDARDIZED DATASET OVERVIEW SECTION
    lines.append("📊 DATASET OVERVIEW:")
    if "validation" in audit:
        validation_info = audit["validation"]
        lines.append(f"   → Total Constituents Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Constituent Groups: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_records = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Constituents Analyzed: {total_records}")
        lines.append(f"   → Constituent Groups: {num_groups}")
        if num_groups <= 10:
            lines.append(f"   → Group Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Group: {max(group_counts.values())} constituents")
            lines.append(f"   → Smallest Group: {min(group_counts.values())} constituents")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    if "error" in audit:
        lines.append("❌ AUDIT ERROR DETECTED:")
        lines.append(f"   → Error: {audit['error']}")
        lines.append("   → The fairness audit could not complete due to technical issues.")
        lines.append("   → Please check your dataset format and try again.")
        lines.append("")
        return lines
    
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    if composite_score is not None:
        lines.append("1) OVERALL POLICY EQUITY ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.10:
            lines.append("   → SEVERITY: HIGH - Significant equity concerns in policy decisions")
            lines.append("   → ACTION: IMMEDIATE POLICY EQUITY REVIEW REQUIRED")
        elif composite_score > 0.03:
            lines.append("   → SEVERITY: MEDIUM - Moderate equity concerns detected")
            lines.append("   → ACTION: SCHEDULE POLICY REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal equity concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    fairness_metrics = audit.get("fairness_metrics", {})
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) SERVICE ALLOCATION DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in service allocation across constituent groups")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable service allocation variations")
        else:
            lines.append("     ✅ LOW: Consistent service allocation across constituent groups")
        lines.append("")
    
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) POLICY ACCESS DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some constituent groups experience many more false service denials")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false service denials")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates across constituent groups")
        lines.append("")
    
    lines.append("4) POLICY EQUITY RECOMMENDATIONS:")
    if composite_score and composite_score > 0.10:
        lines.append("   🚨 IMMEDIATE EQUITY ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive policy equity investigation")
        lines.append("   • Review service allocation decision-making processes")
        lines.append("   • Implement policy equity mitigation protocols")
        lines.append("   • Consider external policy equity audit")
    elif composite_score and composite_score > 0.03:
        lines.append("   ⚖️  RECOMMENDED POLICY REVIEW:")
        lines.append("   • Schedule systematic policy equity review")
        lines.append("   • Monitor service allocation patterns by constituent group")
        lines.append("   • Document policy equity considerations")
        lines.append("   • Plan procedural improvements for equity")
    else:
        lines.append("   ✅ POLICY EQUITY STANDARDS MAINTAINED:")
        lines.append("   • Continue regular policy equity monitoring")
        lines.append("   • Maintain current policy equity standards")
        lines.append("   • Document policy equity assessment")
    lines.append("")
    
    # PUBLIC SUMMARY
    lines.append("=== PUBLIC INTEREST SUMMARY ===")
    lines.append("Plain-English Interpretation for Policy Transparency:")
    lines.append("")
    
    high_bias_detected = False
    medium_bias_detected = False
    
    if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.1:
        high_bias_detected = True
    if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.1:
        high_bias_detected = True
    if 'average_odds_difference' in fairness_metrics and fairness_metrics['average_odds_difference'] > 0.1:
        high_bias_detected = True
    
    if not high_bias_detected:
        if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.05:
            medium_bias_detected = True
        if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.05:
            medium_bias_detected = True
    
    if high_bias_detected or (composite_score and composite_score > 0.10):
        lines.append("🔴 SIGNIFICANT EQUITY CONCERNS")
        lines.append("")
        lines.append("This policy tool shows substantial differences in how it treats different constituent groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Service allocation decisions may be inconsistent across constituent groups")
        lines.append("• Some groups may experience different service access rates")
        lines.append("• Additional review of policy processes is recommended")
    elif medium_bias_detected or (composite_score and composite_score > 0.03):
        lines.append("🟡 MODERATE EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This policy tool generally works fairly but shows some variation across constituent groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its policy decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing policy equity monitoring is recommended")
    else:
        lines.append("🟢 GOOD EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This policy tool demonstrates consistent treatment across all constituent groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Policy decisions are applied consistently regardless of background")
        lines.append("• The tool meets policy equity standards")
        lines.append("• Treatment is equitable across different constituent groups")
    
    lines.append("")
    
    lines.append("=== POLICY EQUITY DISCLAIMER ===")
    lines.append("This policy equity audit complies with:")
    lines.append("• Equal Protection laws")
    lines.append("• Policy equity regulations")
    lines.append("• Anti-discrimination policy laws")
    lines.append("• Algorithmic accountability frameworks in governance")
    lines.append("")
    lines.append("POLICY NOTICE: This tool is for policy equity assessment only and does not:")
    lines.append("• Provide policy guarantees or outcomes")
    lines.append("• Determine policy eligibility")
    lines.append("• Replace professional policy consultation")
    lines.append("")
    lines.append("For policy equity concerns, consult qualified policy professionals.")
    
    return lines

# ================================================================
# GOVERNANCE ROUTES DEFINITION (BLUEPRINT COMPATIBLE)
# ================================================================

@governance_bp.route('/governance-upload')
def governance_upload_page():
    """Governance upload page"""
    session.clear()
    return render_template('upload_governance.html')

@governance_bp.route('/governance-audit', methods=['POST'])
def start_governance_audit_process():
    """Process governance dataset upload"""
    if 'file' not in request.files:
        return render_template("result_governance.html", title="Error", message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_governance.html", title="Error", message="Empty filename.", summary=None)

    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_governance.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        suggested_mappings, column_reasoning = detect_governance_column_mappings(df, columns)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_governance.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_governance.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_governance.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@governance_bp.route('/governance-run-audit')
def run_governance_audit_with_mapping():
    """Run governance audit with detected mapping"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_governance.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_governance.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
        df_mapped = pd.DataFrame()
        
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()
        
        for col in df_mapped.columns:
            if df_mapped[col].dtype == 'bool':
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_integer_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_float_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(float)
        
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_governance.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_governance.html", title="Error",
                                    message=f"Column '{col}' is not a Series. This should never happen.", summary=None)
        
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"governance_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        summary_lines = build_governance_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_governance.html",
            title="Governance Fairness Audit Completed",
            message="Your governance dataset was audited successfully using 27 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Governance audit failed: {str(e)}"
        return render_template("result_governance.html", title="Governance Audit Failed",
                              message=error_msg, summary=None)

@governance_bp.route('/download-governance-report/<filename>')
def download_governance_report(filename):
    """Serve governance audit reports"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404