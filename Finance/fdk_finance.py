# ================================================================
# FDK Finance - Fairness Audit for Financial Services
# ================================================================
# Interactive fairness audit for credit, lending, and financial AI systems
# Compliant with ECOA, Fair Lending, and regulatory requirements
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime

from .fdk_finance_pipeline import run_pipeline

# ================================================================
# Configuration
# ================================================================

UPLOAD_FOLDER = 'uploads_finance'
REPORT_FOLDER = 'reports_finance'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# Finance Blueprint
# ================================================================

finance_bp = Blueprint('finance', __name__, template_folder='templates')

# ================================================================
# Finance-Specific Detection Functions
# ================================================================

def detect_finance_column_mappings(df, columns):
    """
    Universal auto-detection for finance datasets.
    Handles both real financial data and synthetic/test datasets.
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {col: "" for col in columns}
    
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

    # Layer 2: Finance-specific keyword detection
    for col in columns:
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # Group detection: Finance-specific demographic groups
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            finance_group_keywords = [
                'income', 'credit_score', 'employment', 'location', 'region', 'age_group', 
                'education', 'demographic', 'ethnicity', 'gender', 'race', 'geographic',
                'experience', 'seniority', 'tenure', 'bracket', 'level', 'class'
            ]
            if any(keyword in col.lower() for keyword in finance_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Finance domain: Customer groups for fairness analysis"
                continue
                
        # True outcomes: Finance binary outcomes
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                finance_true_keywords = [
                    'default', 'repayment', 'fraud', 'approval', 'denial', 'delinquency', 
                    'outcome', 'result', 'status', 'chargeoff', 'bankruptcy', 'defaulted',
                    'approved', 'denied', 'accepted', 'rejected'
                ]
                if any(keyword in col.lower() for keyword in finance_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Finance domain: Financial outcomes (binary: 0/1)"
                    continue
                    
        # Predictions: Finance algorithm outputs
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                finance_pred_keywords = [
                    'prediction', 'risk_score', 'algorithm', 'model', 'assessment', 'score',
                    'decision', 'recommendation', 'classification', 'output'
                ]
                if any(keyword in col.lower() for keyword in finance_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Finance domain: Financial algorithm predictions (binary: 0/1)"
                    continue
                    
        # Probability scores: Risk probabilities
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = [
                    'probability', 'score', 'risk', 'likelihood', 'confidence', 'propensity',
                    'estimate', 'calibration', 'confidence_score'
                ]
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Finance domain: Risk probability scores (0-1 range)"
                    continue
    
    # Layer 3: Statistical fallbacks for unmapped columns
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
# Finance Summary Generation (Preserves Regulatory Logic)
# ================================================================

def build_finance_summaries(audit: dict) -> list:
    """Generate finance-specific professional and public summaries"""
    lines = []
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    
    # Professional Summary
    lines.extend(_build_finance_professional_summary(audit, composite_score))
    
    # Public Summary
    lines.extend(_build_finance_public_summary(composite_score))
    
    # Regulatory Disclaimer
    lines.extend(_build_regulatory_disclaimer())
    
    return lines

def _build_finance_professional_summary(audit: dict, composite_score: float) -> list:
    """Build professional finance summary with regulatory risk assessment"""
    lines = [
        "=== FINANCE PROFESSIONAL SUMMARY ===",
        "FDK Fairness Audit — Financial Services & Credit Interpretation",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # STANDARDIZED DATASET OVERVIEW
    lines.append("📊 DATASET OVERVIEW:")
    if "validation" in audit:
        validation_info = audit["validation"]
        lines.append(f"   → Total Applications Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Applicant Groups: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_records = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Applications Analyzed: {total_records}")
        lines.append(f"   → Applicant Groups: {num_groups}")
        if num_groups <= 10:
            lines.append(f"   → Group Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Group: {max(group_counts.values())} applications")
            lines.append(f"   → Smallest Group: {min(group_counts.values())} applications")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    # Overall Assessment
    if composite_score is not None:
        lines.append("1) OVERALL FAIRNESS ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.10:
            lines.append("   → SEVERITY: HIGH - Significant fairness concerns in financial decisions")
            lines.append("   → ACTION: IMMEDIATE REGULATORY REVIEW REQUIRED")
        elif composite_score > 0.03:
            lines.append("   → SEVERITY: MEDIUM - Moderate fairness concerns detected")
            lines.append("   → ACTION: SCHEDULE COMPLIANCE REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal fairness concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Finance Metrics Analysis
    lines.extend(_analyze_finance_metrics(audit))
    
    # Financial Recommendations
    lines.extend(_generate_finance_recommendations(composite_score))
    
    return lines

def _analyze_finance_metrics(audit: dict) -> list:
    """Analyze key financial fairness metrics"""
    lines = []
    fairness_metrics = audit.get("fairness_metrics", {})
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) APPROVAL RATE DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in approval rates across groups")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable approval rate variations")
        else:
            lines.append("     ✅ LOW: Consistent approval rates across groups")
        lines.append("")
    
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) ERROR DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some groups experience many more false denials")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false denials")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates")
        lines.append("")
    
    return lines

def _generate_finance_recommendations(composite_score: float) -> list:
    """Generate regulatory recommendations based on risk level"""
    lines = ["4) FINANCIAL & COMPLIANCE RECOMMENDATIONS:"]
    
    if composite_score > 0.10:
        lines.extend([
            "   🚨 IMMEDIATE REGULATORY ACTIONS REQUIRED:",
            "   • Conduct comprehensive bias investigation",
            "   • Review credit decision-making processes",
            "   • Implement bias mitigation protocols",
            "   • Consider external regulatory audit"
        ])
    elif composite_score > 0.03:
        lines.extend([
            "   ⚖️  RECOMMENDED COMPLIANCE REVIEW:",
            "   • Schedule systematic fairness review",
            "   • Monitor decision patterns by customer group",
            "   • Document fairness considerations",
            "   • Plan procedural improvements"
        ])
    else:
        lines.extend([
            "   ✅ REGULATORY COMPLIANCE MAINTAINED:",
            "   • Continue regular fairness monitoring",
            "   • Maintain current compliance standards",
            "   • Document compliance assessment"
        ])
    lines.append("")
    
    return lines

def _build_finance_public_summary(composite_score: float) -> list:
    """Build public-friendly finance summary"""
    lines = [
        "=== PUBLIC INTEREST SUMMARY ===",
        "Plain-English Interpretation for Transparency:",
        ""
    ]
    
    if composite_score > 0.10:
        lines.extend([
            "🔴 SIGNIFICANT FAIRNESS CONCERNS",
            "",
            "This financial tool shows substantial differences in how it treats different customer groups.",
            "",
            "What this means:",
            "• Credit decisions may be inconsistent across demographic groups",
            "• Some groups may experience different approval rates",
            "• Additional review of decision processes is recommended"
        ])
    elif composite_score > 0.03:
        lines.extend([
            "🟡 MODERATE FAIRNESS ASSESSMENT",
            "",
            "This financial tool generally works fairly but shows some variation across customer groups.",
            "",
            "What this means:",
            "• The tool is mostly consistent in its decisions",
            "• Some small differences in treatment may exist",
            "• Ongoing monitoring is recommended"
        ])
    else:
        lines.extend([
            "🟢 GOOD FAIRNESS ASSESSMENT",
            "",
            "This financial tool demonstrates consistent treatment across all customer groups.",
            "",
            "What this means:",
            "• Decisions are applied consistently regardless of background",
            "• The tool meets fairness standards",
            "• Treatment is equitable across different groups"
        ])
    lines.append("")
    
    return lines

def _build_regulatory_disclaimer() -> list:
    """Build regulatory compliance disclaimer"""
    return [
        "=== REGULATORY DISCLAIMER ===",
        "This fairness audit complies with:",
        "• Equal Credit Opportunity Act (ECOA)",
        "• Fair Lending regulations",
        "• Consumer Financial Protection Bureau guidelines",
        "• Algorithmic accountability frameworks",
        "",
        "REGULATORY NOTICE: This tool is for fairness assessment only and does not:",
        "• Provide financial advice or guarantees",
        "• Determine credit eligibility or outcomes",
        "• Replace professional financial consultation",
        "",
        "For regulatory concerns, consult qualified compliance professionals."
    ]

# ================================================================
# Flask Routes
# ================================================================

@finance_bp.route('/finance-upload')
def finance_upload_page():
    """Finance upload page - clean session start"""
    session.clear()
    return render_template('upload_finance.html')

@finance_bp.route('/finance-audit', methods=['POST'])
def start_finance_audit_process():
    """Process finance dataset upload and auto-detect columns"""
    if 'file' not in request.files:
        return render_template("result_finance.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_finance.html", title="Error", 
                             message="Empty filename.", summary=None)

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        # Validate dataset structure
        if len(columns) < 3:
            return render_template("result_finance.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # Auto-detect column mappings
        suggested_mappings, column_reasoning = detect_finance_column_mappings(df, columns)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if not suggested_mappings.get(m)]
        
        if missing_required:
            return render_template("result_finance.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}.", summary=None)
        
        # Store in session
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        
        # Count detected key features
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_finance.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_finance.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@finance_bp.route('/finance-run-audit')
def run_finance_audit_with_mapping():
    """Execute finance fairness audit with detected mappings"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_finance.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_finance.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
        # Create clean mapped DataFrame
        df_mapped = pd.DataFrame()
        
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()
        
        # Convert data types to Python native
        for col in df_mapped.columns:
            if df_mapped[col].dtype == 'bool':
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_integer_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_float_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(float)
        
        # Validate required columns exist after mapping
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_finance.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Run finance audit pipeline
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"finance_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        # Generate finance-specific summary
        summary_lines = build_finance_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_finance.html",
            title="Finance Fairness Audit Completed",
            message="Your finance dataset was audited successfully using 14 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Finance audit failed: {str(e)}"
        return render_template("result_finance.html", title="Finance Audit Failed",
                              message=error_msg, summary=None)

@finance_bp.route('/download-finance-report/<filename>')
def download_finance_report(filename):
    """Serve finance audit reports for download"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404