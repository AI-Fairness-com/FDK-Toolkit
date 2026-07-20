# ================================================================
# FDK Finance - Fairness Audit for Financial Services
# ================================================================
# Interactive fairness audit for credit, lending, and financial AI systems
# Compliant with ECOA, Fair Lending, and regulatory requirements
# UPDATED: Unified Intelligent System Integration (Fixed to match justice standard)
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime

# ================================================================
# INTELLIGENT SELECTION IMPORT (UNIFIED SYSTEM)
# ================================================================

try:
    from FDK import intelligent_target_selection
    HAS_FDK_INTELLIGENT = True
except ImportError:
    HAS_FDK_INTELLIGENT = False
    print(f"⚠️ FDK intelligent selection not available, using fallback detection")

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
# Pipeline Import
# ================================================================

from .fdk_finance_pipeline import run_pipeline

# ================================================================
# UNIFIED FINANCE COLUMN DETECTION WITH INTELLIGENT SYSTEM
# ================================================================

def detect_finance_column_mappings(df, columns, test_type='pre_implementation', user_target=None):
    """
    Unified column detection with FDK intelligent system integration.
    Priority: FDK Intelligent > User Override > Domain-specific detection
    (Matches justice.py standard implementation)
    
    Args:
        df: Pandas DataFrame containing financial data
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
    
    # STEP 1: FDK INTELLIGENT TARGET SELECTION (Matches justice standard)
    if HAS_FDK_INTELLIGENT and test_type in ['pre_implementation', 'post_implementation']:
        try:
            intelligent_suggestion = intelligent_target_selection(df, test_type, 'finance')
            if intelligent_suggestion and intelligent_suggestion in df.columns:
                suggestions['y_true'] = intelligent_suggestion
                reasoning[intelligent_suggestion] = f"✅ FDK INTELLIGENT SELECTION (test_type: {test_type})"
                print(f"🎯 FDK Intelligent suggests: {intelligent_suggestion} for {test_type}")
        except Exception as e:
            print(f"⚠️ FDK intelligent selection failed: {e}")
    
    # STEP 2: USER OVERRIDE (TAKES PRIORITY) (Matches justice standard)
    if user_target and user_target in df.columns:
        suggestions['y_true'] = user_target
        override_source = 'FDK' if intelligent_suggestion else 'auto-detection'
        reasoning[user_target] = f"✅ USER MANUAL SELECTION (overrides {override_source})"
        print(f"🎯 User overrides to: {user_target}")
    
    # STEP 3: FINANCE-SPECIFIC DETECTION (for group, y_pred, y_prob, and fallback)
    finance_keywords = {
        'group': ['income', 'credit_score', 'employment', 'location', 'region', 'age_group', 
                'education', 'demographic', 'ethnicity', 'gender', 'race', 'geographic',
                'experience', 'seniority', 'tenure', 'bracket', 'level', 'class',
                'customer_group', 'applicant_group', 'protected_attribute', 'segment'],
        'y_true': ['default', 'repayment', 'fraud', 'approval', 'denial', 'delinquency', 
                  'outcome', 'result', 'status', 'chargeoff', 'bankruptcy', 'defaulted',
                  'approved', 'denied', 'accepted', 'rejected', 'target', 'label',
                  'ground_truth', 'actual', 'loan_status', 'credit_outcome'],
        'y_pred': ['prediction', 'risk_score', 'algorithm', 'model', 'assessment', 'score',
                  'decision', 'recommendation', 'classification', 'output', 'predicted',
                  'model_score', 'algorithm_output', 'credit_prediction'],
        'y_prob': ['probability', 'score', 'risk', 'likelihood', 'confidence', 'propensity',
                  'estimate', 'calibration', 'confidence_score', 'risk_probability',
                  'default_probability', 'propensity_score']
    }
    
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Detect customer/demographic groups for financial fairness
        if not suggestions['group']:
            if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.nunique() > 1):
                if any(keyword in col.lower() for keyword in finance_keywords['group']):
                    suggestions['group'] = col
                    reasoning[col] = "Customer/demographic groups for financial fairness analysis"
                    continue
                    
        # Y_TRUE COLUMN: Only if not already set by FDK or user
        if not suggestions['y_true']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}):
                    if any(keyword in col.lower() for keyword in finance_keywords['y_true']):
                        suggestions['y_true'] = col
                        reasoning[col] = "Financial outcomes (binary: 0/1)"
                        continue
                        
        # Y_PRED COLUMN: Detect algorithm predictions (binary)
        if not suggestions['y_pred']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}) and col != suggestions['y_true']:
                    if any(keyword in col.lower() for keyword in finance_keywords['y_pred']):
                        suggestions['y_pred'] = col
                        reasoning[col] = "Financial algorithm predictions (binary: 0/1)"
                        continue
                        
        # Y_PROB COLUMN: Detect probability scores (continuous 0-1)
        if not suggestions['y_prob']:
            if col_data.dtype in ['float64', 'float32']:
                if len(unique_vals) > 2 and col_data.between(0, 1).all():
                    if any(keyword in col.lower() for keyword in finance_keywords['y_prob']):
                        suggestions['y_prob'] = col
                        reasoning[col] = "Risk probability scores (0-1 range)"
                        continue

        # TIMESTAMP: optional column enabling temporal fairness metrics
        if not suggestions['timestamp']:
            if any(keyword in col.lower() for keyword in ['timestamp', 'date', 'decision_date', 'time', 'datetime']):
                try:
                    pd.to_datetime(df[col], errors='raise')
                    suggestions['timestamp'] = col
                    reasoning[col] = "Detected as a parseable date/time column for temporal fairness metrics"
                    continue
                except Exception:
                    pass
    
    # STEP 4: FALLBACK DETECTION (Matches justice standard structure)
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested customer groups (categorical)"
                break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                suggestions['y_true'] = col
                reasoning[col] = "Suggested financial outcomes (binary)"
                break
                
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Suggested financial predictions (binary)"
                break
    
    return suggestions, reasoning, intelligent_suggestion

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
# Flask Routes (Updated with Standard Parameter Reading)
# ================================================================

@finance_bp.route('/finance-upload')
def finance_upload_page():
    """Finance upload page - clean session start"""
    session.clear()
    return render_template('upload_finance.html')

@finance_bp.route('/finance-audit', methods=['POST'])
def start_finance_audit_process():
    """Process finance dataset upload with unified intelligent system"""
    if 'file' not in request.files:
        return render_template("result_finance.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_finance.html", title="Error", 
                             message="Empty filename.", summary=None)

    # ✅ STANDARD UNIFIED PARAMETER READING (Matches justice standard)
    user_selected_target = request.form.get('target_column', '').strip()
    if not user_selected_target:
        user_selected_target = request.form.get('target_column_fallback', '').strip()
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
            return render_template("result_finance.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # ✅ STANDARD DETECTION with test_type and user_target (Matches justice standard)
        suggested_mappings, column_reasoning, intelligent_suggestion = detect_finance_column_mappings(
            df, columns, test_type, user_selected_target
        )
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_finance.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}.", summary=None)
        
        # Store in session (Matches justice standard session keys)
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        session['test_type'] = test_type
        session['user_selected_target'] = user_selected_target
        session['intelligent_suggestion'] = intelligent_suggestion
        
        # Count detected key features
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_finance.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename,
            test_type=test_type,
            intelligent_suggestion=intelligent_suggestion,
            user_selected=user_selected_target if user_selected_target else None
        )
        
    except Exception as e:
        return render_template("result_finance.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@finance_bp.route('/finance-run-audit')
def run_finance_audit_with_mapping():
    """Execute finance fairness audit with unified metadata integration"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    test_type = session.get('test_type', 'pre_implementation')
    user_selected_target = session.get('user_selected_target', '')
    intelligent_suggestion = session.get('intelligent_suggestion', None)
    
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

        # Carry through any remaining original columns as additional features,
        # excluding pure identifier columns (every value unique -- never a
        # genuine fairness-relevant feature, and can dominate scale-sensitive
        # calculations like feature attribution gaps).
        mapped_originals = set(v for v in column_mapping.values() if v)
        for col in df.columns:
            if col not in mapped_originals and col not in df_mapped.columns:
                if df[col].nunique() < len(df):
                    df_mapped[col] = df[col].copy()
        
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
        
        # ✅ STANDARD METADATA ADDITION (Matches justice standard exactly)
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
            "fdk_version": "finance_1.0_unified",
            "column_mapping": column_mapping
        }
        audit_response["metadata"] = metadata
        
        # Add validation info with test_type (Matches justice standard)
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
            message=f"Your finance dataset was audited successfully using 30 fairness metrics. Test Type: {test_type.replace('_', ' ').title()}",
            summary=summary_text,
            report_filename=session['report_filename'],
            test_type=test_type,
            metadata=metadata
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