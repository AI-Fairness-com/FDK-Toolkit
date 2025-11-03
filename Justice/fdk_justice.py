# ================================================================
# FDK Justice App - Interactive Fairness Audit for Justice Domain
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# Check if Flask-Session is installed
try:
    from flask_session import Session
except ImportError:
    print("\n❌ ERROR: Flask-Session package not installed!")
    print("💡 SOLUTION: Run this command in terminal:")
    print("   pip install Flask-Session")
    print("Then restart the application: python fdk_justice.py")
    exit(1)

# Import justice pipeline
from fdk_justice_pipeline import interpret_prompt, run_audit_from_request, run_pipeline

# ================================================================
# FOLDER CONFIGURATION
# ================================================================

UPLOAD_FOLDER = 'uploads_justice'
REPORT_FOLDER = 'reports_justice'

# Create justice-specific folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# JUSTICE-SPECIFIC AUTO-DETECTION LOGIC
# ================================================================

def detect_justice_column_mappings(df, columns):
    """
    Auto-detection optimized for justice domain datasets.
    
    Args:
        df: Pandas DataFrame containing justice data
        columns: List of column names in the dataset
        
    Returns:
        tuple: (suggestions_dict, reasoning_dict) containing column mappings and explanations
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {}
    
    # Initialize reasoning for all columns
    for col in columns:
        reasoning[col] = ""
    
    # Justice-specific column detection logic
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Detect defendant/offender demographic groups
        if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.nunique() > 1):
            justice_group_keywords = ['race', 'ethnic', 'gender', 'age_group', 'location', 
                                    'district', 'county', 'socioeconomic']
            if any(keyword in col.lower() for keyword in justice_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Defendant/offender groups for fairness analysis"
                continue
                
        # Y_TRUE COLUMN: Detect actual justice outcomes (binary)
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1}):
                justice_true_keywords = ['recidivism', 'rearrest', 'violation', 'sentencing', 
                                       'bail', 'parole', 'conviction']
                if any(keyword in col.lower() for keyword in justice_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Justice outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED COLUMN: Detect algorithm predictions (binary)
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1}) and col != suggestions['y_true']:
                justice_pred_keywords = ['prediction', 'risk_score', 'algorithm', 'model', 'assessment']
                if any(keyword in col.lower() for keyword in justice_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Justice algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB COLUMN: Detect probability scores (continuous 0-1)
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and col_data.between(0, 1).all():
                prob_keywords = ['probability', 'score', 'risk', 'likelihood']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Risk probability scores (0-1 range)"
                    continue
    
    # FALLBACK DETECTION: If primary detection fails, use fallback logic
    
    # Fallback for group column
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested justice groups (categorical)"
                break
                
    # Fallback for true outcomes column
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                suggestions['y_true'] = col
                reasoning[col] = "Suggested justice outcomes (binary)"
                break
                
    # Fallback for predictions column
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Suggested justice predictions (binary)"
                break
    
    return suggestions, reasoning

# ================================================================
# JUSTICE-SPECIFIC REPORT GENERATION
# ================================================================

def build_justice_summaries(audit: dict) -> list:
    """
    Generate justice-specific human-readable summaries from audit results.
    
    Args:
        audit: Dictionary containing fairness audit results
        
    Returns:
        list: Formatted summary lines for display
    """
    lines = []
    
    # PROFESSIONAL SUMMARY SECTION
    lines.append("=== JUSTICE PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Legal & Justice System Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall Assessment
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    if composite_score is not None:
        lines.append("1) OVERALL FAIRNESS ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.15:
            lines.append("   → SEVERITY: HIGH - Significant fairness concerns in justice decisions")
            lines.append("   → ACTION: IMMEDIATE LEGAL REVIEW REQUIRED")
        elif composite_score > 0.05:
            lines.append("   → SEVERITY: MEDIUM - Moderate fairness concerns detected")
            lines.append("   → ACTION: SCHEDULE SYSTEMATIC REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal fairness concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Justice Metrics Analysis
    fairness_metrics = audit.get("fairness_metrics", {})
    
    # Statistical Parity Difference Analysis
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) DECISION RATE DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in decision rates across groups")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable decision rate variations")
        else:
            lines.append("     ✅ LOW: Consistent decision rates across groups")
        lines.append("")
    
    # False Positive Rate Analysis
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) ERROR DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some groups experience many more false accusations")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false accusations")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates")
        lines.append("")
    
    # Legal Recommendations Based on Findings
    lines.append("4) LEGAL & POLICY RECOMMENDATIONS:")
    if composite_score > 0.15:
        lines.append("   🚨 IMMEDIATE LEGAL ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive bias investigation")
        lines.append("   • Review legal decision-making processes")
        lines.append("   • Implement bias mitigation protocols")
        lines.append("   • Consider external legal audit")
    elif composite_score > 0.05:
        lines.append("   ⚖️  RECOMMENDED LEGAL REVIEW:")
        lines.append("   • Schedule systematic fairness review")
        lines.append("   • Monitor decision patterns by group")
        lines.append("   • Document fairness considerations")
        lines.append("   • Plan procedural improvements")
    else:
        lines.append("   ✅ LEGAL COMPLIANCE MAINTAINED:")
        lines.append("   • Continue regular fairness monitoring")
        lines.append("   • Maintain current legal standards")
        lines.append("   • Document compliance assessment")
    lines.append("")
    
    # PUBLIC INTEREST SUMMARY SECTION
    lines.append("=== PUBLIC INTEREST SUMMARY ===")
    lines.append("Plain-English Interpretation for Transparency:")
    lines.append("")
    
    # Public-facing interpretation
    if composite_score > 0.15:
        lines.append("🔴 SIGNIFICANT FAIRNESS CONCERNS")
        lines.append("")
        lines.append("This justice tool shows substantial differences in how it treats different groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Decisions may be inconsistent across demographic groups")
        lines.append("• Some groups may experience different outcomes")
        lines.append("• Additional review of decision processes is recommended")
    elif composite_score > 0.05:
        lines.append("🟡 MODERATE FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This justice tool generally works fairly but shows some variation across groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing monitoring is recommended")
    else:
        lines.append("🟢 GOOD FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This justice tool demonstrates consistent treatment across all groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Decisions are applied consistently regardless of background")
        lines.append("• The tool meets fairness standards")
        lines.append("• Treatment is equitable across different groups")
    
    lines.append("")
    
    # LEGAL DISCLAIMER SECTION
    lines.append("=== LEGAL DISCLAIMER ===")
    lines.append("This fairness audit complies with:")
    lines.append("• Equal Protection Clause (14th Amendment)")
    lines.append("• Civil Rights Act provisions")
    lines.append("• Algorithmic accountability frameworks")
    lines.append("• Legal professional standards")
    lines.append("")
    lines.append("LEGAL NOTICE: This tool is for fairness assessment only and does not:")
    lines.append("• Provide legal advice or representation")
    lines.append("• Determine legal rights or outcomes")
    lines.append("• Replace professional legal consultation")
    lines.append("")
    lines.append("For legal concerns, consult qualified legal professionals.")
    
    return lines

# ================================================================
# FLASK BLUEPRINT SETUP
# ================================================================

justice_bp = Blueprint('justice', __name__, template_folder='templates')

# Session configuration will be handled by main app
# Secret key will be set by main app

# ================================================================
# JUSTICE ROUTES DEFINITION
# ================================================================

@justice_bp.route('/justice-upload')
def justice_upload_page():
    """Justice dataset upload page - clears previous session"""
    session.clear()
    return render_template('upload_justice.html')

@justice_bp.route('/justice-audit', methods=['POST'])
def start_justice_audit_process():
    """
    Process justice dataset upload and perform auto-detection.
    
    Returns:
        Rendered template with detection results or error message
    """
    if 'file' not in request.files:
        return render_template("result_justice.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_justice.html", title="Error", 
                             message="Empty filename.", summary=None)

    # Save uploaded file to justice uploads folder
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        # Read and validate dataset
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_justice.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # Justice-specific auto-detection
        suggested_mappings, column_reasoning = detect_justice_column_mappings(df, columns)
        
        # Validate required mappings were detected
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_justice.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        # Store detection results in session for audit execution
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        
        # Count actual key features detected (not total columns)
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_justice.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_justice.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@justice_bp.route('/justice-run-audit')
def run_justice_audit_with_mapping():
    """
    Execute justice fairness audit using detected column mappings.
    
    Returns:
        Rendered template with audit results or error message
    """
    # Retrieve session data
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_justice.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings exist
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_justice.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
        # CRITICAL: Create clean DataFrame with standardized column names
        df_mapped = pd.DataFrame()
        
        # Map each detected column to its standard name
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()
        
        # Validate we have the required columns after mapping
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_justice.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Validate each column is a proper Series (not DataFrame)
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_justice.html", title="Error",
                                    message=f"Column '{col}' is not a Series. This should never happen.", summary=None)
        
        # Execute justice fairness audit
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save comprehensive audit report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"justice_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2)
        
        session['report_filename'] = report_filename
        
        # Generate justice-specific human-readable summary
        summary_lines = build_justice_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_justice.html",
            title="Justice Fairness Audit Completed",
            message="Your justice dataset was audited successfully using 18 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Justice audit failed: {str(e)}"
        return render_template("result_justice.html", title="Justice Audit Failed",
                              message=error_msg, summary=None)

@justice_bp.route('/download-justice-report/<filename>')
def download_justice_report(filename):
    """
    Serve justice audit reports for download.
    
    Args:
        filename: Name of the report file to download
        
    Returns:
        File download response or 404 error
    """
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@justice_bp.route('/')
def index():
    """Home page - redirect to justice upload interface"""
    return redirect(url_for('justice.justice_upload_page'))

# ================================================================
# BLUEPRINT EXPORT
# ================================================================

# justice_bp is now ready to be imported by main app.py
