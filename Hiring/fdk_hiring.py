# ================================================================
# FDK Hiring App - Interactive Fairness Audit for Hiring Domain
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# ================================================================
# DEPENDENCY CHECK: Flask-Session Validation
# ================================================================

try:
    from flask_session import Session
except ImportError:
    print("\n❌ ERROR: Flask-Session package not installed!")
    print("💡 SOLUTION: Run this command in terminal:")
    print("   pip install Flask-Session")
    print("Then restart the application: python fdk_hiring.py")
    exit(1)

# Import hiring pipeline
from fdk_hiring_pipeline import interpret_prompt, run_audit_from_request, run_pipeline

# ================================================================
# FOLDER CONFIGURATION
# ================================================================

UPLOAD_FOLDER = 'uploads_hiring'
REPORT_FOLDER = 'reports_hiring'

# Create hiring-specific folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# UNIVERSAL HIRING AUTO-DETECTION SYSTEM
# ================================================================

def detect_hiring_column_mappings(df, columns):
    """
    Universal auto-detection for hiring datasets.
    
    Handles both real hiring data and synthetic/test datasets through
    three-layer detection: direct matching, hiring keywords, and statistical fallbacks.
    
    Args:
        df: Pandas DataFrame containing hiring data
        columns: List of column names in the dataset
        
    Returns:
        tuple: (suggestions_dict, reasoning_dict) containing column mappings and explanations
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {}
    
    # Initialize reasoning for all columns
    for col in columns:
        reasoning[col] = ""
    
    # LAYER 1: Direct matching for standard/generic column names (synthetic/test datasets)
    for col in columns:
        col_lower = col.lower()
        
        # Direct matching for standard column names
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

    # LAYER 2: Hiring-specific keyword detection (real hiring datasets)
    for col in columns:
        # Skip columns already matched in Layer 1
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Hiring-specific demographic and professional groups
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            hiring_group_keywords = ['department', 'education', 'experience', 'location', 'gender', 
                                   'ethnicity', 'race', 'disability', 'veteran', 'age_group', 'major',
                                   'background', 'demographic', 'category', 'segment']
            if any(keyword in col.lower() for keyword in hiring_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Hiring domain: Applicant groups for fairness analysis"
                continue
                
        # Y_TRUE COLUMN: Hiring outcomes and selection decisions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                hiring_true_keywords = ['hired', 'selected', 'promoted', 'interview', 'offer', 
                                      'screened', 'advanced', 'recommended', 'passed', 'success',
                                      'accepted', 'rejected', 'approved', 'denied']
                if any(keyword in col.lower() for keyword in hiring_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Hiring domain: Hiring outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED COLUMN: Algorithm predictions and screening decisions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                hiring_pred_keywords = ['prediction', 'score', 'assessment', 'algorithm', 
                                      'recommendation', 'ranking', 'screening_score', 'model',
                                      'decision', 'classification', 'output']
                if any(keyword in col.lower() for keyword in hiring_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Hiring domain: Hiring algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB COLUMN: Probability scores and confidence values
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                               'estimate', 'calibration', 'confidence_score', 'rating']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Hiring domain: Selection probability scores (0-1 range)"
                    continue
    
    # LAYER 3: Statistical fallbacks for any remaining unmapped columns
    if not suggestions['group']:
        # Try categorical columns first
        for col in columns:
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 20:
                suggestions['group'] = col
                reasoning[col] = "Statistical fallback: Categorical groups (2-20 unique values)"
                break
        # If no categorical, try numeric with few unique values
        if not suggestions['group']:
            for col in columns:
                if df[col].dtype in ['int64', 'float64'] and 2 <= df[col].nunique() <= 10:
                    suggestions['group'] = col
                    reasoning[col] = "Statistical fallback: Numeric groups (2-10 unique values)"
                    break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                if col != suggestions['y_pred']:  # Avoid using same column for both true and pred
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
# HIRING-SPECIFIC REPORT GENERATION
# ================================================================

def build_hiring_summaries(audit: dict) -> list:
    """
    Generate hiring-specific human-readable summaries from audit results.
    
    Includes professional HR compliance language and public-facing explanations
    suitable for hiring managers and compliance officers.
    
    Args:
        audit: Dictionary containing fairness audit results
        
    Returns:
        list: Formatted summary lines for display
    """
    lines = []
    
    # PROFESSIONAL SUMMARY SECTION
    lines.append("=== HIRING PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Recruitment & Selection Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Check if this is an ERROR result
    if "error" in audit:
        lines.append("❌ AUDIT ERROR DETECTED:")
        lines.append(f"   → Error: {audit['error']}")
        lines.append("   → The fairness audit could not complete due to technical issues.")
        lines.append("   → Please check your dataset format and try again.")
        lines.append("")
        return lines
    
    # Overall Assessment (only for successful audits)
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    if composite_score is not None:
        lines.append("1) OVERALL FAIRNESS ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.10:
            lines.append("   → SEVERITY: HIGH - Significant fairness concerns in hiring decisions")
            lines.append("   → ACTION: IMMEDIATE COMPLIANCE REVIEW REQUIRED")
        elif composite_score > 0.03:
            lines.append("   → SEVERITY: MEDIUM - Moderate fairness concerns detected")
            lines.append("   → ACTION: SCHEDULE HR REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal fairness concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Hiring Metrics Analysis (only if they exist)
    fairness_metrics = audit.get("fairness_metrics", {})
    
    # Statistical Parity Difference Analysis
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) SELECTION RATE DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in selection rates across groups")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable selection rate variations")
        else:
            lines.append("     ✅ LOW: Consistent selection rates across groups")
        lines.append("")
    
    # False Positive Rate Analysis
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) SCREENING ERROR DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some groups experience many more false rejections")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false rejections")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates")
        lines.append("")
    
    # HR Compliance Recommendations Based on Findings
    lines.append("4) HR & COMPLIANCE RECOMMENDATIONS:")
    if composite_score and composite_score > 0.10:
        lines.append("   🚨 IMMEDIATE COMPLIANCE ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive bias investigation")
        lines.append("   • Review hiring decision-making processes")
        lines.append("   • Implement bias mitigation protocols")
        lines.append("   • Consider external compliance audit")
    elif composite_score and composite_score > 0.03:
        lines.append("   ⚖️  RECOMMENDED HR REVIEW:")
        lines.append("   • Schedule systematic fairness review")
        lines.append("   • Monitor selection patterns by applicant group")
        lines.append("   • Document fairness considerations")
        lines.append("   • Plan procedural improvements")
    else:
        lines.append("   ✅ COMPLIANCE STANDARDS MAINTAINED:")
        lines.append("   • Continue regular fairness monitoring")
        lines.append("   • Maintain current HR compliance standards")
        lines.append("   • Document compliance assessment")
    lines.append("")
    
    # PUBLIC INTEREST SUMMARY SECTION
    lines.append("=== PUBLIC INTEREST SUMMARY ===")
    lines.append("Plain-English Interpretation for Transparency:")
    lines.append("")
    
    # Public-facing interpretation for hiring transparency
    if composite_score and composite_score > 0.10:
        lines.append("🔴 SIGNIFICANT FAIRNESS CONCERNS")
        lines.append("")
        lines.append("This hiring tool shows substantial differences in how it treats different applicant groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Selection decisions may be inconsistent across demographic groups")
        lines.append("• Some groups may experience different selection rates")
        lines.append("• Additional review of hiring processes is recommended")
    elif composite_score and composite_score > 0.03:
        lines.append("🟡 MODERATE FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This hiring tool generally works fairly but shows some variation across applicant groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing monitoring is recommended")
    else:
        lines.append("🟢 GOOD FAIRNESS ASSESSMENT")
        lines.append("")
        lines.append("This hiring tool demonstrates consistent treatment across all applicant groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Decisions are applied consistently regardless of background")
        lines.append("• The tool meets fairness standards")
        lines.append("• Treatment is equitable across different groups")
    
    lines.append("")
    
    # COMPLIANCE DISCLAIMER SECTION
    lines.append("=== COMPLIANCE DISCLAIMER ===")
    lines.append("This fairness audit complies with:")
    lines.append("• Equal Employment Opportunity (EEO) laws")
    lines.append("• Fair hiring regulations")
    lines.append("• Anti-discrimination employment laws")
    lines.append("• Algorithmic accountability frameworks")
    lines.append("")
    lines.append("COMPLIANCE NOTICE: This tool is for fairness assessment only and does not:")
    lines.append("• Provide hiring guarantees or outcomes")
    lines.append("• Determine employment eligibility")
    lines.append("• Replace professional HR consultation")
    lines.append("")
    lines.append("For compliance concerns, consult qualified HR professionals.")
    
    return lines

# ================================================================
# FLASK APPLICATION SETUP
# ================================================================

app = Flask(__name__)
app.secret_key = 'hiring_fairness_audit_2024'

# Session configuration for persistence
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_FILE_THRESHOLD'] = 100

Session(app)

# ================================================================
# HIRING ROUTES DEFINITION
# ================================================================

@app.route('/hiring-upload')
def hiring_upload_page():
    """Hiring dataset upload page - clears previous session"""
    session.clear()
    return render_template('upload_hiring.html')

@app.route('/hiring-audit', methods=['POST'])
def start_hiring_audit_process():
    """
    Process hiring dataset upload and perform universal auto-detection.
    
    Returns:
        Rendered template with detection results or error message
    """
    if 'file' not in request.files:
        return render_template("result_hiring.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_hiring.html", title="Error", 
                             message="Empty filename.", summary=None)

    # Save uploaded file to hiring uploads folder
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        # Read and validate dataset
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_hiring.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # Universal hiring auto-detection with three-layer approach
        suggested_mappings, column_reasoning = detect_hiring_column_mappings(df, columns)
        
        # Validate required mappings were detected
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_hiring.html", title="Auto-Detection Failed",
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
            'auto_confirm_hiring.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_hiring.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@app.route('/hiring-run-audit')
def run_hiring_audit_with_mapping():
    """
    Execute hiring fairness audit using detected column mappings.
    
    Returns:
        Rendered template with audit results or error message
    """
    # Retrieve session data
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_hiring.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings exist
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_hiring.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
        # CRITICAL: Create clean DataFrame with standardized column names
        df_mapped = pd.DataFrame()
        
        # Map each detected column to its standard name
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()
        
        # CRITICAL FIX: Convert all numpy types to Python native types for JSON serialization
        for col in df_mapped.columns:
            if df_mapped[col].dtype == 'bool':
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_integer_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_float_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(float)
        
        # Validate we have the required columns after mapping
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_hiring.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Validate each column is a proper Series (not DataFrame)
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_hiring.html", title="Error",
                                    message=f"Column '{col}' is not a Series. This should never happen.", summary=None)
        
        # Execute hiring fairness audit
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save comprehensive audit report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"hiring_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        # Generate hiring-specific human-readable summary
        summary_lines = build_hiring_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_hiring.html",
            title="Hiring Fairness Audit Completed",
            message="Your hiring dataset was audited successfully using 15 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Hiring audit failed: {str(e)}"
        return render_template("result_hiring.html", title="Hiring Audit Failed",
                              message=error_msg, summary=None)

@app.route('/download-hiring-report/<filename>')
def download_hiring_report(filename):
    """
    Serve hiring audit reports for download.
    
    Args:
        filename: Name of the report file to download
        
    Returns:
        File download response or 404 error
    """
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/')
def index():
    """Home page - redirect to hiring upload interface"""
    return redirect(url_for('hiring_upload_page'))

# ================================================================
# APPLICATION STARTUP
# ================================================================

if __name__ == '__main__':
    print("👔 Starting Hiring FairDiagApp...")
    print("📊 Session persistence: ENABLED")
    print("🤖 UNIVERSAL Auto-detection: ENABLED (3-layer detection)")
    print("🔧 Full Type Conversion: ENABLED (bool/int64/float64 → Python native)")
    print("👥 HR & Compliance Summaries: ENABLED")
    print("🎯 15 Hiring Fairness Metrics: ENABLED")
    print("📁 Separate Hiring Folders: ENABLED")
    print("🌐 Hiring Server running at: http://localhost:5004")
    port = int(os.environ.get("PORT", 5004))
    app.run(host='0.0.0.0', port=port, debug=True)