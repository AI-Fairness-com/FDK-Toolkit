# ================================================================
# FDK Health - Fairness Audit for Healthcare AI Systems
# ================================================================
# Universal API for healthcare dataset fairness auditing
# Compliant with EU AI Act and medical device regulations
# ================================================================

import os
import json
import pandas as pd
import numpy as np
import traceback
from flask import Flask, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta
from flask_session import Session

from fdk_health_pipeline import run_pipeline

# ================================================================
# Configuration
# ================================================================

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# Flask Application Setup
# ================================================================

app = Flask(__name__)
app.secret_key = 'fdk_health_fairness_2024'

# Session configuration
app.config.update({
    'SESSION_TYPE': 'filesystem',
    'PERMANENT_SESSION_LIFETIME': timedelta(minutes=30),
    'SESSION_FILE_THRESHOLD': 100
})

Session(app)

# ================================================================
# Core Detection Functions
# ================================================================

def detect_column_mappings(df, columns):
    """
    Auto-detect column mappings for healthcare fairness analysis.
    Preserves the carefully tuned detection logic for medical domains.
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {col: "" for col in columns}
    
    # Primary detection by column names and data patterns
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # Group detection: categorical with limited values
        if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.nunique() > 1):
            group_keywords = ['ethnic', 'group', 'gender', 'race', 'age_group', 'location', 'region', 'type', 'category']
            if any(keyword in col.lower() for keyword in group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Patient groups for fairness analysis"
                continue
                
        # True outcomes: binary medical results (0/1)
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1}):
                true_keywords = ['diagnosis', 'outcome', 'result', 'disease', 'positive', 'mortality', 'readmission']
                if any(keyword in col.lower() for keyword in true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Medical outcomes (binary: 0/1)"
                    continue
                    
        # Predictions: binary model outputs (0/1)
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1}) and col != suggestions['y_true']:
                pred_keywords = ['prediction', 'predict', 'model', 'ai', 'recommend', 'classifier']
                if any(keyword in col.lower() for keyword in pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Model predictions (binary: 0/1)"
                    continue
                    
        # Probability scores: continuous values (0-1 range)
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and col_data.between(0, 1).all():
                prob_keywords = ['probabil', 'score', 'confidence', 'risk', 'likelihood']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Probability scores (0-1 range)"
                    continue
    
    # Fallback detection for any missing mappings
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested patient groups (categorical)"
                break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                suggestions['y_true'] = col
                reasoning[col] = "Suggested medical outcomes (binary)"
                break
                
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Suggested model predictions (binary)"
                break
                
    if not suggestions['y_prob']:
        for col in columns:
            if (df[col].dtype in ['float64', 'float32'] and df[col].between(0, 1).all() 
                and df[col].nunique() > 2):
                suggestions['y_prob'] = col
                reasoning[col] = "Suggested probability scores"
                break
    
    return suggestions, reasoning

# ================================================================
# Summary Generation (Preserves Risk Communication Logic)
# ================================================================

def build_human_summaries(audit: dict) -> list:
    """
    Generate consistent professional and patient summaries.
    CRITICAL: Preserves the risk communication consistency between summaries.
    """
    lines = []
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    
    # Professional Summary
    lines.extend(_build_professional_summary(audit, composite_score))
    
    # Patient/Public Summary  
    lines.extend(_build_patient_summary(composite_score))
    
    # Legal Disclaimer
    lines.extend(_build_legal_disclaimer())
    
    return lines

def _build_professional_summary(audit: dict, composite_score: float) -> list:
    """Build professional healthcare summary with risk assessment"""
    lines = [
        "=== PROFESSIONAL SUMMARY ===",
        "FDK Fairness Audit — Healthcare Professional Interpretation",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # Overall Assessment
    if composite_score is not None:
        lines.append("1) OVERALL FAIRNESS ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.3:
            lines.append("   → SEVERITY: HIGH - Significant fairness concerns detected")
            lines.append("   → ACTION: IMMEDIATE REVIEW REQUIRED")
        elif composite_score > 0.1:
            lines.append("   → SEVERITY: MEDIUM - Moderate fairness concerns")
            lines.append("   → ACTION: SCHEDULE REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal fairness concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Performance Gaps Analysis
    lines.extend(_analyze_performance_gaps(audit))
    
    # Calibration Analysis
    lines.extend(_analyze_calibration(audit))
    
    # Worst Group Analysis
    lines.extend(_analyze_worst_group(audit))
    
    # Professional Recommendations
    lines.extend(_generate_recommendations(composite_score))
    
    return lines

def _analyze_performance_gaps(audit: dict) -> list:
    """Analyze performance disparities across groups"""
    lines = []
    perf_gaps = audit.get("performance_gaps", {})
    
    if perf_gaps:
        lines.append("2) KEY PERFORMANCE DISPARITIES:")
        
        # Detection Accuracy (False Negative Rate)
        fnr_gap = perf_gaps.get("FNR", {}).get("range")
        if fnr_gap is not None:
            lines.append(f"   → Missed Cases Gap: {fnr_gap:.3f}")
            if fnr_gap > 0.15:
                lines.append("     🚨 HIGH: Some groups have significantly more missed diagnoses")
            elif fnr_gap > 0.05:
                lines.append("     ⚠️  MEDIUM: Moderate variation in missed cases across groups")
            else:
                lines.append("     ✅ LOW: Consistent detection across groups")
        
        # False Positive Rate
        fpr_gap = perf_gaps.get("FPR", {}).get("range")
        if fpr_gap is not None:
            lines.append(f"   → False Alarm Gap: {fpr_gap:.3f}")
            if fpr_gap > 0.15:
                lines.append("     🚨 HIGH: Some groups experience many more false alarms")
            elif fpr_gap > 0.05:
                lines.append("     ⚠️  MEDIUM: Moderate variation in false alarms")
            else:
                lines.append("     ✅ LOW: Consistent false alarm rates")
        
        # Overall Accuracy
        acc_gap = perf_gaps.get("Accuracy", {}).get("range")
        if acc_gap is not None:
            lines.append(f"   → Accuracy Gap: {acc_gap:.3f}")
            if acc_gap > 0.15:
                lines.append("     🚨 HIGH: Significant accuracy differences between groups")
            elif acc_gap > 0.05:
                lines.append("     ⚠️  MEDIUM: Noticeable accuracy variations")
            else:
                lines.append("     ✅ LOW: Consistent accuracy across groups")
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

def _generate_recommendations(composite_score: float) -> list:
    """Generate professional recommendations based on risk level"""
    lines = ["5) PROFESSIONAL RECOMMENDATIONS:"]
    
    if composite_score > 0.3:
        lines.extend([
            "   🚨 IMMEDIATE ACTIONS REQUIRED:",
            "   • Conduct detailed bias investigation",
            "   • Implement bias mitigation strategies", 
            "   • Increase monitoring of high-risk groups",
            "   • Consider model retraining with balanced data"
        ])
    elif composite_score > 0.1:
        lines.extend([
            "   ⚠️  RECOMMENDED ACTIONS:",
            "   • Schedule systematic bias review",
            "   • Monitor performance by patient group",
            "   • Document fairness considerations",
            "   • Plan for bias mitigation if issues persist"
        ])
    else:
        lines.extend([
            "   ✅ MAINTENANCE ACTIONS:",
            "   • Continue regular fairness monitoring", 
            "   • Maintain current practices",
            "   • Document this positive fairness assessment"
        ])
    lines.append("")
    
    return lines

def _build_patient_summary(composite_score: float) -> list:
    """Build patient-friendly summary with consistent risk communication"""
    lines = [
        "=== PATIENT / PUBLIC-FRIENDLY SUMMARY ===",
        "Plain-English Interpretation for App Users:",
        ""
    ]
    
    if composite_score > 0.3:
        lines.extend([
            "🔴 FAIRNESS CONCERN DETECTED",
            "",
            "This health tool may work differently for different groups of people.",
            "",
            "What this means for you:",
            "• The tool's accuracy may vary depending on your background",
            "• Some results might be less reliable for certain groups", 
            "• We recommend consulting healthcare professionals for important decisions",
            "",
            "Recommended next steps:",
            "• Discuss results with your doctor or healthcare provider",
            "• Use this tool as supplementary information only",
            "• Seek professional medical advice for diagnosis or treatment"
        ])
    elif composite_score > 0.1:
        lines.extend([
            "🟡 MODERATE FAIRNESS ASSESSMENT", 
            "",
            "This health tool generally works well, but shows some variation across different groups.",
            "",
            "What this means for you:",
            "• The tool should work reliably for most users",
            "• Some small differences in performance may exist",
            "• Results are generally trustworthy",
            "",
            "Recommended next steps:", 
            "• You can use this tool with confidence",
            "• For critical health decisions, consult professionals",
            "• Report any concerns about accuracy"
        ])
    else:
        lines.extend([
            "🟢 GOOD FAIRNESS ASSESSMENT",
            "",
            "This health tool works consistently well across all user groups.",
            "", 
            "What this means for you:",
            "• The tool provides reliable results regardless of your background",
            "• You can trust the accuracy and fairness of predictions",
            "• Performance is consistent across different user groups",
            "",
            "Recommended next steps:",
            "• Use this tool with confidence",
            "• Continue following standard health guidelines", 
            "• Always consult professionals for serious concerns"
        ])
    lines.append("")
    
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

@app.route('/')
def index():
    """Home page redirect to health upload"""
    return redirect(url_for('health_upload_page'))

@app.route('/health-upload')
def health_upload_page():
    """Health upload page - clean session start"""
    session.clear()
    return render_template('upload_health.html')

@app.route('/health-audit', methods=['POST'])
def start_health_audit_process():
    """Process uploaded health dataset and auto-detect columns"""
    if 'file' not in request.files:
        return render_template("result_health.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_health.html", title="Error", 
                             message="Empty filename.", summary=None)

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
        
        # Auto-detect column mappings
        suggested_mappings, column_reasoning = detect_column_mappings(df, columns)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if not suggested_mappings.get(m)]
        
        if missing_required:
            return render_template("result_health.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}.", summary=None)
        
        # Store in session and show confirmation
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        
        return render_template(
            'auto_confirm_health.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_health.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@app.route('/health-run-audit')
def run_health_audit_with_mapping():
    """Execute fairness audit with auto-detected mappings"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_health.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping]
        if missing_required:
            return render_template("result_health.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)

        # Create mapped dataframe with duplicate handling
        df_mapped = df.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Remove duplicate columns if they exist
        if len(df_mapped.columns) != len(set(df_mapped.columns)):
            df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated()]
        
        # Run fairness audit pipeline
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save detailed report
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
            message="Your health dataset was audited successfully using auto-detected column mapping.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        print("Audit pipeline error:")
        traceback.print_exc()
        
        return render_template("result_health.html", title="Health Audit Failed",
                              message=f"Health audit failed: {str(e)}", summary=None)

@app.route('/download-health-report/<filename>')
def download_health_report(filename):
    """Serve health audit reports for download"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

# ================================================================
# Application Entry Point
# ================================================================

if __name__ == '__main__':
    print("🚀 Starting FDK Health Fairness Audit Server...")
    print("📊 Healthcare Domain: ENABLED")
    print("🤖 Auto-detection: ENABLED") 
    print("👥 Risk Communication: CONSISTENT")
    print("🌐 Server running at: http://localhost:5001")
    
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)