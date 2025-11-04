# ================================================================
# FDK Education App - Interactive Fairness Audit for Education Domain
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# CHANGE: Flask app → Blueprint
education_bp = Blueprint('education', __name__, template_folder='templates')

# FIX: Import pipeline with relative import
from .fdk_education_pipeline import run_pipeline

# ------------------------------------------------
# Folder Definitions
# ------------------------------------------------
UPLOAD_FOLDER = 'uploads_education'
REPORT_FOLDER = 'reports_education'

# Create education-specific folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ------------------------------------------------
# Education Auto-Detection
# ------------------------------------------------
def detect_education_column_mappings(df, columns):
    """Auto-detection for education datasets"""
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

    # Layer 2: Education-specific keyword detection
    for col in columns:
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP: Education-specific groups
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            education_group_keywords = ['school', 'district', 'demographic', 'ethnicity', 'gender', 
                                      'socioeconomic', 'disability', 'ELL', 'special_ed', 'cohort',
                                      'background', 'category', 'segment', 'program', 'track']
            if any(keyword in col.lower() for keyword in education_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Education domain: Student groups for fairness analysis"
                continue
                
        # Y_TRUE: Education outcomes
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                education_true_keywords = ['admitted', 'placed', 'graduated', 'retained', 'promoted',
                                         'completed', 'advanced', 'remediated', 'certified',
                                         'success', 'passed', 'achieved', 'qualified']
                if any(keyword in col.lower() for keyword in education_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Education domain: Educational outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED: Education predictions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                education_pred_keywords = ['prediction', 'score', 'assessment', 'algorithm', 
                                         'recommendation', 'placement_score', 'admission_prob',
                                         'model', 'decision', 'classification', 'output']
                if any(keyword in col.lower() for keyword in education_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Education domain: Educational algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB: Probability scores
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                               'estimate', 'calibration', 'confidence_score', 'rating']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Education domain: Educational probability scores (0-1 range)"
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

# ------------------------------------------------
# Education-Specific Human Summary
# ------------------------------------------------
def build_education_summaries(audit: dict) -> list:
    """Education-specific human-readable summary"""
    lines = []
    
    # PROFESSIONAL SUMMARY
    lines.append("=== EDUCATION PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Educational Equity & Access Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Check for errors
    if "error" in audit:
        lines.append("❌ AUDIT ERROR DETECTED:")
        lines.append(f"   → Error: {audit['error']}")
        lines.append("   → The fairness audit could not complete due to technical issues.")
        lines.append("   → Please check your dataset format and try again.")
        lines.append("")
        return lines
    
    # DATASET OVERVIEW - STANDARDIZED ACROSS ALL DOMAINS
    lines.append("📊 DATASET OVERVIEW:")
    if "validation" in audit:
        validation_info = audit["validation"]
        lines.append(f"   → Total Students Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Student Groups: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_students = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Students Analyzed: {total_students}")
        lines.append(f"   → Student Groups: {num_groups}")
        # Show group distribution for small number of groups
        if num_groups <= 10:
            lines.append(f"   → Group Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Group: {max(group_counts.values())} students")
            lines.append(f"   → Smallest Group: {min(group_counts.values())} students")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    # Overall Assessment - FIXED: Use composite_fairness_score from education pipeline
    composite_score = audit.get("summary", {}).get("composite_fairness_score")
    if composite_score is None:
        # Fallback for compatibility
        composite_score = audit.get("summary", {}).get("composite_bias_score")
    
    if composite_score is not None:
        lines.append("1) OVERALL EQUITY ASSESSMENT:")
        lines.append(f"   → Composite Fairness Score: {composite_score:.3f}")
        if composite_score < 0.70:  # FIXED: Lower score = worse fairness
            lines.append("   → SEVERITY: HIGH - Significant equity concerns in educational decisions")
            lines.append("   → ACTION: IMMEDIATE EDUCATIONAL EQUITY REVIEW REQUIRED")
        elif composite_score < 0.85:
            lines.append("   → SEVERITY: MEDIUM - Moderate equity concerns detected")
            lines.append("   → ACTION: SCHEDULE EDUCATIONAL REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal equity concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Education Metrics
    fairness_metrics = audit.get("fairness_metrics", {})
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) ADMISSION/PLACEMENT DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if abs(spd) > 0.1:
            lines.append("     🚨 HIGH: Significant differences in admission/placement rates across student groups")
        elif abs(spd) > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable admission/placement rate variations")
        else:
            lines.append("     ✅ LOW: Consistent admission/placement rates across student groups")
        lines.append("")
    
    if 'equal_opportunity_difference' in fairness_metrics:
        eod = fairness_metrics['equal_opportunity_difference']
        lines.append("3) EDUCATIONAL OPPORTUNITY DISPARITIES:")
        lines.append(f"   → Equal Opportunity Difference: {eod:.3f}")
        if abs(eod) > 0.1:
            lines.append("     🚨 HIGH: Some student groups experience many more false rejections")
        elif abs(eod) > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false rejections")
        else:
            lines.append("     ✅ LOW: Consistent opportunity rates across student groups")
        lines.append("")
    
    # Education Recommendations - FIXED: Use composite_score consistently
    lines.append("4) EDUCATIONAL EQUITY RECOMMENDATIONS:")
    if composite_score and composite_score < 0.70:
        lines.append("   🚨 IMMEDIATE EQUITY ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive educational equity investigation")
        lines.append("   • Review admission/placement decision-making processes")
        lines.append("   • Implement educational equity mitigation protocols")
        lines.append("   • Consider external educational equity audit")
    elif composite_score and composite_score < 0.85:
        lines.append("   ⚖️  RECOMMENDED EDUCATIONAL REVIEW:")
        lines.append("   • Schedule systematic educational equity review")
        lines.append("   • Monitor admission/placement patterns by student group")
        lines.append("   • Document educational equity considerations")
        lines.append("   • Plan procedural improvements for equity")
    else:
        lines.append("   ✅ EDUCATIONAL EQUITY STANDARDS MAINTAINED:")
        lines.append("   • Continue regular educational equity monitoring")
        lines.append("   • Maintain current educational equity standards")
        lines.append("   • Document educational equity assessment")
    lines.append("")
    
    # PUBLIC SUMMARY - FIXED: Use composite_score consistently
    lines.append("=== PUBLIC INTEREST SUMMARY ===")
    lines.append("Plain-English Interpretation for Educational Transparency:")
    lines.append("")
    
    if composite_score and composite_score < 0.70:
        lines.append("🔴 SIGNIFICANT EQUITY CONCERNS")
        lines.append("")
        lines.append("This educational tool shows substantial differences in how it treats different student groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Admission/placement decisions may be inconsistent across student groups")
        lines.append("• Some groups may experience different admission/placement rates")
        lines.append("• Additional review of educational processes is recommended")
    elif composite_score and composite_score < 0.85:
        lines.append("🟡 MODERATE EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This educational tool generally works fairly but shows some variation across student groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its educational decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing educational equity monitoring is recommended")
    else:
        lines.append("🟢 GOOD EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This educational tool demonstrates consistent treatment across all student groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Educational decisions are applied consistently regardless of background")
        lines.append("• The tool meets educational equity standards")
        lines.append("• Treatment is equitable across different student groups")
    
    lines.append("")
    
    # EDUCATIONAL EQUITY DISCLAIMER
    lines.append("=== EDUCATIONAL EQUITY DISCLAIMER ===")
    lines.append("This educational equity audit complies with:")
    lines.append("• Equal Educational Opportunity laws")
    lines.append("• Educational equity regulations")
    lines.append("• Anti-discrimination educational laws")
    lines.append("• Algorithmic accountability frameworks in education")
    lines.append("")
    lines.append("EDUCATIONAL NOTICE: This tool is for educational equity assessment only and does not:")
    lines.append("• Provide educational guarantees or outcomes")
    lines.append("• Determine educational eligibility")
    lines.append("• Replace professional educational consultation")
    lines.append("")
    lines.append("For educational equity concerns, consult qualified educational professionals.")
    
    return lines

# ------------------------------------------------
# Education Routes
# ------------------------------------------------

@education_bp.route('/education-upload')
def education_upload_page():
    """Education upload page"""
    session.clear()
    return render_template('upload_education.html')

@education_bp.route('/education-audit', methods=['POST'])
def start_education_audit_process():
    """Process education dataset upload"""
    if 'file' not in request.files:
        return render_template("result_education.html", title="Error", message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_education.html", title="Error", message="Empty filename.", summary=None)

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_education.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # Education auto-detection
        suggested_mappings, column_reasoning = detect_education_column_mappings(df, columns)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_education.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        # Store in session
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        
        # Count actual key features detected
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        return render_template(
            'auto_confirm_education.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_education.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@education_bp.route('/education-run-audit')
def run_education_audit_with_mapping():
    """Run education audit with detected mapping"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_education.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_education.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
        # Create clean DataFrame with mapped columns
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
        
        # Validate required columns
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_education.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Validate each column is a proper Series
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_education.html", title="Error",
                                    message=f"Column '{col}' is not a Series.", summary=None)
        
        # Run education audit
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"education_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        # Generate education-specific summary
        summary_lines = build_education_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_education.html",
            title="Education Fairness Audit Completed",
            message="Your education dataset was audited successfully using 15 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Education audit failed: {str(e)}"
        return render_template("result_education.html", title="Education Audit Failed",
                              message=error_msg, summary=None)

@education_bp.route('/download-education-report/<filename>')
def download_education_report(filename):
    """Serve education audit reports"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404