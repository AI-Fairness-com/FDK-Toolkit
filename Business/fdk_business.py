# ================================================================
# FDK Business App - Interactive Fairness Audit for Business Services Domain
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# CHANGE: Flask app → Blueprint
business_bp = Blueprint('business', __name__, template_folder='templates')

# FIX: Import pipeline with relative import
from .fdk_business_pipeline import run_pipeline

# ------------------------------------------------
# Folder Definitions
# ------------------------------------------------
UPLOAD_FOLDER = 'uploads_business'
REPORT_FOLDER = 'reports_business'

# Create business-specific folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ------------------------------------------------
# Business Auto-Detection
# ------------------------------------------------
def detect_business_column_mappings(df, columns):
    """Auto-detection for business datasets"""
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {}
    
    for col in columns:
        reasoning[col] = ""
    
    # Layer 1: Direct matching for standard column names
    for col in columns:
        col_lower = col.lower()
        if col_lower in ['group', 'segment', 'customer_segment', 'demographic', 'category', 'cohort']:
            suggestions['group'] = col
            reasoning[col] = "Direct match: customer segment/group column"
            continue
        elif col_lower in ['y_true', 'actual', 'true', 'outcome', 'target', 'label', 'ground_truth', 'conversion']:
            suggestions['y_true'] = col
            reasoning[col] = "Direct match: true business outcomes/target variable"
            continue
        elif col_lower in ['y_pred', 'predicted', 'prediction', 'estimate', 'model_output', 'forecast']:
            suggestions['y_pred'] = col
            reasoning[col] = "Direct match: business model predictions"
            continue
        elif col_lower in ['y_prob', 'probability', 'score', 'confidence', 'risk_score', 'propensity', 'clv_score']:
            suggestions['y_prob'] = col
            reasoning[col] = "Direct match: probability/confidence scores"
            continue

    # Layer 2: Business-specific keyword detection
    for col in columns:
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP: Business-specific segments
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            business_group_keywords = ['customer_type', 'segment', 'cohort', 'region', 'market', 
                                     'loyalty_tier', 'age_group', 'income_bracket', 'geographic',
                                     'product_category', 'service_tier', 'marketing_channel']
            if any(keyword in col.lower() for keyword in business_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Business domain: Customer segments for fairness analysis"
                continue
                
        # Y_TRUE: Business outcomes
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                business_true_keywords = ['conversion', 'purchase', 'churn', 'retention', 'response',
                                        'approval', 'engagement', 'satisfaction', 'loyalty',
                                        'campaign_success', 'service_usage', 'renewal']
                if any(keyword in col.lower() for keyword in business_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Business domain: Customer outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED: Business predictions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                business_pred_keywords = ['predicted_conversion', 'churn_risk', 'response_score', 
                                        'retention_prediction', 'engagement_forecast', 'clv_prediction',
                                        'recommendation_score', 'personalization_score']
                if any(keyword in col.lower() for keyword in business_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Business domain: Business algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB: Probability scores
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                               'estimate', 'calibration', 'confidence_score', 'rating', 'clv']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Business domain: Business probability scores (0-1 range)"
                    continue
    
    # Layer 3: Statistical fallbacks
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 20:
                suggestions['group'] = col
                reasoning[col] = "Statistical fallback: Customer segments (2-20 unique values)"
                break
        if not suggestions['group']:
            for col in columns:
                if df[col].dtype in ['int64', 'float64'] and 2 <= df[col].nunique() <= 10:
                    suggestions['group'] = col
                    reasoning[col] = "Statistical fallback: Numeric segments (2-10 unique values)"
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
# Business-Specific Human Summary
# ------------------------------------------------
def build_business_summaries(audit: dict) -> list:
    """Business-specific human-readable summary"""
    lines = []
    
    # PROFESSIONAL SUMMARY
    lines.append("=== BUSINESS SERVICES PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Customer Equity & Service Interpretation")
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
        lines.append(f"   → Total Customers Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Customer Segments: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_customers = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Customers Analyzed: {total_customers}")
        lines.append(f"   → Customer Segments: {num_groups}")
        if num_groups <= 10:
            lines.append(f"   → Segment Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Segment: {max(group_counts.values())} customers")
            lines.append(f"   → Smallest Segment: {min(group_counts.values())} customers")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    # Overall Assessment
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    if composite_score is not None:
        lines.append("1) OVERALL CUSTOMER EQUITY ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.10:
            lines.append("   → SEVERITY: HIGH - Significant customer equity concerns in service decisions")
            lines.append("   → ACTION: IMMEDIATE CUSTOMER EQUITY REVIEW REQUIRED")
        elif composite_score > 0.03:
            lines.append("   → SEVERITY: MEDIUM - Moderate customer equity concerns detected")
            lines.append("   → ACTION: SCHEDULE CUSTOMER EXPERIENCE REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal customer equity concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Business Metrics
    fairness_metrics = audit.get("fairness_metrics", {})
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) SERVICE ALLOCATION DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in service allocation across customer segments")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable service allocation variations")
        else:
            lines.append("     ✅ LOW: Consistent service allocation across customer segments")
        lines.append("")
    
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) CUSTOMER ACCESS DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some customer segments experience many more false service denials")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false service denials")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates across customer segments")
        lines.append("")
    
    # Business Recommendations
    lines.append("4) CUSTOMER EQUITY RECOMMENDATIONS:")
    if composite_score and composite_score > 0.10:
        lines.append("   🚨 IMMEDIATE EQUITY ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive customer equity investigation")
        lines.append("   • Review service allocation decision-making processes")
        lines.append("   • Implement customer equity mitigation protocols")
        lines.append("   • Consider external customer experience audit")
    elif composite_score and composite_score > 0.03:
        lines.append("   ⚖️  RECOMMENDED CUSTOMER REVIEW:")
        lines.append("   • Schedule systematic customer equity review")
        lines.append("   • Monitor service allocation patterns by customer segment")
        lines.append("   • Document customer equity considerations")
        lines.append("   • Plan procedural improvements for equity")
    else:
        lines.append("   ✅ CUSTOMER EQUITY STANDARDS MAINTAINED:")
        lines.append("   • Continue regular customer equity monitoring")
        lines.append("   • Maintain current customer equity standards")
        lines.append("   • Document customer equity assessment")
    lines.append("")
    
    # PUBLIC SUMMARY
    lines.append("=== CUSTOMER TRANSPARENCY SUMMARY ===")
    lines.append("Plain-English Interpretation for Customer Trust:")
    lines.append("")
    
    # Check for high individual metrics even if composite score is low
    high_bias_detected = False
    medium_bias_detected = False
    
    # Check specific high-impact metrics
    if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.1:
        high_bias_detected = True
    if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.1:
        high_bias_detected = True
    if 'average_odds_difference' in fairness_metrics and fairness_metrics['average_odds_difference'] > 0.1:
        high_bias_detected = True
    
    # Check for medium bias indicators
    if not high_bias_detected:
        if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.05:
            medium_bias_detected = True
        if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.05:
            medium_bias_detected = True
    
    # Determine public summary based on actual bias levels
    if high_bias_detected or (composite_score and composite_score > 0.10):
        lines.append("🔴 SIGNIFICANT EQUITY CONCERNS")
        lines.append("")
        lines.append("This business tool shows substantial differences in how it treats different customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Service decisions may be inconsistent across customer groups")
        lines.append("• Some segments may experience different service access rates")
        lines.append("• Additional review of business processes is recommended")
    elif medium_bias_detected or (composite_score and composite_score > 0.03):
        lines.append("🟡 MODERATE EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This business tool generally works fairly but shows some variation across customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its business decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing customer equity monitoring is recommended")
    else:
        lines.append("🟢 GOOD EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This business tool demonstrates consistent treatment across all customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Business decisions are applied consistently regardless of customer background")
        lines.append("• The tool meets customer equity standards")
        lines.append("• Treatment is equitable across different customer segments")
    
    lines.append("")
    
    # CUSTOMER EQUITY DISCLAIMER
    lines.append("=== CUSTOMER EQUITY DISCLAIMER ===")
    lines.append("This customer equity audit complies with:")
    lines.append("• Consumer protection laws")
    lines.append("• Fair business practice regulations")
    lines.append("• Anti-discrimination business laws")
    lines.append("• Algorithmic accountability frameworks in business services")
    lines.append("")
    lines.append("BUSINESS NOTICE: This tool is for customer equity assessment only and does not:")
    lines.append("• Provide business guarantees or outcomes")
    lines.append("• Determine customer eligibility")
    lines.append("• Replace professional business consultation")
    lines.append("")
    lines.append("For customer equity concerns, consult qualified business professionals.")
    
    return lines

# ------------------------------------------------
# Business Routes
# ------------------------------------------------

@business_bp.route('/business-upload')
def business_upload_page():
    """Business upload page"""
    session.clear()
    return render_template('upload_business.html')

@business_bp.route('/business-audit', methods=['POST'])
def start_business_audit_process():
    """Process business dataset upload"""
    if 'file' not in request.files:
        return render_template("result_business.html", title="Error", message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_business.html", title="Error", message="Empty filename.", summary=None)

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_business.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # Business auto-detection
        suggested_mappings, column_reasoning = detect_business_column_mappings(df, columns)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_business.html", title="Auto-Detection Failed",
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
            'auto_confirm_business.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename
        )
        
    except Exception as e:
        return render_template("result_business.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@business_bp.route('/business-run-audit')
def run_business_audit_with_mapping():
    """Run business audit with detected mapping"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    
    if not dataset_path or not column_mapping:
        return render_template("result_business.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_business.html", title="Error",
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
            return render_template("result_business.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Validate each column is a proper Series
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_business.html", title="Error",
                                    message=f"Column '{col}' is not a Series.", summary=None)
        
        # Run business audit
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"business_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        # Generate business-specific summary
        summary_lines = build_business_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_business.html",
            title="Business Services Fairness Audit Completed",
            message="Your business dataset was audited successfully using 36 fairness metrics.",
            summary=summary_text,
            report_filename=session['report_filename']
        )
        
    except Exception as e:
        error_msg = f"Business audit failed: {str(e)}"
        return render_template("result_business.html", title="Business Audit Failed",
                              message=error_msg, summary=None)

@business_bp.route('/download-business-report/<filename>')
def download_business_report(filename):
    """Serve business audit reports"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404