# ================================================================
# FDK Business App - Interactive Fairness Audit for Business Services Domain
# ================================================================

import os
import re
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# CHANGE: Flask app → Blueprint
business_bp = Blueprint('business', __name__, template_folder='templates')

# UNIFIED INTELLIGENT SYSTEM: FDK Import with fallback
# No longer needs try/except -- intelligent_selection.py has no Flask
# dependency and imports no domain blueprint, so there's no circular
# import to guard against.
from intelligent_selection import intelligent_target_selection
HAS_FDK_INTELLIGENT = True

def _is_binary_column(series):
    """True only if the column has exactly two unique values, both in {0, 1}."""
    try:
        unique_vals = series.dropna().unique()
        return len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})
    except Exception:
        return False

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
# Business-Specific Keyword Mappings (Phase 3A)
# ------------------------------------------------
BUSINESS_KEYWORDS = {
    'group': ['customer_type', 'segment', 'cohort', 'region', 'market', 'loyalty_tier',
              'age_group', 'income_bracket', 'geographic', 'product_category', 
              'service_tier', 'marketing_channel', 'customer_segment', 'demographic',
              'category', 'protected_attribute'],
    'y_true': ['conversion', 'purchase', 'churn', 'retention', 'response', 'approval',
               'engagement', 'satisfaction', 'loyalty', 'campaign_success', 
               'service_usage', 'renewal', 'actual', 'outcome', 'target', 'label',
               'ground_truth'],
    'y_pred': ['predicted_conversion', 'churn_risk', 'response_score', 'retention_prediction',
               'engagement_forecast', 'clv_prediction', 'recommendation_score', 
               'personalization_score', 'prediction', 'predicted', 'estimate', 'model_output',
               'forecast', 'model', 'decision', 'classification', 'output',
               'recommended', 'recommendation'],
    'y_prob': ['probability', 'score', 'confidence', 'likelihood', 'propensity',
               'estimate', 'calibration', 'confidence_score', 'rating', 'clv',
               'risk_score', 'clv_score']
}

# ------------------------------------------------
# Unified Business Auto-Detection with FDK Integration
# ------------------------------------------------
def detect_business_column_mappings(df, columns, test_type='pre_implementation', user_target=None):
    """
    Unified column detection with FDK intelligent system integration.
    Priority: User Override > FDK Intelligent > Domain-specific detection
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None, 'timestamp': None}
    reasoning = {}
    intelligent_suggestion = None
    
    # Initialize reasoning for all columns
    for col in columns:
        reasoning[col] = ""
    
    # PRIORITY 1: User Override (if provided)
    if user_target and user_target in df.columns:
        suggestions['y_true'] = user_target
        reasoning[user_target] = "User override: Manually selected target column"
        print(f"🎯 Using user override target: {user_target}")
    
    # PRIORITY 2: FDK Intelligent Selection (if available)
    if HAS_FDK_INTELLIGENT and not suggestions['y_true']:
        try:
            intelligent_result = intelligent_target_selection(
                df=df,
                domain='business',
                test_type=test_type
            )
            
            if intelligent_result and 'suggested_target' in intelligent_result:
                suggested_target = intelligent_result['suggested_target']
                if suggested_target in df.columns:
                    suggestions['y_true'] = suggested_target
                    reasoning[suggested_target] = f"FDK Intelligent: {intelligent_result.get('reasoning', 'AI-suggested target')}"
                    intelligent_suggestion = suggested_target
                    print(f"🤖 FDK intelligent suggestion: {suggested_target}")
        except Exception as e:
            print(f"⚠️ FDK intelligent selection failed: {e}")
    
    # PRIORITY 3: Domain-specific detection (with business keywords)
    # Layer 1: Direct matching for standard column names
    for col in columns:
        if col in [suggestions.get('group'), suggestions.get('y_true'), suggestions.get('y_pred'), suggestions.get('y_prob')]:
            continue
            
        col_lower = col.lower()
        
        # GROUP detection
        if not suggestions['group']:
            if col_lower in ['group', 'segment', 'customer_segment', 'demographic', 'category', 'cohort']:
                suggestions['group'] = col
                reasoning[col] = "Direct match: customer segment/group column"
                continue
            # Business-specific group keywords
            elif any(keyword in col_lower for keyword in BUSINESS_KEYWORDS['group']):
                suggestions['group'] = col
                reasoning[col] = "Business domain: Customer segments for fairness analysis"
                continue
        
        # Y_TRUE detection (skip if already set by user or FDK)
        if not suggestions['y_true']:
            if col_lower in ['y_true', 'actual', 'true', 'outcome', 'target', 'label', 'ground_truth', 'conversion']:
                if _is_binary_column(df[col]):
                    suggestions['y_true'] = col
                    reasoning[col] = "Direct match: true business outcomes/target variable"
                    continue
            # Business-specific outcome keywords
            elif any(keyword in col_lower for keyword in BUSINESS_KEYWORDS['y_true']):
                if _is_binary_column(df[col]):
                    suggestions['y_true'] = col
                    reasoning[col] = "Business domain: Customer outcomes (binary: 0/1)"
                    continue
        
        # Y_PRED detection
        if not suggestions['y_pred']:
            if col_lower in ['y_pred', 'predicted', 'prediction', 'estimate', 'model_output', 'forecast']:
                if _is_binary_column(df[col]):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Direct match: business model predictions"
                    continue
            # Business-specific prediction keywords
            elif any(keyword in col_lower for keyword in BUSINESS_KEYWORDS['y_pred']):
                if _is_binary_column(df[col]):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Business domain: Business algorithm predictions"
                    continue
        
        # Y_PROB detection
        if not suggestions['y_prob']:
            if col_lower in ['y_prob', 'probability', 'score', 'confidence', 'risk_score', 'propensity', 'clv_score']:
                suggestions['y_prob'] = col
                reasoning[col] = "Direct match: probability/confidence scores"
                continue
            # Business-specific probability keywords
            elif any(keyword in col_lower for keyword in BUSINESS_KEYWORDS['y_prob']):
                suggestions['y_prob'] = col
                reasoning[col] = "Business domain: Business probability scores"
                continue

        # TIMESTAMP detection (optional — enables temporal fairness metrics)
        if not suggestions['timestamp']:
            timestamp_keywords = ['timestamp', 'date', 'decision_date', 'time', 'datetime']
            col_tokens = set(re.split(r'[^a-z0-9]+', col_lower))
            keyword_matched = any(set(kw.split('_')).issubset(col_tokens) for kw in timestamp_keywords)
            if keyword_matched:
                try:
                    parsed = pd.to_datetime(df[col], errors='raise')
                    non_null = parsed.dropna()
                    if non_null.empty or non_null.dt.year.min() < 1971:
                        raise ValueError("degenerate epoch-adjacent parse")
                    suggestions['timestamp'] = col
                    reasoning[col] = "Detected as a parseable date/time column for temporal fairness metrics"
                    continue
                except Exception:
                    pass
    
    # Layer 2: Data type and statistical fallbacks
    for col in columns:
        if col in [suggestions.get('group'), suggestions.get('y_true'), suggestions.get('y_pred'), suggestions.get('y_prob')]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP fallback: Categorical columns
        if not suggestions['group']:
            if 1 < col_data.nunique() <= 20:
                suggestions['group'] = col
                reasoning[col] = "Statistical fallback: Customer segments (2-20 unique values)"
                continue
                
        # Y_TRUE fallback: genuinely binary columns only
        if not suggestions['y_true']:
            if col_data.dtype in ['int64', 'float64'] and _is_binary_column(col_data):
                if col != suggestions['y_pred']:
                    suggestions['y_true'] = col
                    reasoning[col] = "Statistical fallback: Binary outcomes (2 unique values)"
                    continue
                    
        # Y_PRED fallback: genuinely binary columns only (different from y_true)
        if not suggestions['y_pred']:
            if (col != suggestions['y_true'] and col_data.dtype in ['int64', 'float64'] 
                and _is_binary_column(col_data)):
                suggestions['y_pred'] = col
                reasoning[col] = "Statistical fallback: Binary predictions (2 unique values)"
                continue
                
        # Y_PROB fallback: Probability range columns
        if not suggestions['y_prob']:
            if col_data.dtype in ['float64', 'float32']:
                if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Statistical fallback: Probability scores (0-1 range)"
                    continue
    
    # Final validation and return
    return suggestions, reasoning, intelligent_suggestion

# ================================================================
# No-Code Dropdown Support (ported from Finance/Health/Hiring reference
# implementation) -- powers the confirmation-page manual-override UX.
# Business's own binary convention is strict {0,1} only, matching the
# existing _is_binary_column helper used elsewhere in this file.
# ================================================================

def _describe_column(df, col, role=None):
    """
    Plain-language, jargon-free description of a candidate column, for
    display next to each dropdown option on the confirmation page.
    `role` matters: for group/target/prediction, "every value is unique"
    means "this is an ID column, not usable." For probability, every
    value being distinct is completely normal.
    """
    series = df[col]
    n_unique = series.nunique()
    n_total = len(series)

    if role == 'y_prob':
        vmin, vmax = series.min(), series.max()
        return f"{col} — continuous values ranging {vmin:.3f} to {vmax:.3f}"

    if n_unique == n_total:
        return f"{col} — every value is unique (likely an ID column, not usable here)"

    if series.dtype == object or n_unique <= 10:
        top_vals = series.dropna().unique()[:5]
        vals_str = ", ".join(str(v) for v in top_vals)
        return f"{col} — {n_unique} categories ({vals_str}{', ...' if n_unique > 5 else ''})"

    return f"{col} — {n_unique} distinct numeric values"


def _candidate_columns(df, role, exclude=None):
    """
    Filters a dataset's columns down to the ones that are statistically
    plausible for a given role.

    - group: low-cardinality (2-10 unique values), not a pure identifier
    - y_true / y_pred: genuinely binary, strict {0,1} only (via the same
      _is_binary_column convention used elsewhere in this file)
    - y_prob: continuous float column in the 0-1 range. Deliberately does
      NOT apply the identifier-exclusion check -- a real probability
      column naturally has every value unique.
    """
    exclude = exclude or set()
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        n_unique = series.nunique()
        n_total = len(series)
        is_identifier = (n_unique == n_total)

        if role == 'group':
            if is_identifier:
                continue
            if 1 < n_unique <= 20:
                candidates.append(col)

        elif role in ('y_true', 'y_pred'):
            if is_identifier:
                continue
            if series.dtype in ['int64', 'float64', 'bool'] and _is_binary_column(series):
                candidates.append(col)

        elif role == 'y_prob':
            if series.dtype in ['float64', 'float32'] and n_unique > 2:
                try:
                    if series.dropna().between(0, 1).all():
                        candidates.append(col)
                except Exception:
                    continue

    return candidates


def build_column_options(df, suggested_mappings):
    """
    Assembles the four per-role candidate lists for the confirmation
    page, each excluding whatever's already suggested for a different
    role, with a plain-language description attached to every option.

    Returns: {role: [{"column": name, "description": str, "selected": bool}, ...]}
    """
    roles = ['group', 'y_true', 'y_pred', 'y_prob']
    options = {}

    for role in roles:
        already_used_elsewhere = {
            v for r, v in suggested_mappings.items()
            if r in roles and r != role and v
        }
        candidates = _candidate_columns(df, role, exclude=already_used_elsewhere)

        suggested = suggested_mappings.get(role)
        if suggested and suggested not in candidates and suggested in df.columns:
            candidates = [suggested] + candidates

        options[role] = [
            {
                "column": col,
                "description": _describe_column(df, col, role=role),
                "selected": (col == suggested),
            }
            for col in candidates
        ]

    return options

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
    """Process business dataset upload with unified parameter system"""
    if 'file' not in request.files:
        return render_template("result_business.html", title="Error", message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_business.html", title="Error", message="Empty filename.", summary=None)

    # UNIFIED PARAMETER READING (Phase 2A, Step 3)
    user_selected_target = request.form.get('target_column', '').strip()
    if not user_selected_target:
        user_selected_target = request.form.get('target_column_fallback', '').strip()
    test_type = request.form.get('test_type', 'pre_implementation')
    
    print(f"📋 Business Audit Parameters: user_target={user_selected_target}, test_type={test_type}")

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_business.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # UNIFIED BUSINESS AUTO-DETECTION with FDK integration
        suggested_mappings, column_reasoning, intelligent_suggestion = detect_business_column_mappings(
            df, columns, test_type=test_type, user_target=user_selected_target
        )
        
        # Sanitization: null out any suggested mapping that isn't actually in
        # its own role's valid-candidate list, rather than trusting the
        # detection heuristics blindly. Same fix pattern as Finance/Health/Hiring.
        for role in ['group', 'y_true', 'y_pred', 'y_prob']:
            candidate_list = _candidate_columns(df, role)
            if suggested_mappings.get(role) and suggested_mappings[role] not in candidate_list:
                print(f"⚠️ Sanitizing invalid '{role}' suggestion: {suggested_mappings[role]} is not a valid candidate for this role")
                suggested_mappings[role] = None
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_business.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        # Store in session with additional metadata
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        session['test_type'] = test_type
        session['user_selected_target'] = user_selected_target
        session['intelligent_suggestion'] = intelligent_suggestion
        
        # Count actual key features detected
        detected_key_features = len([k for k in ('group', 'y_true', 'y_pred', 'y_prob') if suggested_mappings.get(k) is not None])

        # Build the filtered, role-appropriate dropdown options for the
        # confirmation page's manual-override UX.
        column_options = build_column_options(df, suggested_mappings)
        
        return render_template(
            'auto_confirm_business.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            column_options=column_options,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename,
            test_type=test_type,
            intelligent_suggestion=intelligent_suggestion
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

        # Apply any manual overrides submitted from the confirmation page's
        # dropdowns. Each override is re-validated against its role's own
        # candidate list before being accepted.
        override_params = {
            'group': request.args.get('group_col'),
            'y_true': request.args.get('y_true_col'),
            'y_pred': request.args.get('y_pred_col'),
            'y_prob': request.args.get('y_prob_col'),
        }
        for role, override_value in override_params.items():
            if not override_value:
                continue
            valid_candidates = _candidate_columns(df, role)
            if override_value in valid_candidates:
                column_mapping[role] = override_value
            else:
                print(f"⚠️ Ignoring invalid '{role}' override from request: {override_value}")
        
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
        
        # UNIFIED METADATA ADDITION (Phase 2A, Step 4)
        metadata = {
            "target_column_used": column_mapping.get('y_true'),
            "target_column_original": column_mapping.get('y_true'),
            "prediction_column_used": column_mapping.get('y_pred'),
            "group_column_used": column_mapping.get('group'),
            "probability_column_used": column_mapping.get('y_prob'),
            "test_type": session.get('test_type', 'pre_implementation'),
            "intelligent_suggestion": session.get('intelligent_suggestion'),
            "user_override_applied": bool(session.get('user_selected_target') and session.get('user_selected_target') in df.columns),
            "user_selected_target": session.get('user_selected_target') if session.get('user_selected_target') else None,
            "timestamp": datetime.now().isoformat(),
            "dataset_filename": os.path.basename(dataset_path),
            "fdk_version": "business_1.0_unified",
            "column_mapping": column_mapping
        }
        audit_response["metadata"] = metadata
        
        # Save report with metadata
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
            message="Your business dataset was audited successfully using 60 fairness metrics.",
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
