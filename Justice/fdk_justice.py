# ================================================================
# FDK Justice App - Interactive Fairness Audit for Justice Domain
# FIXED VERSION WITH UNIFIED INTELLIGENT SYSTEM
# ================================================================

import os
import re
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime, timedelta

# Import justice pipeline
from .fdk_justice_pipeline import interpret_prompt, run_audit_from_request, run_pipeline, JUSTICE_METRICS_CONFIG

# ================================================================
# FOLDER CONFIGURATION
# ================================================================

UPLOAD_FOLDER = 'uploads_justice'
REPORT_FOLDER = 'reports_justice'

# Create justice-specific folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# UNIFIED INTELLIGENT SYSTEM INTEGRATION
# ================================================================

# No longer needs try/except -- intelligent_selection.py has no Flask
# dependency and imports no domain blueprint, so there's no circular
# import to guard against.
from intelligent_selection import intelligent_target_selection
HAS_FDK_INTELLIGENT = True

def detect_justice_column_mappings(df, columns, test_type='pre_implementation', user_target=None):
    """
    Unified column detection with FDK intelligent system integration.
    Priority: User Override > FDK Intelligent > Justice-specific detection
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None, 'timestamp': None}
    reasoning = {}
    intelligent_suggestion = None
    
    for col in columns:
        reasoning[col] = ""
    
    # STEP 1: FDK INTELLIGENT TARGET SELECTION
    if HAS_FDK_INTELLIGENT and test_type in ['pre_implementation', 'post_implementation']:
        try:
            intelligent_suggestion = intelligent_target_selection(df, test_type, 'justice')
            if intelligent_suggestion and intelligent_suggestion in df.columns:
                suggestions['y_true'] = intelligent_suggestion
                reasoning[intelligent_suggestion] = f"✅ FDK INTELLIGENT SELECTION (test_type: {test_type})"
                print(f"🎯 FDK Intelligent suggests: {intelligent_suggestion} for {test_type}")
        except Exception as e:
            print(f"⚠️ FDK intelligent selection failed: {e}")
    
    # STEP 2: USER OVERRIDE (TAKES PRIORITY)
    if user_target and user_target in df.columns:
        suggestions['y_true'] = user_target
        override_source = 'FDK' if intelligent_suggestion else 'auto-detection'
        reasoning[user_target] = f"✅ USER MANUAL SELECTION (overrides {override_source})"
        print(f"🎯 User overrides to: {user_target}")
    
    # STEP 3: JUSTICE-SPECIFIC DETECTION (for group, y_pred, y_prob, and fallback)
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Detect defendant/offender demographic groups
        if not suggestions['group']:
            if 1 < col_data.nunique() <= 20:
                justice_group_keywords = ['race', 'ethnic', 'gender', 'age_group', 'location', 
                                        'district', 'county', 'socioeconomic']
                if any(keyword in col.lower() for keyword in justice_group_keywords):
                    suggestions['group'] = col
                    reasoning[col] = "Defendant/offender groups for fairness analysis"
                    continue
                    
        # Y_TRUE COLUMN: Only if not already set by FDK or user
        if not suggestions['y_true']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}):
                    justice_true_keywords = ['recidivism', 'rearrest', 'violation', 'sentencing', 
                                           'bail', 'parole', 'conviction']
                    if any(keyword in col.lower() for keyword in justice_true_keywords):
                        suggestions['y_true'] = col
                        reasoning[col] = "Justice outcomes (binary: 0/1)"
                        continue
                        
        # Y_PRED COLUMN: Detect algorithm predictions (binary)
        if not suggestions['y_pred']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}) and col != suggestions['y_true']:
                    justice_pred_keywords = ['prediction', 'predicted', 'risk_score', 'algorithm', 'model',
                                              'assessment', 'recommended', 'recommendation']
                    if any(keyword in col.lower() for keyword in justice_pred_keywords):
                        suggestions['y_pred'] = col
                        reasoning[col] = "Justice algorithm predictions (binary: 0/1)"
                        continue
                        
        # Y_PROB COLUMN: Detect probability scores (continuous 0-1)
        if not suggestions['y_prob']:
            if col_data.dtype in ['float64', 'float32']:
                if len(unique_vals) > 2 and col_data.between(0, 1).all():
                    prob_keywords = ['probability', 'score', 'risk', 'likelihood']
                    if any(keyword in col.lower() for keyword in prob_keywords):
                        suggestions['y_prob'] = col
                        reasoning[col] = "Risk probability scores (0-1 range)"
                        continue

        # TIMESTAMP: optional column enabling temporal fairness metrics
        if not suggestions.get('timestamp'):
            timestamp_keywords = ['timestamp', 'date', 'decision_date', 'time', 'datetime']
            col_tokens = set(re.split(r'[^a-z0-9]+', col.lower()))
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
    
    # STEP 4: FALLBACK DETECTION
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested justice groups (categorical)"
                break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                if set(df[col].dropna().unique()).issubset({0, 1}):
                    suggestions['y_true'] = col
                    reasoning[col] = "Suggested justice outcomes (binary)"
                    break
                
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                if set(df[col].dropna().unique()).issubset({0, 1}):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Suggested justice predictions (binary)"
                    break
    
    return suggestions, reasoning, intelligent_suggestion

# ================================================================
# No-Code Dropdown Support (ported from Finance/Health/Hiring/Business/
# Governance reference implementation) -- powers the confirmation-page
# manual-override UX. Justice has no shared _is_binary_column helper
# (this file predates that convention), so the strict {0,1} check is
# inlined directly here, matching Justice's own primary detection logic.
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
    - y_true / y_pred: genuinely binary, strict {0,1} only (matching
      Justice's own primary detection convention)
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
            if series.dtype in ['int64', 'float64', 'bool'] and n_unique == 2:
                unique_vals = set(series.dropna().unique())
                if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}) or unique_vals.issubset({True, False}):
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

# ================================================================
# JUSTICE-SPECIFIC REPORT GENERATION
# ================================================================

def build_justice_summaries(audit: dict) -> list:
    """Generate justice-specific human-readable summaries from audit results."""
    lines = []
    
    # PROFESSIONAL SUMMARY SECTION
    lines.append("=== JUSTICE PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit – Legal & Justice System Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # STANDARDIZED DATASET OVERVIEW SECTION
    lines.append("📊 DATASET OVERVIEW:")
    if "validation" in audit:
        validation_info = audit["validation"]
        lines.append(f"   → Total Cases Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Protected Groups: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_records = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Cases Analyzed: {total_records}")
        lines.append(f"   → Protected Groups: {num_groups}")
        if num_groups <= 10:
            lines.append(f"   → Group Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Group: {max(group_counts.values())} cases")
            lines.append(f"   → Smallest Group: {min(group_counts.values())} cases")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    # Check for errors
    if "error" in audit:
        lines.append("❌ AUDIT ERROR DETECTED:")
        lines.append(f"   → Error: {audit['error']}")
        lines.append("   → The fairness audit could not complete due to technical issues.")
        lines.append("   → Please check your dataset format and try again.")
        lines.append("")
        return lines
    
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
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) DECISION RATE DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in decision rates across groups")
        elif spd > 0.05:
            lines.append("     ⚠️ MEDIUM: Noticeable decision rate variations")
        else:
            lines.append("     ✅ LOW: Consistent decision rates across groups")
        lines.append("")
    
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) ERROR DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some groups experience many more false accusations")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️ MEDIUM: Moderate variation in false accusations")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates")
        lines.append("")
    
    # Legal Recommendations
    lines.append("4) LEGAL & POLICY RECOMMENDATIONS:")
    if composite_score and composite_score > 0.15:
        lines.append("   🚨 IMMEDIATE LEGAL ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive bias investigation")
        lines.append("   • Review legal decision-making processes")
        lines.append("   • Implement bias mitigation protocols")
        lines.append("   • Consider external legal audit")
    elif composite_score and composite_score > 0.05:
        lines.append("   ⚖️ RECOMMENDED LEGAL REVIEW:")
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
    
    # PUBLIC INTEREST SUMMARY
    lines.append("=== PUBLIC INTEREST SUMMARY ===")
    lines.append("Plain-English Interpretation for Transparency:")
    lines.append("")
    
    if composite_score and composite_score > 0.15:
        lines.append("🔴 SIGNIFICANT FAIRNESS CONCERNS")
        lines.append("")
        lines.append("This justice tool shows substantial differences in how it treats different groups.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Decisions may be inconsistent across demographic groups")
        lines.append("• Some groups may experience different outcomes")
        lines.append("• Additional review of decision processes is recommended")
    elif composite_score and composite_score > 0.05:
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
    
    # LEGAL DISCLAIMER
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

# ================================================================
# JUSTICE ROUTES DEFINITION - UPDATED
# ================================================================

@justice_bp.route('/justice-upload')
def justice_upload_page():
    """Justice dataset upload page"""
    session.clear()
    return render_template('upload_justice.html')

@justice_bp.route('/justice-audit', methods=['POST'])
def start_justice_audit_process():
    """
    Process justice dataset upload with unified intelligent system.
    """
    if 'file' not in request.files:
        return render_template("result_justice.html", title="Error", message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_justice.html", title="Error", message="Empty filename.", summary=None)

    # ✅ UNIFIED PARAMETER READING
    user_selected_target = request.form.get('target_column', '').strip()
    if not user_selected_target:
        user_selected_target = request.form.get('target_column_fallback', '').strip()
    
    test_type = request.form.get('test_type', 'pre_implementation')
    
    print(f"🎯 UNIFIED INTELLIGENT SYSTEM: test_type={test_type}, user_target='{user_selected_target}'")

    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return render_template("result_justice.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # ✅ USE UNIFIED DETECTION WITH TEST TYPE AND USER TARGET
        suggested_mappings, column_reasoning, intelligent_suggestion = detect_justice_column_mappings(
            df, columns, test_type, user_selected_target
        )

        # Sanitization: null out any suggested mapping that isn't actually in
        # its own role's valid-candidate list, rather than trusting the
        # detection heuristics blindly. Same fix pattern as every other FDK
        # domain in this project.
        for role in ['group', 'y_true', 'y_pred', 'y_prob']:
            candidate_list = _candidate_columns(df, role)
            if suggested_mappings.get(role) and suggested_mappings[role] not in candidate_list:
                print(f"⚠️ Sanitizing invalid '{role}' suggestion: {suggested_mappings[role]} is not a valid candidate for this role")
                suggested_mappings[role] = None
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_justice.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names.", summary=None)
        
        session.clear()
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        session['test_type'] = test_type
        session['user_selected_target'] = user_selected_target
        session['intelligent_suggestion'] = intelligent_suggestion
        
        detected_key_features = len([k for k in ('group', 'y_true', 'y_pred', 'y_prob') if suggested_mappings.get(k) is not None])

        # Build the filtered, role-appropriate dropdown options for the
        # confirmation page's manual-override UX.
        column_options = build_column_options(df, suggested_mappings)
        
        return render_template(
            'auto_confirm_justice.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            column_options=column_options,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename,
            test_type=test_type,
            intelligent_suggestion=intelligent_suggestion,
            user_selected=user_selected_target if user_selected_target else None
        )
        
    except Exception as e:
        return render_template("result_justice.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@justice_bp.route('/justice-run-audit')
def run_justice_audit_with_mapping():
    """Execute justice fairness audit with complete metadata."""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    test_type = session.get('test_type', 'pre_implementation')
    user_selected_target = session.get('user_selected_target', '')
    intelligent_suggestion = session.get('intelligent_suggestion', None)
    
    if not dataset_path or not column_mapping:
        return render_template("result_justice.html", title="Error", 
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
            return render_template("result_justice.html", title="Error",
                                message=f"Missing required mappings: {missing_required}", summary=None)
        
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
        
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return render_template("result_justice.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return render_template("result_justice.html", title="Error",
                                    message=f"Column '{col}' is not a Series.", summary=None)
        
        audit_response = run_pipeline(df_mapped)
        
        # ✅ COMPLETE METADATA SECTION
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
            "fdk_version": "justice_1.0_unified",
            "column_mapping": column_mapping
        }
        
        audit_response["metadata"] = metadata
        
        # Validation info
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"justice_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2)
        
        session['report_filename'] = report_filename
        
        summary_lines = build_justice_summaries(audit_response)
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_justice.html",
            title="Justice Fairness Audit Completed",
            message=f"Your justice dataset was audited successfully using 36 fairness metrics. Test Type: {test_type.replace('_', ' ').title()}",
            summary=summary_text,
            report_filename=session['report_filename'],
            test_type=test_type,
            metadata=metadata,
            # Computed from the real config length, not hardcoded, so this
            # can't drift out of sync with the pipeline again the way the
            # static "7" in the template did.
            metric_category_count=len(JUSTICE_METRICS_CONFIG)
        )
        
    except Exception as e:
        error_msg = f"Justice audit failed: {str(e)}"
        return render_template("result_justice.html", title="Justice Audit Failed",
                              message=error_msg, summary=None)

@justice_bp.route('/download-justice-report/<filename>')
def download_justice_report(filename):
    """Serve justice audit reports for download."""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@justice_bp.route('/')
def index():
    """Home page - redirect to justice upload interface"""
    return redirect(url_for('justice.justice_upload_page'))
