# ================================================================
# FDK Hiring - Fairness Audit for Hiring & Recruitment
# ================================================================
# Interactive fairness audit for hiring, recruitment, and selection AI systems
# Compliant with EEOC, Equal Employment Opportunity, and hiring regulations
# UPDATED: Unified Intelligent System Integration (Fixed to match justice standard)
# COMPARATIVE STUDY: Enhanced with pre-post consistency enforcement
# ================================================================

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, session, redirect, url_for, send_from_directory
from datetime import datetime

# ================================================================
# INTELLIGENT SELECTION IMPORT (UNIFIED SYSTEM WITH COMPARATIVE SUPPORT)
# ================================================================

try:
    from FDK import intelligent_target_selection, register_comparative_study, get_comparative_target
    HAS_FDK_INTELLIGENT = True
    HAS_COMPARATIVE_SUPPORT = True
except ImportError:
    HAS_FDK_INTELLIGENT = False
    HAS_COMPARATIVE_SUPPORT = False
    print(f"⚠️ FDK intelligent selection not available, using fallback detection")

# ================================================================
# Configuration
# ================================================================

UPLOAD_FOLDER = 'uploads_hiring'
REPORT_FOLDER = 'reports_hiring'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ================================================================
# Hiring Blueprint
# ================================================================

hiring_bp = Blueprint('hiring', __name__, template_folder='templates')

# ================================================================
# Pipeline Import
# ================================================================

from .fdk_hiring_pipeline import run_pipeline

# ================================================================
# UNIFIED HIRING COLUMN DETECTION WITH INTELLIGENT SYSTEM
# ================================================================

def detect_hiring_column_mappings(df, columns, test_type='pre_implementation', user_target=None, session_id=None):
    """
    ENHANCED: Unified column detection with FDK intelligent system integration.
    NOW WITH COMPARATIVE STUDY SUPPORT via session_id parameter
    
    Args:
        df: Pandas DataFrame containing hiring data
        columns: List of column names in the dataset
        test_type: Type of test ('pre_implementation' or 'post_implementation')
        user_target: User-specified target column (optional override)
        session_id: Session ID for comparative study consistency (NEW)
        
    Returns:
        tuple: (suggestions_dict, reasoning_dict, intelligent_suggestion) containing column mappings, explanations, and intelligent suggestion metadata
    """
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None, 'timestamp': None}
    reasoning = {}
    intelligent_suggestion = None
    
    for col in columns:
        reasoning[col] = ""
    
    # DEBUG: Log comparative study info
    print(f"🔍 [HIRING] COMPARATIVE DEBUG: test_type={test_type}, session_id={session_id}")
    
    # STEP 1: FDK INTELLIGENT TARGET SELECTION WITH COMPARATIVE SUPPORT
    if HAS_FDK_INTELLIGENT and test_type in ['pre_implementation', 'post_implementation']:
        try:
            # CRITICAL FIX: Pass session_id to enforce comparative consistency
            intelligent_suggestion = intelligent_target_selection(df, test_type, 'hiring', session_id)
            print(f"🔍 [HIRING] COMPARATIVE RESULT: FDK returned '{intelligent_suggestion}' for session_id={session_id}")
            
            if intelligent_suggestion and intelligent_suggestion in df.columns:
                suggestions['y_true'] = intelligent_suggestion
                
                # Special reasoning for comparative studies
                if session_id and test_type == 'post_implementation':
                    # Check if this was enforced by comparative study
                    comparative_target = None
                    if HAS_COMPARATIVE_SUPPORT:
                        comparative_target = get_comparative_target(session_id)
                    
                    if comparative_target and comparative_target == intelligent_suggestion:
                        reasoning[intelligent_suggestion] = f"✅ COMPARATIVE STUDY ENFORCED: Using pre-test target '{intelligent_suggestion}' for valid comparison"
                    else:
                        reasoning[intelligent_suggestion] = f"✅ FDK INTELLIGENT SELECTION (test_type: {test_type})"
                else:
                    reasoning[intelligent_suggestion] = f"✅ FDK INTELLIGENT SELECTION (test_type: {test_type})"
                    
                print(f"🎯 FDK Intelligent suggests: {intelligent_suggestion} for {test_type}")
        except Exception as e:
            print(f"⚠️ FDK intelligent selection failed: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 2: USER OVERRIDE (TAKES PRIORITY) (Matches justice and finance standard)
    if user_target and user_target in df.columns:
        suggestions['y_true'] = user_target
        override_source = 'FDK' if intelligent_suggestion else 'auto-detection'
        reasoning[user_target] = f"✅ USER MANUAL SELECTION (overrides {override_source})"
        print(f"🎯 User overrides to: {user_target}")
    
    # STEP 3: HIRING-SPECIFIC DETECTION (for group, y_pred, y_prob, and fallback)
    hiring_keywords = {
        'group': ['department', 'education', 'experience', 'location', 'gender', 
                 'ethnicity', 'race', 'disability', 'veteran', 'age_group', 'major',
                 'background', 'demographic', 'category', 'segment', 'team', 'role',
                 'protected_attribute', 'applicant_group'],
        'y_true': ['hired', 'selected', 'promoted', 'interview', 'offer', 
                  'screened', 'advanced', 'recommended', 'passed', 'success',
                  'accepted', 'rejected', 'approved', 'denied', 'outcome',
                  'decision', 'final_decision', 'hire_status'],
        'y_pred': ['prediction', 'score', 'assessment', 'algorithm', 
                  'recommendation', 'ranking', 'screening_score', 'model',
                  'decision', 'classification', 'output', 'predicted',
                  'model_score', 'algorithm_score', 'predicted_outcome'],
        'y_prob': ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                  'estimate', 'calibration', 'confidence_score', 'rating',
                  'risk_score', 'propensity_score', 'selection_probability']
    }
    
    for col in columns:
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP COLUMN: Detect applicant/demographic groups for hiring fairness
        if not suggestions['group']:
            if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.nunique() > 1):
                if any(keyword in col.lower() for keyword in hiring_keywords['group']):
                    suggestions['group'] = col
                    reasoning[col] = "Applicant/demographic groups for hiring fairness analysis"
                    continue
                    
        # Y_TRUE COLUMN: Only if not already set by FDK or user
        if not suggestions['y_true']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}):
                    if any(keyword in col.lower() for keyword in hiring_keywords['y_true']):
                        suggestions['y_true'] = col
                        reasoning[col] = "Hiring outcomes (binary: 0/1)"
                        continue
                        
        # Y_PRED COLUMN: Detect algorithm predictions (binary)
        if not suggestions['y_pred']:
            if col_data.dtype in ['int64', 'float64'] and len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1}) and col != suggestions['y_true']:
                    if any(keyword in col.lower() for keyword in hiring_keywords['y_pred']):
                        suggestions['y_pred'] = col
                        reasoning[col] = "Hiring algorithm predictions (binary: 0/1)"
                        continue
                        
        # Y_PROB COLUMN: Detect probability scores (continuous 0-1)
        if not suggestions['y_prob']:
            if col_data.dtype in ['float64', 'float32']:
                if len(unique_vals) > 2 and col_data.between(0, 1).all():
                    if any(keyword in col.lower() for keyword in hiring_keywords['y_prob']):
                        suggestions['y_prob'] = col
                        reasoning[col] = "Selection probability scores (0-1 range)"
                        continue

        # TIMESTAMP: optional column enabling temporal fairness metrics
        if not suggestions.get('timestamp'):
            if any(keyword in col.lower() for keyword in ['timestamp', 'date', 'decision_date', 'time', 'datetime']):
                try:
                    pd.to_datetime(df[col], errors='raise')
                    suggestions['timestamp'] = col
                    reasoning[col] = "Detected as a parseable date/time column for temporal fairness metrics"
                    continue
                except Exception:
                    pass
    
    # STEP 4: FALLBACK DETECTION (Matches justice and finance standard structure)
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 10:
                suggestions['group'] = col
                reasoning[col] = "Suggested applicant groups (categorical)"
                break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                suggestions['y_true'] = col
                reasoning[col] = "Suggested hiring outcomes (binary)"
                break
                
    if not suggestions['y_pred']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                if col != suggestions['y_true']:
                    suggestions['y_pred'] = col
                    reasoning[col] = "Suggested hiring predictions (binary)"
                    break
    
    return suggestions, reasoning, intelligent_suggestion

def build_hiring_summaries(audit_response):
    """Generate hiring-specific summaries from audit response"""
    summary_lines = []
    
    # Overall assessment
    if 'overall_assessment' in audit_response:
        overall = audit_response['overall_assessment']
        summary_lines.append(f"<strong>Overall Fairness Score:</strong> {overall.get('fairness_score', 'N/A')}/100")
        summary_lines.append(f"<strong>Risk Level:</strong> {overall.get('risk_level', 'Unknown')}")
        summary_lines.append(f"<strong>Recommendation:</strong> {overall.get('recommendation', 'No recommendation available')}")
    
    # Group-level metrics
    if 'group_metrics' in audit_response:
        summary_lines.append("<br><strong>Group Performance:</strong>")
        for group, metrics in audit_response['group_metrics'].items():
            if isinstance(metrics, dict):
                tpr = metrics.get('true_positive_rate', 'N/A')
                summary_lines.append(f"  • {group}: Selection Rate = {tpr}")
    
    # Fairness violations
    if 'violations' in audit_response and audit_response['violations']:
        summary_lines.append(f"<br><strong>⚠️ Fairness Concerns Detected:</strong> {len(audit_response['violations'])} issues")
        for violation in audit_response['violations'][:3]:  # Show top 3
            summary_lines.append(f"  • {violation.get('metric', 'Unknown')}: {violation.get('description', 'No description')}")
    
    # Key recommendations
    if 'recommendations' in audit_response and audit_response['recommendations']:
        summary_lines.append("<br><strong>Key Recommendations:</strong>")
        for rec in audit_response['recommendations'][:3]:  # Show top 3
            summary_lines.append(f"  • {rec}")
    
    return summary_lines

# ================================================================
# Flask Routes (Updated with Comparative Study Support)
# ================================================================

@hiring_bp.route('/hiring-upload', methods=['GET', 'POST'])
def hiring_upload_page():
    """
    Hiring upload page and processing handler
    NOW WITH COMPARATIVE STUDY REGISTRATION - FIXED VERSION
    """
    if request.method == 'GET':
        # Display the upload form - PRESERVE comparative study data
        comparative_keys = ['comparative_study_id', 'pre_target_column']
        preserved_data = {}
        for key in comparative_keys:
            if key in session:
                preserved_data[key] = session[key]
        
        # Clear only non-comparative session data
        for key in list(session.keys()):
            if key not in comparative_keys:
                session.pop(key, None)
        
        # Restore comparative data
        for key, value in preserved_data.items():
            session[key] = value
            
        return render_template('upload_hiring.html')
    
    # POST: Process the uploaded file
    if 'file' not in request.files:
        return render_template("result_hiring.html", title="Error", 
                             message="No file uploaded.", summary=None)

    file = request.files['file']
    if file.filename == '':
        return render_template("result_hiring.html", title="Error", 
                             message="Empty filename.", summary=None)

    # ✅ STANDARD UNIFIED PARAMETER READING (Matches justice and finance standard)
    user_selected_target = request.form.get('target_column', '').strip()
    if not user_selected_target:
        user_selected_target = request.form.get('target_column_fallback', '').strip()
    test_type = request.form.get('test_type', 'pre_implementation')
    
    print(f"🎯 [HIRING] UNIFIED INTELLIGENT SYSTEM: test_type={test_type}, user_target='{user_selected_target}'")

    # Save uploaded file
    dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(dataset_path)
    
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        # Validate dataset structure
        if len(columns) < 3:
            return render_template("result_hiring.html", title="Error", 
                                message="Dataset too small. Need at least 3 columns.", summary=None)
        
        # ===== CRITICAL FIX: REGISTER COMPARATIVE STUDY FOR PRE-TESTS =====
        comparative_study_id = None
        pre_target_column = None
        
        if test_type == 'pre_implementation':
            # Generate unique comparative study ID
            comparative_study_id = f"hiring_comparative_{datetime.now().timestamp()}"
            
            # FIXED: Use proper detection WITH session_id for consistency
            # Even though it's pre-test, pass session_id=None for proper detection
            suggested_mappings, _, _ = detect_hiring_column_mappings(
                df, columns, test_type, user_selected_target, session_id=None
            )
            pre_target_column = suggested_mappings.get('y_true')
            
            if pre_target_column:
                print(f"📊 [HIRING] PRE-TEST TARGET DETECTED: {pre_target_column}")
                
                # Register with universal comparative study system
                try:
                    if HAS_COMPARATIVE_SUPPORT:
                        register_comparative_study(comparative_study_id, pre_target_column, 'hiring')
                        print(f"📊 [HIRING] COMPARATIVE STUDY REGISTERED: {comparative_study_id} -> {pre_target_column}")
                    else:
                        # Fallback: Direct API call if function not available
                        import requests
                        # Use request.host_url to get current server URL dynamically
                        base_url = request.host_url.rstrip('/')
                        api_url = f"{base_url}/api/register-comparative-study"
                        requests.post(api_url, 
                                    json={
                                        'session_id': comparative_study_id,
                                        'pre_target_column': pre_target_column,
                                        'domain': 'hiring'
                                    }, timeout=2)
                        print(f"📊 [HIRING] COMPARATIVE STUDY REGISTERED VIA API: {comparative_study_id}")
                except Exception as reg_error:
                    print(f"⚠️ [HIRING] Comparative study registration failed (non-critical): {reg_error}")
            else:
                print(f"⚠️ [HIRING] Could not detect target column for comparative study registration")
        
        # ✅ ENHANCED DETECTION with session_id support
        # For post-tests, check if we have a comparative study ID from session
        if test_type == 'post_implementation':
            comparative_study_id = session.get('comparative_study_id')
            if comparative_study_id:
                print(f"🔍 [HIRING] POST-TEST: Found comparative_study_id: {comparative_study_id}")
                # Verify it's registered in FDK
                if HAS_COMPARATIVE_SUPPORT:
                    registered_target = get_comparative_target(comparative_study_id)
                    if registered_target:
                        print(f"🔍 [HIRING] POST-TEST: Registered target is '{registered_target}'")
        
        # Get comparative_study_id from session for post-tests or use newly created for pre-tests
        if not comparative_study_id and test_type == 'post_implementation':
            comparative_study_id = session.get('comparative_study_id')
        
        print(f"🔍 [HIRING] FINAL comparative_study_id for detection: {comparative_study_id}")
        
        # Perform detection WITH session_id
        suggested_mappings, column_reasoning, intelligent_suggestion = detect_hiring_column_mappings(
            df, columns, test_type, user_selected_target,
            session_id=comparative_study_id  # PASS session_id
        )
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in suggested_mappings or not suggested_mappings[m]]
        
        if missing_required:
            return render_template("result_hiring.html", title="Auto-Detection Failed",
                                message=f"Could not automatically detect: {missing_required}.", summary=None)
        
        # ===== CRITICAL FIX: PRESERVE COMPARATIVE STUDY DATA =====
        # Save comparative data before any session clearing
        preserved_comparative_id = comparative_study_id or session.get('comparative_study_id')
        preserved_pre_target = pre_target_column or session.get('pre_target_column')
        
        # Clear most session data but preserve comparative study info
        keys_to_preserve = ['comparative_study_id', 'pre_target_column']
        preserved_data = {}
        for key in keys_to_preserve:
            if key in session:
                preserved_data[key] = session[key]
        
        # Update with new values if we have them
        if preserved_comparative_id:
            preserved_data['comparative_study_id'] = preserved_comparative_id
        if preserved_pre_target:
            preserved_data['pre_target_column'] = preserved_pre_target
        
        # Clear non-comparative session data
        for key in list(session.keys()):
            if key not in keys_to_preserve:
                session.pop(key, None)
        
        # Set new session values
        session['dataset_path'] = dataset_path
        session['dataset_columns'] = columns
        session['column_mapping'] = suggested_mappings
        session['column_reasoning'] = column_reasoning
        session['test_type'] = test_type
        session['user_selected_target'] = user_selected_target
        session['intelligent_suggestion'] = intelligent_suggestion
        
        # Restore comparative data
        for key, value in preserved_data.items():
            session[key] = value
        
        # Count detected key features
        detected_key_features = len([m for m in suggested_mappings.values() if m is not None])
        
        # Check if comparative study is active
        is_comparative_active = bool(session.get('comparative_study_id'))
        
        # Show comparative status in UI
        comparative_status = ""
        if is_comparative_active:
            if test_type == 'pre_implementation':
                comparative_status = f"📊 Comparative Study Registered: Target '{preserved_pre_target}' locked for post-test"
            else:
                reg_target = None
                if HAS_COMPARATIVE_SUPPORT:
                    reg_target = get_comparative_target(session.get('comparative_study_id'))
                if reg_target:
                    comparative_status = f"📊 Comparative Study Mode: Using target '{reg_target}' from pre-test"
                else:
                    comparative_status = "📊 Comparative Study Mode: Target consistency enforced"
        
        return render_template(
            'auto_confirm_hiring.html',
            suggested_mappings=suggested_mappings,
            column_reasoning=column_reasoning,
            total_columns=len(columns),
            detected_key_features=detected_key_features,
            filename=file.filename,
            test_type=test_type,
            intelligent_suggestion=intelligent_suggestion,
            user_selected=user_selected_target if user_selected_target else None,
            comparative_study_id=session.get('comparative_study_id'),
            is_comparative=is_comparative_active,
            comparative_status=comparative_status
        )
        
    except Exception as e:
        return render_template("result_hiring.html", title="Error", 
                              message=f"Error reading dataset: {str(e)}", summary=None)

@hiring_bp.route('/hiring-audit', methods=['POST'])
def start_hiring_audit_process():
    """
    Alternative POST endpoint for hiring audit (for backward compatibility)
    Redirects to the main hiring-upload handler
    """
    return hiring_upload_page()

@hiring_bp.route('/hiring-run-audit')
def run_hiring_audit_with_mapping():
    """Execute hiring fairness audit with unified metadata integration"""
    dataset_path = session.get('dataset_path')
    column_mapping = session.get('column_mapping', {})
    test_type = session.get('test_type', 'pre_implementation')
    user_selected_target = session.get('user_selected_target', '')
    intelligent_suggestion = session.get('intelligent_suggestion', None)
    comparative_study_id = session.get('comparative_study_id', None)
    pre_target_column = session.get('pre_target_column', None)
    
    if not dataset_path or not column_mapping:
        return render_template("result_hiring.html", title="Error", 
                              message="Missing dataset or column mapping.", summary=None)
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validate required mappings
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mapping or not column_mapping[m]]
        if missing_required:
            return render_template("result_hiring.html", title="Error",
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
            return render_template("result_hiring.html", title="Error",
                                message=f"After mapping, missing columns: {missing_cols}", summary=None)
        
        # Run hiring audit pipeline
        audit_response = run_pipeline(df_mapped, save_to_disk=False)
        
        # ✅ ENHANCED METADATA WITH COMPARATIVE STUDY INFO
        # Check if comparative study target was actually used
        comparative_target_used = False
        if comparative_study_id and HAS_COMPARATIVE_SUPPORT:
            registered_target = get_comparative_target(comparative_study_id)
            actual_target = column_mapping.get('y_true')
            comparative_target_used = (registered_target == actual_target) if registered_target and actual_target else False
        
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
            "comparative_study_id": comparative_study_id,
            "comparative_study_mode": bool(comparative_study_id),
            "comparative_target_used": comparative_target_used,
            "pre_target_column_registered": pre_target_column,
            "timestamp": datetime.now().isoformat(),
            "dataset_filename": os.path.basename(dataset_path),
            "fdk_version": "hiring_1.2_comparative_fixed",
            "column_mapping": column_mapping
        }
        audit_response["metadata"] = metadata
        
        # Add validation info with test_type
        if "validation" not in audit_response:
            group_counts = df_mapped['group'].value_counts().to_dict()
            audit_response["validation"] = {
                "sample_size": len(df_mapped),
                "groups_analyzed": len(df_mapped['group'].unique()),
                "statistical_power": "strong" if len(df_mapped) >= 1000 else "adequate" if len(df_mapped) >= 500 else "moderate",
                "group_counts": group_counts,
                "test_type": test_type,
                "comparative_study": bool(comparative_study_id),
                "comparative_target_consistent": comparative_target_used
            }
        else:
            audit_response["validation"]["test_type"] = test_type
            audit_response["validation"]["comparative_study"] = bool(comparative_study_id)
            audit_response["validation"]["comparative_target_consistent"] = comparative_target_used
        
        # Save detailed report with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"hiring_audit_report_{timestamp}.json"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        session['report_filename'] = report_filename
        
        # Generate hiring-specific summary
        summary_lines = build_hiring_summaries(audit_response)
        
        # Add comparative study note if applicable
        if comparative_study_id:
            if comparative_target_used:
                summary_lines.append(f"<br><strong>📊 Comparative Study:</strong> Target consistency enforced ✓")
                summary_lines.append(f"<em>Using same target '{pre_target_column}' as pre-test for valid comparison</em>")
            else:
                summary_lines.append(f"<br><strong>⚠️ Comparative Study Warning:</strong> Target changed from pre-test")
                if pre_target_column:
                    summary_lines.append(f"<em>Pre-test used '{pre_target_column}', post-test uses '{column_mapping.get('y_true')}'</em>")
        
        summary_text = "<br>".join(summary_lines)
        
        return render_template(
            "result_hiring.html",
            title="Hiring Fairness Audit Completed",
            message=f"Your hiring dataset was audited successfully using 34 fairness metrics. Test Type: {test_type.replace('_', ' ').title()}",
            summary=summary_text,
            report_filename=session['report_filename'],
            test_type=test_type,
            metadata=metadata,
            comparative_study_id=comparative_study_id,
            comparative_target_used=comparative_target_used
        )
        
    except Exception as e:
        error_msg = f"Hiring audit failed: {str(e)}"
        return render_template("result_hiring.html", title="Hiring Audit Failed",
                              message=error_msg, summary=None)

@hiring_bp.route('/download-hiring-report/<filename>')
def download_hiring_report(filename):
    """Serve hiring audit reports for download"""
    try:
        return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404