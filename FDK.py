# main/app.py
from flask import Flask, redirect, request, jsonify
from datetime import timedelta
from flask_session import Session
from flask_cors import CORS
import pandas as pd
from io import BytesIO

# Import all domain blueprints
try:
    from Justice.fdk_justice import justice_bp
    from Business.fdk_business import business_bp
    from Education.fdk_education import education_bp
    from Finance.fdk_finance import finance_bp
    from Health.fdk_health import health_bp
    from Hiring.fdk_hiring import hiring_bp
    from Governance.fdk_governance import governance_bp
    BLUEPRINTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import all blueprints: {e}")
    print("⚠️ Running in API-only mode")
    BLUEPRINTS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Session configuration (ONCE in main app)
app.secret_key = 'fdk_toolkit_secret_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
Session(app)

# Register all blueprints if available
if BLUEPRINTS_AVAILABLE:
    app.register_blueprint(justice_bp, url_prefix='/justice')
    app.register_blueprint(business_bp, url_prefix='/business')
    app.register_blueprint(education_bp, url_prefix='/education')
    app.register_blueprint(finance_bp, url_prefix='/finance')
    app.register_blueprint(health_bp, url_prefix='/health')
    app.register_blueprint(hiring_bp, url_prefix='/hiring')
    app.register_blueprint(governance_bp, url_prefix='/governance')

# ================================================================
# UNIVERSAL INTELLIGENT TARGET SELECTION SYSTEM - FINAL FIXED VERSION
# ================================================================

def detect_domain_from_columns(columns):
    """
    Auto-detect which of 7 FDK domains based on column name patterns
    Returns: 'justice', 'health', 'education', 'hiring', 'finance', 'business', 'governance', or 'general'
    """
    if hasattr(columns, 'tolist'):
        columns = columns.tolist()
    
    column_text = ' '.join([str(col).lower() for col in columns])
    
    # Domain-specific keyword patterns
    domain_patterns = {
        'justice': ['race', 'recid', 'sentencing', 'bail', 'parole', 'violation', 
                   'defendant', 'offender', 'arrest', 'charge', 'conviction'],
        'health': ['diagnosis', 'treatment', 'mortality', 'readmission', 'clinical',
                  'patient', 'hospital', 'medication', 'symptom', 'procedure'],
        'education': ['grade', 'admission', 'dropout', 'attendance', 'course',
                     'student', 'teacher', 'school', 'test', 'graduation'],
        'hiring': ['hire', 'applicant', 'interview', 'resume', 'candidate',
                  'position', 'recruitment', 'selection', 'offer', 'rejection'],
        'finance': ['loan', 'credit', 'default', 'approval', 'financial',
                   'transaction', 'payment', 'interest', 'score', 'risk'],
        'business': ['customer', 'churn', 'conversion', 'purchase', 'revenue',
                    'sales', 'marketing', 'product', 'subscription', 'retention'],
        'governance': ['approval', 'license', 'permit', 'compliance', 'regulation',
                      'policy', 'government', 'public', 'service', 'application']
    }
    
    # Score each domain based on keyword matches
    domain_scores = {}
    for domain, keywords in domain_patterns.items():
        score = sum(1 for keyword in keywords if keyword in column_text)
        if score > 0:
            domain_scores[domain] = score
    
    # Return domain with highest score, or 'general' if no matches
    if domain_scores:
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    return 'general'

def is_binary_column(series):
    """Check if a pandas Series contains binary 0/1 values"""
    try:
        unique_vals = series.dropna().unique()
        return len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})
    except:
        return False

def find_first_binary_column(columns, df):
    """Find first binary (0/1) column in dataset"""
    for col in columns:
        if is_binary_column(df[col]):
            return col
    return None

def find_bias_corrected_columns(columns, df):
    """Find columns that indicate bias-corrected targets"""
    bias_corrected_patterns = [
        'svm_fair_target', 'biasclean_target', 'fair_target', 'corrected_target',
        'debiased_target', 'mitigated_target', 'fairness_corrected', 'post_correction'
    ]
    for pattern in bias_corrected_patterns:
        for col in columns:
            if pattern in col.lower():
                if is_binary_column(df[col]):
                    return col
    return None

def general_intelligent_selection(df, test_type):
    """
    General intelligent target selection when no domain rules match
    FIXED VERSION: Enhanced semantic priority for hiring domain
    """
    columns = df.columns.tolist()
    
    # For post-implementation: first look for bias-corrected columns
    if test_type == 'post_implementation':
        bias_corrected = find_bias_corrected_columns(columns, df)
        if bias_corrected:
            return bias_corrected
    
    # Priority 1: Columns with target/outcome keywords (ENHANCED)
    target_keywords = ['target', 'outcome', 'label', 'y_true', 'decision', 'result',
                      'callback', 'hired', 'selected', 'approved', 'default', 
                      'churn', 'admission', 'recid', 'mortality']  # Domain-specific additions
    
    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in target_keywords):
            if test_type in ['pre_implementation', 'post_implementation']:
                if is_binary_column(df[col]):
                    return col
            else:
                return col
    
    # Priority 2: First binary column (for pre/post tests) - ORIGINAL BUGGY LOGIC
    # FIX: Search for binary columns with semantic meaning first
    if test_type in ['pre_implementation', 'post_implementation']:
        # First try to find binary columns with outcome semantics
        semantic_patterns = ['outcome', 'result', 'status', 'decision', 'flag']
        for pattern in semantic_patterns:
            for col in columns:
                col_lower = str(col).lower()
                if pattern in col_lower and is_binary_column(df[col]):
                    return col
        
        # Fallback to original logic if no semantic binary columns found
        binary_col = find_first_binary_column(columns, df)
        if binary_col:
            return binary_col
    
    # Priority 3: Last column (ultimate fallback)
    return columns[-1] if columns else None

def intelligent_target_selection(df, test_type, domain_hint=None):
    """
    Intelligently select target column based on test type and domain
    test_type: 'pre_implementation' or 'post_implementation'
    domain_hint: Optional domain hint from user interface
    
    FINAL FIX: Enhanced post-implementation detection for all domains
    """
    columns = df.columns.tolist()
    
    # 1. Detect or use provided domain
    domain = domain_hint or detect_domain_from_columns(columns)
    
    # 2. DOMAIN-SPECIFIC RULES - ENHANCED POST-IMPLEMENTATION DETECTION
    domain_rules = {
        'justice': {
            'pre_implementation': ['two_year_recid', 'is_recid', 'recidivism'],
            'post_implementation': ['two_year_recid', 'is_recid', 'recidivism'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'health': {
            'pre_implementation': ['mortality', 'readmission', 'complication'],
            'post_implementation': ['mortality', 'readmission', 'complication'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'education': {
            'pre_implementation': ['admission', 'dropout', 'graduation'],
            'post_implementation': ['admission', 'dropout', 'graduation'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'hiring': {
            'pre_implementation': ['hired', 'selected', 'offer_accepted', 'callback'],
            'post_implementation': ['hired', 'selected', 'offer_accepted', 'callback'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'finance': {
            'pre_implementation': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk'],
            'post_implementation': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'business': {
            'pre_implementation': ['churn', 'conversion', 'purchase'],
            'post_implementation': ['churn', 'conversion', 'purchase'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        },
        'governance': {
            'pre_implementation': ['approved', 'granted', 'permitted'],
            'post_implementation': ['approved', 'granted', 'permitted'],
            'fallback': lambda cols: find_first_binary_column(cols, df)
        }
    }
    
    
    # 3. Apply domain-specific rules if available
    if domain in domain_rules:
        priority_list = domain_rules[domain].get(test_type, [])
        
        # Check each priority column (case-insensitive)
        for col_pattern in priority_list:
            for actual_col in columns:
                if col_pattern.lower() in actual_col.lower():
                    # Verify it's binary if that's required
                    if test_type in ['pre_implementation', 'post_implementation']:
                        if is_binary_column(df[actual_col]):
                            return actual_col
                    else:
                        return actual_col
        
        # Try fallback function if no priority column found
        fallback_func = domain_rules[domain].get('fallback')
        if fallback_func:
            fallback_result = fallback_func(columns)
            if fallback_result:
                return fallback_result
    
    # 4. GENERAL INTELLIGENT SELECTION (fallback) - NOW WITH ENHANCED SEMANTICS
    return general_intelligent_selection(df, test_type)

def _suggest_target_column(columns):
    """
    Conservative heuristic: returns a suggested target column if found, else last column.
    Maintained for backward compatibility.
    """
    if not columns:
        return None

    # Common target names across domains (case-insensitive)
    preferred = [
        "target", "label", "y", "y_true", "outcome", "class", "decision",
        "ground_truth", "gt", "approved", "hired", "hire", "selected", "callback"
    ]
    lower_map = {c.lower(): c for c in columns if isinstance(c, str)}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    # Fallback: last column
    return columns[-1]

# ================================================================
# NEW API ENDPOINT FOR INTELLIGENT SELECTION
# ================================================================

@app.route("/api/intelligent-target", methods=["POST"])
def api_intelligent_target():
    """
    Universal API for intelligent target selection across all 7 domains
    Input: CSV file + test_type + optional domain
    Output: Recommended target column with reasoning
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    test_type = request.form.get("test_type", "pre_implementation")
    domain_hint = request.form.get("domain", None)
    
    try:
        # Read the CSV
        df = pd.read_csv(file)
        
        # Intelligent target selection
        recommended_target = intelligent_target_selection(df, test_type, domain_hint)
        
        # Provide reasoning
        reasoning_map = {
            "pre_implementation": "Selected for baseline fairness assessment of original algorithm",
            "post_implementation": "Selected for bias-corrected model evaluation"
        }
        reasoning = reasoning_map.get(test_type, "Selected for fairness analysis")
        
        # Also get column suggestions for dropdown (backward compatibility)
        column_list = df.columns.tolist()
        universal_suggestion = _suggest_target_column(column_list)
        
        return jsonify({
            "success": True,
            "recommended_target": recommended_target,
            "reasoning": reasoning,
            "domain": detect_domain_from_columns(column_list),
            "all_columns": column_list,
            "universal_suggestion": universal_suggestion,
            "test_type": test_type,
            "intelligent_selection": True,
            "selection_algorithm": "unified_intelligent_system_v3_fixed"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "recommended_target": None
        }), 400

# ================================================================
# ENHANCED COLUMN DETECTION ENDPOINT
# ================================================================

@app.route("/api/detect-columns", methods=["POST"])
def detect_columns():
    """
    Enhanced column detection with intelligent target suggestion
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file"}), 400

    try:
        raw = f.read()
        df = pd.read_csv(BytesIO(raw))
        columns = df.columns.tolist()
        
        # NEW: Get test type and domain hint from request if provided
        test_type = request.form.get("test_type", "pre_implementation")
        domain_hint = request.form.get("domain", None)
        
        # Use intelligent selection with domain hint
        recommended_target = intelligent_target_selection(df, test_type, domain_hint)
        
        return jsonify({
            "columns": columns, 
            "suggested_target": recommended_target,
            "test_type": test_type,
            "domain": domain_hint or detect_domain_from_columns(columns),
            "intelligent_selection": True,
            "selection_source": "FDK_intelligent_system_v3"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================================================================
# ROUTE REDIRECTS
# ================================================================

@app.route('/justice')
@app.route('/justice/')
def justice_redirect():
    return redirect('/justice/justice-upload')

@app.route('/business')
@app.route('/business/')
def business_redirect():
    return redirect('/business/business-upload')

@app.route('/education')
@app.route('/education/')
def education_redirect():
    return redirect('/education/education-upload')

@app.route('/finance')
@app.route('/finance/')
def finance_redirect():
    return redirect('/finance/finance-upload')

@app.route('/health')
@app.route('/health/')
def health_redirect():
    return redirect('/health/health-upload')

@app.route('/hiring')
@app.route('/hiring/')
def hiring_redirect():
    return redirect('/hiring/hiring-upload')

@app.route('/governance')
@app.route('/governance/')
def governance_redirect():
    return redirect('/governance/governance-upload')

@app.route('/')
def home():
    return "FDK Toolkit - Navigate to /justice, /business, /education, /finance, /health, /hiring, or /governance for fairness audits"

if __name__ == '__main__':
    print("=" * 60)
    print("FDK Unified Intelligent System")
    print("Version: 3.1 - Enhanced Semantic Priority Fix")
    print("=" * 60)
    
    if BLUEPRINTS_AVAILABLE:
        print("✅ All domain blueprints loaded successfully")
    else:
        print("⚠️ Running in API-only mode (some features limited)")
    
    print("\nAvailable endpoints:")
    print("  • /api/intelligent-target - Intelligent target selection")
    print("  • /api/detect-columns - Column detection")
    print("  • /justice - Justice fairness audit")
    print("  • /business - Business fairness audit")
    print("  • ... and other domains")
    print("\nStarting server on http://localhost:5009")
    print("=" * 60)
    
    app.run(debug=True, port=5009)