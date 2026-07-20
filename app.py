# main/app.py
from flask import Flask, redirect, request, jsonify
from datetime import timedelta
from flask_session import Session
from flask_cors import CORS
import pandas as pd
from io import BytesIO

# Import all domain blueprints
from Justice.fdk_justice import justice_bp
from Business.fdk_business import business_bp
from Education.fdk_education import education_bp
from Finance.fdk_finance import finance_bp
from Health.fdk_health import health_bp
from Hiring.fdk_hiring import hiring_bp
from Governance.fdk_governance import governance_bp  # ✅ ADD GOVERNANCE IMPORT

app = Flask(__name__)
CORS(app)

# Session configuration (ONCE in main app)
app.secret_key = 'fdk_toolkit_secret_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
Session(app)

# Register all blueprints
app.register_blueprint(justice_bp, url_prefix='/justice')
app.register_blueprint(business_bp, url_prefix='/business')
app.register_blueprint(education_bp, url_prefix='/education')
app.register_blueprint(finance_bp, url_prefix='/finance')
app.register_blueprint(health_bp, url_prefix='/health')
app.register_blueprint(hiring_bp, url_prefix='/hiring')
app.register_blueprint(governance_bp, url_prefix='/governance')  # ✅ ADD GOVERNANCE BLUEPRINT

# ================================================================
# UNIVERSAL INTELLIGENT TARGET SELECTION SYSTEM
# ================================================================

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

def detect_domain_from_columns(columns):
    """Auto-detect which of 7 FDK domains based on column name patterns"""
    domain_patterns = {
        'justice': ['recid', 'bail', 'sentenc', 'parole', 'defendant'],
        'health': ['mortality', 'readmission', 'complication', 'patient', 'diagnos'],
        'education': ['admission', 'dropout', 'graduation', 'student'],
        'hiring': ['hired', 'selected', 'offer_accepted', 'callback', 'applicant'],
        'finance': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk', 'loan'],
        'business': ['churn', 'conversion', 'purchase', 'customer'],
        'governance': ['approved', 'granted', 'permitted', 'constituent'],
    }
    col_string = ' '.join(str(c).lower() for c in columns)
    for domain, keywords in domain_patterns.items():
        if any(keyword in col_string for keyword in keywords):
            return domain
    return None

def general_intelligent_selection(df, test_type):
    """General intelligent target selection when no domain rules match"""
    columns = df.columns.tolist()

    if test_type == 'post_implementation':
        bias_corrected = find_bias_corrected_columns(columns, df)
        if bias_corrected:
            return bias_corrected

    target_keywords = ['target', 'outcome', 'label', 'y_true', 'decision', 'result',
                      'callback', 'hired', 'selected', 'approved', 'default',
                      'churn', 'admission', 'recid', 'mortality']

    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in target_keywords):
            if test_type in ['pre_implementation', 'post_implementation']:
                if is_binary_column(df[col]):
                    return col
            else:
                return col

    if test_type in ['pre_implementation', 'post_implementation']:
        semantic_patterns = ['outcome', 'result', 'status', 'decision', 'flag']
        for pattern in semantic_patterns:
            for col in columns:
                col_lower = str(col).lower()
                if pattern in col_lower and is_binary_column(df[col]):
                    return col

        binary_col = find_first_binary_column(columns, df)
        if binary_col:
            return binary_col

    return columns[-1] if columns else None

def intelligent_target_selection(df, test_type, domain_hint=None):
    """Intelligently select target column based on test type and domain"""
    columns = df.columns.tolist()
    domain = domain_hint or detect_domain_from_columns(columns)

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

    if domain in domain_rules:
        priority_list = domain_rules[domain].get(test_type, [])
        for col_pattern in priority_list:
            for actual_col in columns:
                if col_pattern.lower() in actual_col.lower():
                    if test_type in ['pre_implementation', 'post_implementation']:
                        if is_binary_column(df[actual_col]):
                            return actual_col
                    else:
                        return actual_col

        fallback_func = domain_rules[domain].get('fallback')
        if fallback_func:
            fallback_result = fallback_func(columns)
            if fallback_result:
                return fallback_result

    return general_intelligent_selection(df, test_type)

@app.route("/api/detect-columns", methods=["POST"])
def detect_columns():
    """Enhanced column detection with intelligent target suggestion"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file"}), 400

    try:
        raw = f.read()
        df = pd.read_csv(BytesIO(raw))
        columns = df.columns.tolist()

        test_type = request.form.get("test_type", "pre_implementation")
        domain_hint = request.form.get("domain", None)

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

# Route redirects
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

@app.route('/governance')  # ✅ ADD GOVERNANCE REDIRECT
@app.route('/governance/')
def governance_redirect():
    return redirect('/governance/governance-upload')

@app.route('/')
def home():
    return "FDK Toolkit - Navigate to /justice, /business, /education, /finance, /health, /hiring, or /governance for fairness audits"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
