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
# Moved to intelligent_selection.py -- this file now imports the same
# shared logic FDK.py and every domain blueprint use, instead of
# maintaining its own separate copy (which had already drifted: this
# file's version had picked up the prediction-indicator-term exclusion
# and safer no-blind-fallback behavior that FDK.py's copy never got --
# intelligent_selection.py now uses this file's safer logic as the
# single source of truth for everyone).
# ================================================================
from intelligent_selection import (
    is_binary_column,
    find_first_binary_column,
    find_bias_corrected_columns,
    detect_domain_from_columns,
    general_intelligent_selection,
    intelligent_target_selection,
)

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
