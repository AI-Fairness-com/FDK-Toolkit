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
# UNIVERSAL INTELLIGENT TARGET SELECTION SYSTEM
# Moved to intelligent_selection.py -- see that file's docstring for why.
# ================================================================
from intelligent_selection import (
    detect_domain_from_columns,
    is_binary_column,
    find_first_binary_column,
    find_bias_corrected_columns,
    general_intelligent_selection,
    intelligent_target_selection,
    _suggest_target_column,
)

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
