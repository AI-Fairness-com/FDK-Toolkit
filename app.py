# main/app.py
from flask import Flask, redirect
from datetime import timedelta
from flask_session import Session
from flask_cors import CORS

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