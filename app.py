# main/app.py
from flask import Flask, redirect
from datetime import timedelta
from flask_session import Session
from flask_cors import CORS  # ADD FOR WEBSITE INTEGRATION

# Import all domain blueprints
from Justice.fdk_justice import justice_bp	# ADD JUSTICE BLUEPRINT
from Business.fdk_business import business_bp  # ADD BUSINESS BLUEPRINT

app = Flask(__name__)
CORS(app)  # ENABLE CROSS-ORIGIN REQUESTS

# Session configuration (ONCE in main app)
app.secret_key = 'fdk_toolkit_secret_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
Session(app)

# Register all blueprints
app.register_blueprint(justice_bp, url_prefix='/justice')
app.register_blueprint(business_bp, url_prefix='/business')  # ADD BUSINESS REGISTRATION

# Route redirects
@app.route('/justice')
@app.route('/justice/')
def justice_redirect():
    return redirect('/justice/justice-upload')

@app.route('/business')  # ADD BUSINESS REDIRECT
@app.route('/business/')
def business_redirect():
    return redirect('/business/business-upload')

@app.route('/')
def home():
    return "FDK Toolkit - Navigate to /justice or /business for fairness audits"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
