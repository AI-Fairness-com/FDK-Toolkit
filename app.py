# main/app.py
from flask import Flask, redirect
from Justice.fdk_justice import justice_bp  # CHANGED: import justice_bp instead of app

app = Flask(__name__)

# Configure session (moved from Justice app)
app.secret_key = 'fdk_toolkit_secret_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Import and configure Flask-Session
from flask_session import Session
Session(app)

# Route to Justice domain
@app.route('/justice')
@app.route('/justice/')
def justice_redirect():
    return redirect('/justice/justice-upload')  # CHANGED: full blueprint path

# Mount Justice blueprint
app.register_blueprint(justice_bp, url_prefix='/justice')  # CHANGED: using justice_bp

@app.route('/')
def home():
    return "FDK Toolkit - Navigate to /justice for Justice fairness audit"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
