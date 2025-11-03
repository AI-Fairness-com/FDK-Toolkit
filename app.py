# main/app.py
from flask import Flask, redirect
from Justice.fdk_justice import app as justice_app

app = Flask(__name__)

# Route to Justice domain
@app.route('/justice')
@app.route('/justice/')
def justice_redirect():
    return redirect('/justice-upload')

# Mount Justice app
app.register_blueprint(justice_app, url_prefix='/justice')

@app.route('/')
def home():
    return "FDK Toolkit - Navigate to /justice for Justice fairness audit"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
