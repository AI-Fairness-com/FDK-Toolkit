from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/api/justice', methods=['GET', 'POST'])
def justice_audit():
    return jsonify({"message": "Justice API is working on Render!"})

@app.route('/')
def home():
    return "AI Fairness Toolkit API Server"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
