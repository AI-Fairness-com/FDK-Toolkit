📦 FDK™ Installation Guide
A complete installation guide for running the Fairness Diagnostic Kit (FDK™) locally for research, teaching, and fairness auditing demonstrations.
This document follows the same structural and design style as the FDK™ repository README.
🌍 Overview
FDK™ provides domain-specific fairness diagnostics for seven high-impact AI application areas.
This guide explains how to install, configure, and run the toolkit on your local machine using Python and Flask.
🖥️ System Requirements
Python Version
FDK™ requires:
Python 3.10.x
The repository includes a .python-version file specifying 3.10.13.
Supported Operating Systems
macOS
Linux
Windows (with Python properly installed)
Required Python Packages
All runtime dependencies are listed in:
requirements.txt
These include:
Flask
Flask-CORS
Flask-Session
NumPy
Pandas
scikit-learn
SciPy
📁 Cloning the Repository
Open a terminal and run:
git clone https://github.com/AI-Fairness-com/FDK-Toolkit.git
cd FDK-Toolkit
This places you inside the project directory.
📦 Installing Dependencies
Install required Python dependencies with:
pip install -r requirements.txt
If you have multiple Python versions installed, use:
python3 -m pip install -r requirements.txt
▶️ Running the Toolkit
To start the FDK™ application locally:
python app.py
Or if Python 3 is required explicitly:
python3 app.py
This launches the Flask service containing all seven domain-specific UIs.
🌐 Accessing Domain Audit Interfaces
Once the server is running, access the following endpoints in any browser:
Domain	URL
Business	/business-upload
Education	/education-upload
Finance	/finance-upload
Health	/health-upload
Hiring	/hiring-upload
Justice	/justice-upload
Governance	/governance-upload
Each endpoint provides:
CSV upload form
Auto-detected mapping review
Fairness audit execution
JSON report download
Human-readable summary
🧾 Dataset Requirements
Your dataset must follow these rules:
Be in CSV format
Contain no personal identifiers (GDPR-safe)
Include:
At least one sensitive group attribute
A ground-truth outcome column (y_true)
A prediction column (y_pred)
Optional probability scores (y_prob)
FDK™ automatically detects these during upload.
☁️ Optional: Deploying on Render
A render.yaml file is provided for single-click deployment.
It defines:
Python version
Install command
Start command
Render.com will automatically build and deploy the application based on this configuration.
🛠️ Troubleshooting
ModuleNotFoundError
Run:
pip install -r requirements.txt
App not starting
Ensure you are inside the main directory:
cd FDK-Toolkit
Python not recognised
Use:
python3 app.py
⚖️ Licence
The FDK™ Toolkit source code is released under:
Apache License 2.0
See LICENSE and NOTICE in the repository root.
📬 Contact
For academic or technical queries:
info@ai-fairness.com
