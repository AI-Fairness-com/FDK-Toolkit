# 📙 FDK™ Installation Guide

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

A complete installation guide for running the **Fairness Diagnostic Kit (FDK™)** locally for research, teaching, and fairness auditing demonstrations.  
This document follows the same style and layout as the main repository README.

---

## 🌍 Overview

The **FDK™ Toolkit** provides domain-specific fairness diagnostics for seven key AI application areas.  
This guide explains how to install, configure, and run the toolkit on your local machine using Python and Flask.

---

## 🖥️ System Requirements

### Python Version
FDK™ requires:
```text
Python 3.10.x
The repository includes a .python-version file specifying 3.10.13.

Supported Operating Systems
macOS

Linux

Windows (with Python installed)

Required Python Packages
FDK™ depends on the packages listed in requirements.txt:

Flask

Flask-CORS

Flask-Session

NumPy

Pandas

scikit-learn

SciPy

📁 Cloning the Repository
Open a terminal and run:

bash
git clone https://github.com/AI-Fairness-com/FDK-Toolkit.git
cd FDK-Toolkit
You should now be inside the project directory.

📦 Installing Dependencies
Install all required Python packages using:

bash
pip install -r requirements.txt
If multiple Python versions exist on your system, run:

bash
python3 -m pip install -r requirements.txt
▶️ Running the Toolkit
To start the FDK™ application locally:

bash
python app.py
Or, if necessary:

bash
python3 app.py
The Flask server will start and expose all seven domain-specific UIs.

🌐 Accessing Domain Upload Interfaces
Once the server is running, open a browser and use:

Domain	URL
Business	/business-upload
Education	/education-upload
Finance	/finance-upload
Health	/health-upload
Hiring	/hiring-upload
Justice	/justice-upload
Governance	/governance-upload
Each interface provides:

CSV upload

Automatic column detection

Mapping confirmation

Domain-specific fairness audit

JSON report download

Plain-language summary

🧾 Dataset Requirements
Your dataset must:

Be provided as a CSV file

Contain no personal identifiers (GDPR-compliant)

Include the following columns:

At least one sensitive group attribute

Ground-truth outcome (y_true)

Model prediction (y_pred)

Optional probability scores (y_prob)

FDK™ automatically identifies these features during upload.

☁️ Optional: Deploying on Render
The repository includes a render.yaml file which defines:

Python version

Install command

Start command

Render.com can use this file to build and deploy the app automatically.

🛠️ Troubleshooting
"ModuleNotFoundError: X"

bash
pip install -r requirements.txt
Application does not start

bash
cd FDK-Toolkit
Python not recognised

bash
python3 app.py
⚖️ Licence
The FDK™ Toolkit source code is released under:
Apache License 2.0

See LICENSE and NOTICE in the repository root.

📬 Contact
For academic or technical enquiries:
info@ai-fairness.com
