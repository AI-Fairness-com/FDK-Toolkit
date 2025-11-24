
# ⚖️ Fairness Diagnostic Kit (FDK™)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub Repository](https://img.shields.io/badge/GitHub-AI--Fairness--com%2FFDK--Toolkit-lightgrey?logo=github)](https://github.com/AI-Fairness-com/FDK-Toolkit)

**Open-source toolkit for fairness diagnostics across seven key AI domains.**  
Developed to support the *Fairness Diagnostic Kit (FDK™)* framework described in the book  
**_The Fairness Diagnostic Kit: Tools for Auditing, Education, and Governance of Responsible AI_** (Tavakoli, 2025).

---

## 🌍 Overview

The **FDK™ Toolkit** enables non-technical professionals, regulators, and educators to test and interpret algorithmic fairness **without coding**.  
It offers domain-specific APIs, automatic feature detection, and plain-language audit reports in JSON and human-readable form.

Each domain API can be accessed through the AI Fairness Portal or run locally for demonstration and research.

---

## 🏗️ System Architecture
[ARCHITECTURE_DIAGRAM_PLACEHOLDER - Professional PNG diagram will be inserted here]
Diagram: User Upload (CSV) → Automatic Column Detection → Domain-Specific Pipeline → Fairness Metrics Calculation → JSON & Human-Readable Report

<img width="547" height="436" alt="Screenshot 2025-11-24 at 11 59 54" src="https://github.com/user-attachments/assets/22fa5e32-3f88-417f-ac7e-e9a80a1f882d" />


## 📊 Core Fairness Metrics

The FDK™ Toolkit implements comprehensive fairness metrics across all domains. Below are key metrics consistently applied:

| Metric | Definition | Domain Relevance |
|:--|:--|:--|
| **Statistical Parity Difference** | Difference in selection rates between groups | All domains - Base fairness measure |
| **Disparate Impact Ratio** | Ratio of selection rates between groups | Hiring, Justice - Legal compliance |
| **Equal Opportunity Difference** | Difference in true positive rates between groups | Health, Justice - Error fairness |
| **Equalized Odds** | Both TPR and FPR equality across groups | All domains - Comprehensive fairness |
| **Predictive Parity** | Equality of positive predictive values | Health, Finance - Predictive reliability |
| **False Discovery Rate Difference** | Difference in false discovery rates between groups | Justice, Business - Error distribution |
| **Average Odds Difference** | Average of FPR and FNR differences | All domains - Balanced performance |
| **Treatment Equality** | Ratio of FNR to FPR across groups | Health, Education - Resource allocation |
| **Demographic Parity Ratio** | Ratio of positive outcomes between groups | All domains - Outcome fairness |
| **Predicted Positives per Group** | Count of positive predictions by group | All domains - Impact assessment |

## 🏥 Real-World Use Cases

### Healthcare: Glaucoma Diagnosis AI
**Context**: AI system for early glaucoma detection from retinal images  
**Sensitive Attribute**: Ethnicity, Age, Gender  
**Fairness Risk**: Lower diagnostic accuracy for minority ethnic groups and older patients, potentially causing irreversible blindness through delayed detection  
**FDK Solution**: Tests 45 healthcare-specific metrics including calibration gaps, error rate parity, and subgroup performance to ensure equitable diagnostic accuracy across all demographic groups.

### Justice: Risk Assessment Tools  
**Context**: Algorithm predicting recidivism risk for bail decisions  
**Sensitive Attribute**: Race, Socio-economic status  
**Fairness Risk**: Systematic over-prediction of risk for minority defendants  
**FDK Solution**: Applies 36 justice metrics including statistical parity, false positive rate differences, and causal fairness checks.

### Hiring: Resume Screening AI
**Context**: Automated screening of job applications  
**Sensitive Attribute**: Gender, Age, Education background  
**Fairness Risk**: Bias against female applicants in technical roles or older candidates  
**FDK Solution**: Evaluates 34 hiring metrics including selection rates, individual fairness consistency, and counterfactual fairness.

---

## 🧩 Supported Domains

FDK™ currently supports seven domains, each with its own fairness metrics and pipelines:

| Domain   | Folder        | Description |
|:--|:--|:--|
| Business  | `/Business/`   | Fairness auditing for customer, employee and corporate decision pipelines |
| Education | `/Education/`  | Fairness testing of grading, admissions and educational decision systems |
| Finance   | `/Finance/`    | Auditing for credit scoring, lending and financial inclusion models |
| Health    | `/Health/`     | Fairness assessment of diagnostic and healthcare support models |
| Hiring    | `/Hiring/`     | Detection of demographic and procedural bias in recruitment pipelines |
| Justice   | `/Justice/`    | Evaluation of algorithmic fairness in justice and risk assessment tools |
| Governance| `/Governance/` | Diagnostics for public-sector, policy and governance-related AI systems |

---

## ⚙️ Repository Structure

Each domain folder includes:

- Python pipeline (`fdk_<domain>_pipeline.py`)
- Flask routing / API file (`fdk_<domain>.py`)
- HTML interface templates (`upload_*.html`, `auto_confirm_*.html`, `result_*.html`)
- Example synthetic outputs (JSON reports), aligned with the book

Top-level structure:

```text
FDK-Toolkit/
│
├── docs/                           # Comprehensive documentation
│   ├── installation.md            # Step-by-step installation guide
│   ├── architecture.md            # System architecture details
│   ├── domains.md                 # Domain-specific explanations
│   ├── example_usage.md           # Practical usage examples
│   └── disclaimer.md              # Legal and ethical guidelines
│
├── Business/                      # Business domain API
│   ├── fdk_business_pipeline.py   # Core fairness pipeline
│   ├── fdk_business.py            # Flask API routes
│   ├── upload_business.html       # Web interface
│   ├── auto_confirm_business.html # Column mapping confirmation
│   └── result_business.html       # Results display
│
├── Education/                     # Education domain API
│   ├── fdk_education_pipeline.py
│   ├── fdk_education.py
│   └── [corresponding HTML templates]
│
├── Finance/                       # Finance domain API  
│   ├── fdk_finance_pipeline.py
│   ├── fdk_finance.py
│   └── [corresponding HTML templates]
│
├── Health/                        # Health domain API
│   ├── fdk_health_pipeline.py
│   ├── fdk_health.py
│   └── [corresponding HTML templates]
│
├── Hiring/                        # Hiring domain API
│   ├── fdk_hiring_pipeline.py
│   ├── fdk_hiring.py
│   └── [corresponding HTML templates]
│
├── Justice/                       # Justice domain API
│   ├── fdk_justice_pipeline.py
│   ├── fdk_justice.py
│   └── [corresponding HTML templates]
│
├── Governance/                    # Governance domain API
│   ├── fdk_governance_pipeline.py
│   ├── fdk_governance.py
│   └── [corresponding HTML templates]
│
├── app.py                         # Main Flask application
├── requirements.txt               # Python dependencies
├── render.yaml                    # Deployment configuration
├── .python-version               # Python version specification
├── LICENSE                       # Apache 2.0 License
├── NOTICE                        # Copyright notices
└── README.md                     # Project documentation


🚀 Installation (Local Use)
Requirements

Python 3.10.x

pip (Python package manager)

Install dependencies

bash
pip install -r requirements.txt
Run the Flask application

bash
python app.py
This starts the FDK™ web interface locally.

Domain upload pages (for example):

/business-upload

/education-upload

/finance-upload

/health-upload

/hiring-upload

/justice-upload

/governance-upload

🧠 High-Level Architecture
Conceptual pipeline (common pattern across domains):

text
User Upload (CSV)
        ↓
Automatic Column Detection and Mapping
        ↓
Domain-Specific Fairness Pipeline
        ↓
Fairness Metrics and Composite Indicators
        ↓
Plain-Language Summary and Recommendations
        ↓
Downloadable JSON Audit Report
The underlying fairness metric definitions, taxonomies and domain rationales are documented in the FDK™ book.

🧪 Example Usage (Business Domain)
Open the Business upload page (e.g. /business-upload)

Upload a CSV file with appropriate, de-identified business data

Review automatically detected column mappings:

Group / segment attributes

Outcome labels

Model predictions

Optional probability scores

Confirm the mapping and run the fairness audit

Review:

On-screen human-readable summary in plain language

Downloadable JSON audit report for further analysis or archiving

The same pattern applies to the other six domains, each with its own set of metrics and narrative summary logic.

🧾 Documentation and Demos (Planned Additions)
In response to peer review, the following documentation components are planned:

Jupyter notebook demos for each domain (Business, Education, Finance, Health, Hiring, Justice, Governance)

Example synthetic datasets aligned with the book's scenarios

Extended metric documentation, linking narrative explanations with mathematical definitions

Additional usage examples for research and teaching

These materials will make it easier for researchers and practitioners to reproduce and extend FDK™ results.

🧪 Testing and Validation (Planned)
Planned enhancements include:

Unit tests for core pipeline functions and column-mapping logic

Validation of selected fairness outputs against known benchmark cases

Regression tests to ensure consistency of metrics across versions

These additions will strengthen the empirical robustness and reproducibility of the toolkit.

⚖️ Legal and Ethical Disclaimer
FDK™ is a research and educational toolkit for fairness diagnostics.

It does not provide legal, financial, healthcare or regulatory advice.

It should not be used as the sole basis for any decision affecting individuals or groups.

Users are responsible for ensuring that datasets are appropriately anonymised and compliant with relevant regulations.

Full legal disclaimer is provided via the associated web interface and accompanying documentation.

📄 Licence
Software (FDK™ Toolkit code): Apache License 2.0
See LICENSE and NOTICE in the repository root.

Book and explanatory text: CC BY-NC-SA 4.0
The book The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI remains under a Creative Commons licence suitable for educational and non-commercial use.

📚 Citation and Credits
If you use or reference this toolkit in your research, please cite:

Tavakoli, H. (2025). The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI. London: Apress.

Repository: AI-Fairness-com/FDK-Toolkit
Correspondence: info@ai-fairness.com

📖 Book BibTeX

bibtex
@book{Tavakoli2025FDK,
  author    = {Hamid Tavakoli},
  title     = {The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI},
  year      = {2025},
  publisher = {Apress},
  address   = {London},
  url       = {https://github.com/AI-Fairness-com/FDK-Toolkit}
}
📖 Software BibTeX

bibtex
@software{Tavakoli2025FDKToolkit,
  author  = {Hamid Tavakoli},
  title   = {FDK™ Toolkit: Fairness Diagnostic Kit for Multi-Domain AI Auditing},
  year    = {2025},
  url     = {https://github.com/AI-Fairness-com/FDK-Toolkit},
  version = {v1.0.0}
}
