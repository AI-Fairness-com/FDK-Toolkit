# ⚖️ Fairness Diagnostic Kit (FDK™)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub Repository](https://img.shields.io/badge/GitHub-AI--Fairness--com%2FFDK--Toolkit-lightgrey?logo=github)](https://github.com/AI-Fairness-com/FDK-Toolkit)
![Version](https://img.shields.io/badge/version-v1.1.0-blue)

**Open-source toolkit for fairness diagnostics across seven key AI domains.**  
Developed to support the *Fairness Diagnostic Kit (FDK™)* framework described in the book  
**_The Fairness Diagnostic Kit: Tools for Auditing, Education, and Governance of Responsible AI_** (Tavakoli, 2025).

---
**Open-source toolkit for fairness diagnostics across seven key AI domains.**  
Developed to support the *Fairness Diagnostic Kit (FDK™)* framework described in the book  
**_The Fairness Diagnostic Kit: Tools for Auditing, Education, and Governance of Responsible AI_** (Tavakoli, 2025).

---

## 🌍 Overview

The **FDK™ Toolkit** enables non-technical professionals, regulators, and educators to test and interpret algorithmic fairness **without coding**.  
It offers domain-specific APIs, automatic feature detection, and plain-language audit reports in JSON and human-readable form.

Each domain API can be accessed through the AI Fairness Portal or run locally for demonstration and research.

---
## 🎯 Universal Intelligent Target Selection
New in v1.1.0: FDK now features an intelligent target selection system that automatically detects the most appropriate target column based on:

Domain detection (justice, health, education, hiring, finance, business, governance)

Test type (pre-implementation vs. post-implementation)

Column patterns and domain-specific keywords

Binary column verification for fairness metrics

***Key Capabilities:***

**🎯 Auto-detects domain from dataset column names**

**⚙️ Applies domain-specific rules for different test types**

**🤖 Provides intelligent suggestions with reasoning**

**🔄 Maintains backward compatibility with manual selection**

**🌐 Works across all 7 fairness domains**

### Usage:
text
**API endpoint for intelligent selection**
curl -X POST -F "file=@dataset.csv" \
     -F "test_type=post_implementation" \
     http://localhost:5009/api/intelligent-target

**UI Features:**
✅ Pre/Post implementation test type selection

🔍 Real-time column analysis

🎯 Automatic target column suggestions

💡 Context-aware help and explanations

This system simplifies the fairness audit process while ensuring appropriate target column selection for different testing scenarios.

---

## 🏗️ System Architecture

<img width="1089" height="374" alt="Screenshot 2025-11-25 at 08 00 25" src="https://github.com/user-attachments/assets/10a57fc5-ac14-4a4e-8ba9-ac1cce0bdd1f" />


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
├── data/                           # Real datasets for validation
│   └── real_datasets/
│       ├── compas_dataset.csv      # COMPAS dataset (6,172 samples)
│       ├── compas_processed.csv    # Preprocessed for fairness analysis
│       └── dataset_info.json       # Dataset documentation
│
├── demos/                          # Jupyter notebook demonstrations
│   └── FDK_Justice_Demo.ipynb      # Complete justice domain demo
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
├── tests/                         # Comprehensive test suite
│   ├── test_column_detection.py
│   └── test_justice_pipeline.py
│
├── app.py                         # Main Flask application
├── requirements.txt               # Python dependencies
├── render.yaml                    # Deployment configuration
├── .python-version               # Python version specification
├── LICENSE                       # Apache 2.0 License
├── NOTICE                        # Copyright notices
├── CHANGELOG.md                  # Version history and roadmap
└── README.md                     # Project documentation
```

### 🚀 Quick Start
Installation (Local Use)
Requirements:

Python 3.10.x

pip (Python package manager)

Install dependencies:

bash
pip install -r requirements.txt
Run the Flask application:

bash
python app.py
This starts the FDK™ web interface locally at http://localhost:5009.

Access Domain Upload Pages:
/justice - Justice domain with intelligent target selection

/business - Business domain

/education - Education domain

/finance - Finance domain

/health - Health domain

/hiring - Hiring domain

/governance - Governance domain


## 🎯 Jupyter Demo - Justice Domain
Explore the complete fairness audit workflow with real COMPAS dataset:

### Open the demo notebook
demos/FDK_Justice_Demo.ipynb
Demo Features:

Real COMPAS dataset analysis (6,172 samples)

36 justice-specific fairness metrics

Interactive visualizations

Legal compliance assessment

Exportable audit reports

### 🧠 High-Level Architecture
Conceptual pipeline (common pattern across domains):

User Upload (CSV)

        ↓
        
Automatic Domain Detection & Column Mapping

        ↓
        
Intelligent Target Selection (Pre/Post Implementation)

        ↓
        
Domain-Specific Fairness Pipeline

        ↓
        
36-56 Fairness Metrics and Composite Indicators

        ↓
        
Plain-Language Summary and Recommendations

        ↓
        
Downloadable JSON Audit Report

The underlying fairness metric definitions, taxonomies and domain rationales are documented in the FDK™ book.

#### 🎯 Intelligent Target Selection Workflow
Step-by-Step Usage:
Navigate to a domain (e.g., /justice)

Select test type:

Pre-Implementation: Baseline fairness of original algorithm

Post-Implementation: Fairness after bias correction

Upload your CSV dataset

FDK automatically:

Detects domain from column patterns

Selects appropriate target column

Provides reasoning for selection

Review and run the fairness audit

Download comprehensive JSON audit report

API Usage Examples:
python
import requests

### Intelligent target selection API
response = requests.post(
    'http://localhost:5009/api/intelligent-target',
    files={'file': open('dataset.csv', 'rb')},
    data={'test_type': 'post_implementation', 'domain': 'justice'}
)

print(response.json())
 {
   "success": True,
   "recommended_target": "two_year_recid",
   "reasoning": "Selected for bias-corrected model evaluation",
   "domain": "justice",
   "test_type": "post_implementation"
 }


### 🧪 Example Usage (Justice Domain with Intelligent Selection)
Open the Justice upload page (/justice)

Select test type:

Pre-Implementation for baseline fairness assessment

Post-Implementation for bias-corrected model evaluation

Upload COMPAS or similar justice dataset

Review automatically detected column mappings with intelligent suggestions

Run the fairness audit

Review:

On-screen human-readable summary

36 justice-specific fairness metrics

Legal compliance assessment

Downloadable JSON audit report

The same intelligent pattern applies to all seven domains, each with domain-specific rules for target selection.

### 📊 API Endpoints
Core Endpoints:
/api/detect-columns - Enhanced with test_type parameter

/api/intelligent-target - New intelligent selection endpoint

/api/domain/domain-audit - Domain-specific audit endpoints

Enhanced Detection API:
bash
curl -X POST -F "file=@compas.csv" \
     -F "test_type=pre_implementation" \
     http://localhost:5009/api/detect-columns

### 🧾 Documentation and Demos
Complete documentation suite available:

Jupyter notebook demo - Justice domain with real COMPAS dataset

Example usage guides - Step-by-step workflows for all domains

Architecture documentation - System design and component interactions

Domain-specific metrics - 36-56 fairness metrics per domain

Legal disclaimers - Compliance guidance for high-risk applications

### 🧪 Testing and Validation
Comprehensive test suite implemented:

17 unit tests for core pipeline functions

COMPAS dataset validation against known fairness benchmarks

Column detection and mapping logic tests

Error handling and edge case validation

### ⚖️ Legal and Ethical Disclaimer
FDK™ is a research and educational toolkit for fairness diagnostics.

It does not provide legal, financial, healthcare or regulatory advice.

It should not be used as the sole basis for any decision affecting individuals or groups.

Users are responsible for ensuring that datasets are appropriately anonymised and compliant with relevant regulations.

Full legal disclaimer is provided via the associated web interface and accompanying documentation.

#### 📄 Licence
Software (FDK™ Toolkit code): Apache License 2.0
See LICENSE and NOTICE in the repository root.

Book and explanatory text: CC BY-NC-SA 4.0
The book The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI remains under a Creative Commons licence suitable for educational and non-commercial use.

#### 📚 Citation and Credits
If you use or reference this toolkit in your research, please cite:

Tavakoli, H. (2025). The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI. London: Apress.

Repository: AI-Fairness-com/FDK-Toolkit
Correspondence: info@ai-fairness.com

#### 📖 Book BibTeX
bibtex
@book{Tavakoli2025FDK,
  author    = {Hamid Tavakoli},
  title     = {The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI},
  year      = {2025},
  publisher = {Apress},
  address   = {London},
  url       = {https://github.com/AI-Fairness-com/FDK-Toolkit}
}
#### 📖 Software BibTeX
bibtex
@software{Tavakoli2025FDKToolkit,
  author  = {Hamid Tavakoli},
  title   = {FDK™ Toolkit: Fairness Diagnostic Kit for Multi-Domain AI Auditing},
  year    = {2025},
  url     = {https://github.com/AI-Fairness-com/FDK-Toolkit},
  version = {v1.0.0}
}
