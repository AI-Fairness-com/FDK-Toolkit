# ⚖️ Fairness Diagnostic Kit (FDK™)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub Repository](https://img.shields.io/badge/GitHub-AI--Fairness--com%2FFDK--Toolkit-lightgrey?logo=github)](https://github.com/AI-Fairness-com/FDK-Toolkit)
![Version](https://img.shields.io/badge/version-v2.0.0-blue)

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
## 🎯 Unified Intelligent System & Transparency Framework
**New in v2.0.0:** FDK now features a complete unified intelligent selection system across all 7 domains with full transparency tracking.

### 🚨 Key Methodological Enhancement
- **Ensures valid comparisons** by using identical target columns for pre/post tests
- **Complete metadata tracking** in all JSON audit reports
- **Self-documenting system** that prevents methodological errors

### 🤖 Intelligent Target Selection Features
FDK automatically detects the most appropriate target column based on:
- **Domain detection** (justice, health, education, hiring, finance, business, governance)
- **Test type** (pre-implementation vs. post-implementation)
- **Domain-specific priority rules** with validated configurations
- **Binary column verification** for fairness metrics

### 📊 Complete Transparency
Every JSON audit report now includes comprehensive metadata:
```
"metadata": {
  "target_column_used": "two_year_recid",
  "prediction_column_used": "is_recid",
  "test_type": "post_implementation",
  "intelligent_suggestion": "two_year_recid",
  "user_override_applied": true,
  "column_mapping": { ... },
  "timestamp": "2024-01-23T11:57:25.648398",
  "fdk_version": "justice_1.0_unified"
}
```
### 🎯 Validated Results
Justice Domain Validation: 15.2% fairness improvement confirmed with consistent target columns

All 7 Domains: Unified intelligent system operational

Full Audit Trail: Every calculation documented for scientific reproducibility

🔧 Usage:

### API endpoint for intelligent selection
curl -X POST -F "file=@dataset.csv" \
     -F "test_type=post_implementation" \
     http://localhost:5009/api/intelligent-target

### Enhanced detection with metadata
curl -X POST -F "file=@dataset.csv" \
     -F "test_type=pre_implementation" \
     http://localhost:5009/api/detect-columns

### UI Features:

✅ Pre/Post implementation test type selection across all domains
🔍 Real-time intelligent column analysis with visual feedback
🎯 Automatic target suggestions with domain-specific logic
📝 Complete metadata tracking in all reports
💡 Context-aware help and methodological guidance

This system ensures methodological integrity while providing full transparency for regulatory compliance and scientific validation.
---

## 🏗️ System Architecture

<img width="1089" height="374" alt="FDK System Architecture" src="https://github.com/user-attachments/assets/10a57fc5-ac14-4a4e-8ba9-ac1cce0bdd1f" />

### 🎯 Enhanced Workflow with Unified Intelligent System
1. **User Upload (CSV)** → Domain auto-detection
2. **Intelligent Target Selection** → Pre/Post implementation logic
3. **Transparent Column Mapping** → Complete metadata tracking
4. **Domain-Specific Fairness Pipeline** → 36-56 fairness metrics
5. **Metadata-Enhanced Audit Report** → JSON with full audit trail
6. **Plain-Language Summary** → Human-readable recommendations

### 🔍 Key Architectural Improvements (v2.0.0)
- **Unified Intelligence Layer**: Single `intelligent_target_selection()` across all domains
- **Metadata Injection**: Automatic audit trail generation in all reports
- **Methodological Validation**: Target column consistency verification
- **User Override Tracking**: Records when users manually select columns
- **Cross-Domain Consistency**: Same intelligent behavior across all 7 APIs

---

## 📊 Core Fairness Metrics with Methodological Tracking

The FDK™ Toolkit implements comprehensive fairness metrics across all domains with **complete methodological tracking**. Each metric calculation is now documented with target column, prediction column, and test type metadata.

| Metric | Definition | Domain Relevance | Metadata Tracking |
|:--|:--|:--|:--|
| **Statistical Parity Difference** | Difference in selection rates between groups | All domains - Base fairness measure | ✅ Target column verified |
| **Disparate Impact Ratio** | Ratio of selection rates between groups | Hiring, Justice - Legal compliance | ✅ Legal compliance documented |
| **Equal Opportunity Difference** | Difference in true positive rates between groups | Health, Justice - Error fairness | ✅ Prediction column tracked |
| **Equalized Odds** | Both TPR and FPR equality across groups | All domains - Comprehensive fairness | ✅ Test type recorded |
| **Predictive Parity** | Equality of positive predictive values | Health, Finance - Predictive reliability | ✅ Column mapping preserved |
| **False Discovery Rate Difference** | Difference in false discovery rates between groups | Justice, Business - Error distribution | ✅ User override logged |
| **Average Odds Difference** | Average of FPR and FNR differences | All domains - Balanced performance | ✅ Timestamped calculations |
| **Treatment Equality** | Ratio of FNR to FPR across groups | Health, Education - Resource allocation | ✅ Version control |
| **Demographic Parity Ratio** | Ratio of positive outcomes between groups | All domains - Outcome fairness | ✅ Domain-specific logic |
| **Predicted Positives per Group** | Count of positive predictions by group | All domains - Impact assessment | ✅ Intelligent selection source |

### 🔍 Enhanced Metric Validation (v2.0.0)
- **Column Consistency Check**: Ensures same target columns for pre/post comparisons
- **Methodological Documentation**: Every metric includes calculation parameters
- **Reproducibility Guarantee**: Full audit trail for scientific validation
- **Domain-Specific Calibration**: Metrics adjusted per domain requirements
- **Transparent Trade-offs**: Documented when metrics conflict

---

## 🏥 Real-World Use Cases with Methodological Validation

### Healthcare: Glaucoma Diagnosis AI
**Context**: AI system for early glaucoma detection from retinal images  
**Sensitive Attribute**: Ethnicity, Age, Gender  
**Fairness Risk**: Lower diagnostic accuracy for minority ethnic groups and older patients, potentially causing irreversible blindness through delayed detection  
**FDK Solution**: Tests 45 healthcare-specific metrics including calibration gaps, error rate parity, and subgroup performance to ensure equitable diagnostic accuracy across all demographic groups.  
**✅ v2.0.0 Enhancement**: Complete metadata tracking ensures same diagnostic criteria (target columns) are used when comparing pre/post algorithm improvements.

### Justice: Risk Assessment Tools - **VALIDATED EXAMPLE**
**Context**: Algorithm predicting recidivism risk for bail decisions  
**Sensitive Attribute**: Race, Socio-economic status  
**Fairness Risk**: Systematic over-prediction of risk for minority defendants  
**FDK Solution**: Applies 36 justice metrics including statistical parity, false positive rate differences, and causal fairness checks.  
**✅ v2.0.0 Validation**: Using consistent target columns (`two_year_recid`), FDK verified **15.2% fairness improvement** in BiasClean v2.7, demonstrating the importance of methodological consistency in fairness comparisons.

### Hiring: Resume Screening AI
**Context**: Automated screening of job applications  
**Sensitive Attribute**: Gender, Age, Education background  
**Fairness Risk**: Bias against female applicants in technical roles or older candidates  
**FDK Solution**: Evaluates 34 hiring metrics including selection rates, individual fairness consistency, and counterfactual fairness.  
**✅ v2.0.0 Enhancement**: Unified intelligent system ensures appropriate target column selection (`hired` vs `selected`) based on test type and dataset characteristics.

---

## 🧩 Supported Domains with Unified Intelligence

FDK™ currently supports seven domains, each with **unified intelligent selection** and **complete metadata tracking**:

| Domain   | Folder        | Description | v2.0.0 Enhancement |
|:--|:--|:--|:--|
| Business  | `/Business/`   | Fairness auditing for customer, employee and corporate decision pipelines | ✅ Unified intelligent selection |
| Education | `/Education/`  | Fairness testing of grading, admissions and educational decision systems | ✅ Complete metadata tracking |
| Finance   | `/Finance/`    | Auditing for credit scoring, lending and financial inclusion models | ✅ Methodological validation |
| Health    | `/Health/`     | Fairness assessment of diagnostic and healthcare support models | ✅ Target column consistency |
| Hiring    | `/Hiring/`     | Detection of demographic and procedural bias in recruitment pipelines | ✅ Pre/post test logic |
| Justice   | `/Justice/`    | **Validated** evaluation of algorithmic fairness in justice and risk assessment tools | ✅ **15.2% improvement verified** |
| Governance| `/Governance/` | Diagnostics for public-sector, policy and governance-related AI systems | ✅ Cross-domain consistency |
---

## ⚙️ Repository Structure with Enhanced Components

Each domain folder now includes **unified intelligent system integration**:

- Python pipeline (`fdk_<domain>_pipeline.py`) - **Enhanced with metadata injection**
- Flask routing / API file (`fdk_<domain>.py`) - **Updated with unified intelligent selection**
- HTML interface templates (`upload_*.html`, `auto_confirm_*.html`, `result_*.html`) - **Enhanced with visual feedback**
- Example synthetic outputs (JSON reports) with **complete metadata tracking**

Top-level structure (v2.0.0 enhancements highlighted):

```text
FDK-Toolkit/
│
├── data/                           # Real datasets for validation
│   └── real_datasets/
│       ├── compas_dataset.csv      # COMPAS dataset (6,172 samples) - **VALIDATED**
│       ├── compas_processed.csv    # Preprocessed for fairness analysis
│       └── dataset_info.json       # Dataset documentation
│
├── demos/                          # Jupyter notebook demonstrations
│   └── FDK_Justice_Demo.ipynb      # Complete justice domain demo with **methodological validation**
│
├── docs/                           # Comprehensive documentation
│   ├── installation.md            # Step-by-step installation guide
│   ├── architecture.md            # **Updated** system architecture details
│   ├── domains.md                 # Domain-specific explanations
│   ├── example_usage.md           # Practical usage examples
│   └── disclaimer.md              # Legal and ethical guidelines
│
├── Business/                      # Business domain API - **UPDATED TO v2.0.0**
│   ├── fdk_business_pipeline.py   # Core fairness pipeline with metadata
│   ├── fdk_business.py            # Flask API routes with unified intelligence
│   ├── upload_business.html       # Web interface with enhanced UI
│   ├── auto_confirm_business.html # Column mapping confirmation
│   └── result_business.html       # Results display
│
├── Education/                     # Education domain API - **UPDATED TO v2.0.0**
│   ├── fdk_education_pipeline.py  # **Enhanced with metadata**
│   ├── fdk_education.py           # **Unified intelligent system**
│   └── [corresponding HTML templates updated]
│
├── Finance/                       # Finance domain API - **UPDATED TO v2.0.0**
│   ├── fdk_finance_pipeline.py    # **Enhanced with metadata**
│   ├── fdk_finance.py             # **Unified intelligent system**
│   └── [corresponding HTML templates updated]
│
├── Health/                        # Health domain API - **UPDATED TO v2.0.0**
│   ├── fdk_health_pipeline.py     # **Enhanced with metadata**
│   ├── fdk_health.py              # **Unified intelligent system**
│   └── [corresponding HTML templates updated]
│
├── Hiring/                        # Hiring domain API - **UPDATED TO v2.0.0**
│   ├── fdk_hiring_pipeline.py     # **Enhanced with metadata**
│   ├── fdk_hiring.py              # **Unified intelligent system**
│   └── [corresponding HTML templates updated]
│
├── Justice/                       # Justice domain API - **VALIDATED v2.0.0**
│   ├── fdk_justice_pipeline.py    # **15.2% improvement verified**
│   ├── fdk_justice.py             # **Methodological bug fixed**
│   └── [corresponding HTML templates updated]
│
├── Governance/                    # Governance domain API - **UPDATED TO v2.0.0**
│   ├── fdk_governance_pipeline.py # **Enhanced with metadata**
│   ├── fdk_governance.py          # **Unified intelligent system**
│   └── [corresponding HTML templates updated]
│
├── tests/                         # Comprehensive test suite
│   ├── test_column_detection.py   # **Enhanced for unified system**
│   └── test_justice_pipeline.py   # **Methodological validation tests**
│
├── app.py                         # **Main FDK.py application** (renamed/updated)
├── FDK.py                         # **Universal intelligent system core**
├── requirements.txt               # Python dependencies
├── render.yaml                    # Deployment configuration
├── .python-version               # Python version specification
├── LICENSE                       # Apache 2.0 License
├── NOTICE                        # Copyright notices
├── CHANGELOG.md                  # **Updated to v2.0.0**
└── README.md                     # **Updated project documentation**
```

### 🚀 Quick Start
Installation (Local Use)
Requirements:

Python 3.10.x

pip (Python package manager)

Install dependencies:

```text
pip install -r requirements.txt
```

Run the Flask application:

```text
python app.py
```

This starts the FDK™ web interface locally at http://localhost:5009.

Access Domain Upload Pages:

```text
/justice - Justice domain with intelligent target selection

/business - Business domain

/education - Education domain

/finance - Finance domain

/health - Health domain

/hiring - Hiring domain

/governance - Governance domain
```
---

## 🎯 Jupyter Demo - Justice Domain with Methodological Validation
Explore the complete fairness audit workflow with **validated methodology** using the real COMPAS dataset:

### Open the enhanced demo notebook
`demos/FDK_Justice_Demo.ipynb`

**Demo Features (v2.0.0 Enhanced):**
- Real COMPAS dataset analysis (6,172 samples)
- 36 justice-specific fairness metrics with **metadata tracking**
- **Methodological validation** of target column consistency
- Interactive visualizations with **audit trail documentation**
- Legal compliance assessment with **transparent calculations**
- Exportable JSON audit reports with **complete metadata**
- **15.2% fairness improvement verification** example

---

### 🧠 Enhanced High-Level Architecture (v2.0.0)
Conceptual pipeline with **unified intelligence** (common pattern across all 7 domains):

**User Upload (CSV) with test type selection**
        ↓
**Automatic Domain Detection & Unified Intelligent Target Selection**
        ↓
**Transparent Column Mapping with Complete Metadata Tracking**
        ↓
**Domain-Specific Fairness Pipeline with Methodological Validation**
        ↓
**36-56 Fairness Metrics with Audit Trail Documentation**
        ↓
**Plain-Language Summary and Methodological Recommendations**
        ↓
**Downloadable JSON Audit Report with Full Metadata**

The underlying fairness metric definitions, taxonomies, and **methodological validation protocols** are documented in the FDK™ book and enhanced in v2.0.0.

---

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

### 🧪 Enhanced Testing and Validation (v2.0.0)
Comprehensive test suite implemented with **methodological focus**:

- 17+ unit tests for core pipeline functions
- **Methodological validation tests** for target column consistency
- COMPAS dataset validation against known fairness benchmarks
- **Metadata completeness verification** for all JSON reports
- Column detection and mapping logic tests with **unified intelligence**
- Error handling and edge case validation
- **Cross-domain consistency tests** for unified intelligent system
- **User override functionality verification**

---

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
  version = {v2.0.0}
}
