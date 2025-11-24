# 🧪 FDK™ Example Usage Guide

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This guide provides clear, practical examples of how to use each FDK™ domain interface.  
The examples follow the upload → auto-confirm → fairness audit → JSON report workflow used across all seven domains.

---

## 🌍 Overview

Every FDK™ domain follows the same operational pattern:

1. Upload a **de-identified CSV dataset**
2. FDK™ automatically detects:
   - Sensitive group attributes  
   - Outcome (`y_true`)  
   - Model predictions (`y_pred`)  
   - Optional probability scores (`y_prob`)
3. User confirms or adjusts detected mappings
4. Domain-specific fairness metrics are calculated
5. A human-readable summary and detailed JSON report are generated

This document demonstrates these steps using example datasets.

---

## 🚀 Jupyter Notebook Demo - Justice Domain

### 📓 Interactive Tutorial Now Available

Access our comprehensive Justice domain demo:

```bash
demos/FDK_Justice_Demo.ipynb
🎯 Demo Features:
Complete Justice pipeline execution (36 metrics)

COMPAS-like dataset analysis

Interactive visualizations

Legal compliance assessment

Exportable audit reports

🔬 Sample Results:
Statistical Parity Difference: 0.165

Disparate Impact Ratio: 0.734

Composite Bias Score: 0.1793

Severity: HIGH_BIAS

📊 Real Dataset Integration
Justice Domain with COMPAS Dataset
python
# Load and analyze real COMPAS dataset
from Justice.compas_loader import load_compas_data
from fdk_justice_pipeline import run_pipeline

# Load real COMPAS data (6,172 samples)
df = load_compas_data()

# Run comprehensive fairness audit
results = run_pipeline(df)

print(f"Composite Bias Score: {results['summary']['composite_bias_score']:.4f}")
print(f"Assessment: {results['summary']['overall_assessment']}")
Dataset Information
COMPAS Dataset: 6,172 real-world recidivism risk assessment records

Source: ProPublica COMPAS Analysis

Use Case: Justice domain fairness validation

Features: Demographic groups, true outcomes, model predictions

Jupyter Demo
Access the complete analysis in: demos/FDK_Justice_Demo.ipynb

The demo includes:

Real COMPAS dataset loading and preprocessing

36 justice-specific fairness metrics

Interactive visualizations

Legal compliance assessment

Exportable audit reports

1️⃣ Business Domain — Example Workflow
📁 Uploading the dataset
Navigate to:

/business-upload

Upload a CSV file with columns such as:

Gender

Ethnicity

CustomerServiceOutcome

ModelPrediction

ProbabilityScore

FDK™ automatically detects appropriate mappings.

🔍 Auto-Confirmed Mappings
The system displays:

Group attributes: Gender, Ethnicity

Outcome column: CustomerServiceOutcome

Prediction column: ModelPrediction

Probability column (optional): ProbabilityScore

You may adjust these if needed.

▶️ Running the Fairness Audit
Select "Run Fairness Audit".
The Business pipeline calculates fairness metrics, including:

Statistical Parity Difference

Disparate Impact Ratio

TPR / FPR / FNR gaps

Calibration differences

Balanced accuracy differences

📄 Example JSON Output (Excerpt)
json
{
  "domain": "Business",
  "severity": "MEDIUM",
  "composite_bias_score": 62.5,
  "fairness_metrics": {
    "statistical_parity_difference": -0.18,
    "disparate_impact": 0.77,
    "fpr_gap": 0.05,
    "fnr_gap": 0.12
  },
  "summary": "Group A is experiencing lower service outcomes..."
}
🧠 Human-Readable Summary
FDK™ also generates a narrative overview explaining:

Key disparities

Groups most affected

Direction of bias

Summary of potential risks

Contextual interpretation based on the Business domain

2️⃣ Education Domain — Example Workflow
📁 Upload
Go to:

/education-upload

Upload a dataset containing:

Demographics (e.g., Ethnicity, Gender, Region)

FinalGrade or PassFail

PredictedGrade

🔍 Auto-Confirm
FDK™ detects:

Sensitive attributes

Ground truth (y_true)

Prediction (y_pred)

▶️ Run Audit
Key Education-specific metrics include:

Equal Opportunity Difference

FNR Gap (critical in admissions fairness)

Calibration gaps

Prediction parity differences

Group error decomposition

3️⃣ Finance Domain — Example Workflow
Upload via:

/finance-upload

Dataset example columns:

AgeBand

Ethnicity

CreditOutcome

PredictedApproval

ProbabilityScore

Metrics computed include:

Disparate Impact

Approval Rate Ratio

Error rate gaps

Calibration-by-group

4️⃣ Health Domain — Example Workflow
Upload via:

/health-upload

Dataset should include:

Demographic attributes

Clinical ground truth (e.g., diagnosis)

Model prediction

Optional risk probability

Metrics highlight:

FNR Gap (critical in clinical safety)

Equalised Odds

Sensitivity / Specificity gaps

Predictive value differences

Narrative summary focuses on healthcare risk implications.

5️⃣ Hiring Domain — Example Workflow
Upload via:

/hiring-upload

Dataset may include:

Candidate demographic fields

Shortlisted (true outcome)

ModelShortlistPrediction

Metrics include:

Adverse Impact Ratio

TPR / FNR gaps

Selection Rate Ratio

Statistical Parity

Narrative focuses on employment transparency and diversity impact.

6️⃣ Justice Domain — Example Workflow
Upload via:

/justice-upload

Dataset may include:

Region, Ethnicity, AgeBand

JusticeOutcome (0/1)

PredictedRisk

Probability score

Metrics include:

FPR / FNR Gap

Predictive parity

Calibration error

Balanced accuracy difference

Narrative emphasises legal fairness concerns.

7️⃣ Governance Domain — Example Workflow
Upload via:

/governance-upload

Dataset example columns:

Demographic attributes

Eligibility or resource decision outcome

Model prediction

Optional scoring probability

Metrics include:

Statistical parity

Resource allocation disparities

Region-wise inclusion indicators

Narrative focuses on public-sector transparency and equality of access.

🎯 Key Takeaways
Across all domains:

The workflow is identical for user simplicity

The metrics differ based on domain fairness risks

The JSON output follows a shared schema

Narrative summaries help non-technical users interpret fairness results

Composite scoring standardises fairness evaluation across sectors

📬 Contact
For demonstration datasets, academic usage, or domain-specific questions:

info@ai-fairness.com
