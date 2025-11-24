# 🌐 FDK™ Domain Explanations

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This document provides clear explanations of the **seven domains** covered by the Fairness Diagnostic Kit (FDK™).  
Each domain has its own bias patterns, fairness risks, stakeholder expectations, and metric combinations.  
The content follows the structure and definitions introduced in the *FDK™ book* (Tavakoli, 2025).

---

## 🧩 Overview of Domains

FDK™ covers fairness diagnostics in:

1. **Business**
2. **Education**
3. **Finance**
4. **Health**
5. **Hiring**
6. **Justice**
7. **Governance**

Each domain uses:
- Domain-specific fairness rationales  
- Tailored metric sets (36–56 metrics depending on domain)  
- Narrative summaries written for public-sector and educational use  
- Composite scoring aligned with risk-assessment principles  

---

# 1️⃣ Business Domain

## 📘 Purpose  
To evaluate fairness in **customer, employee, and corporate decision processes**, such as:

- Customer service outcomes  
- Loan eligibility pre-screens  
- Subscription risk models  
- Product recommendation fairness  
- Corporate compliance scoring  

## 💡 Typical Bias Risks  
- Unequal treatment of demographic groups in service outcomes  
- Disparate resolution rates for similar complaints  
- Biased risk assessments for customers with protected characteristics  
- Over- or under-representation of specific groups in fraud flags  

## 📊 Metrics Used (Representative Subset)  
- Statistical Parity Difference  
- Disparate Impact Ratio  
- TPR / FPR / FNR gaps  
- Calibration error gaps  
- Balanced accuracy difference  
- Group-wise error decomposition  

## 🧩 Pipeline Output  
- Composite bias score (0–100)  
- Severity: LOW / MEDIUM / HIGH  
- Narrative summary explaining fairness patterns  
- JSON report with complete metric set  

---

# 2️⃣ Education Domain

## 📘 Purpose  
To assess fairness in:

- Grading algorithms  
- Automated marking  
- Prediction of academic success  
- Admission screening tools  
- School or university risk models  

## 💡 Typical Bias Risks  
- Grade inflation/deflation for certain demographic groups  
- Unequal false negative rates (missing capable students)  
- Harmful misclassification in special-needs contexts  
- Region-based or socioeconomic discrimination  

## 📊 Metrics Used  
- Equal Opportunity Difference  
- FNR Gap (critical for admissions fairness)  
- Demographic Parity metrics  
- Predictive parity differences  
- Group error decomposition  

## 🧩 Pipeline Output  
Same structured JSON schema as other domains, with additional emphasis on:

- Misclassification harms  
- Group-wise academic opportunity gaps  

---

# 3️⃣ Finance Domain

## 📘 Purpose  
To provide fairness diagnostics for:

- Credit approval algorithms  
- Lending risk models  
- Financial inclusion scoring  
- Debt recovery prioritisation  
- Insurance pre-screening  

## 💡 Typical Bias Risks  
- Unequal loan approval rates across demographic groups  
- High false positives in high-risk predictions  
- Algorithmic reinforcement of existing financial inequality  
- Region-based stability bias (postcode effect)  

## 📊 Metrics Used  
- Disparate Impact  
- Approval Rate Ratio  
- Error-rate gaps across groups  
- Calibration-by-group  
- Group-wise ROC metrics  
- Financial inclusion indicators  

## 🧩 Pipeline Output  
Financial-domain narrative templates emphasise:

- Fair access  
- Regulatory compliance (FCA expectations)  
- Bias amplification detection  

---

# 4️⃣ Health Domain

## 📘 Purpose  
To evaluate fairness in:

- Diagnostic risk models  
- Disease prediction tools  
- Clinical triage scoring  
- Patient prioritisation systems  
- Preventive screening models  

## 💡 Typical Bias Risks  
- Higher misdiagnosis rates for minority groups  
- Unequal false negatives (critical clinical safety issue)  
- Region-linked disparities in predicted risk  
- Under-representation of disability groups in outcomes  

## 📊 Metrics Used  
- False Negative Rate Gap (core metric in clinical fairness)  
- Equalised Odds  
- Sensitivity / Specificity gaps  
- Calibration errors  
- Predictive value differences  

## 🧩 Pipeline Output  
Plain-language summary emphasises:

- Clinical risk  
- Safety implications  
- Potential harm severity  

---

# 5️⃣ Hiring Domain

## 📘 Purpose  
To evaluate fairness in:

- Resume screening algorithms  
- Promotion scoring  
- Shortlisting tools  
- Automated assessment outcomes  

## 💡 Typical Bias Risks  
- Disparate rejection rates  
- Over-penalising employment gaps for some groups  
- Gender-based false negatives  
- Algorithmic overreliance on proxy attributes  

## 📊 Metrics Used  
- Statistical Parity  
- Selection Rate Ratio  
- TPR / FNR gaps  
- Adverse Impact Ratio (commonly used in HR contexts)  
- Group-wise confusion matrix decomposition  

## 🧩 Pipeline Output  
The narrative emphasises:

- Hiring transparency  
- Diversity impact  
- Group-level selection disparities  

---

# 6️⃣ Justice Domain

## 📘 Purpose  
To evaluate fairness in:

- Automated recidivism scoring  
- Risk assessment tools  
- Sentencing recommendations  
- Pre-trial decision algorithms  

## 💡 Typical Bias Risks  
- Overprediction of risk for minority groups  
- FNR gaps that create unequal incarceration patterns  
- Statistical bias due to historical policing inequalities  
- Unequal calibration across groups  

## 📊 Metrics Used  
- FPR gap / FNR gap (critical metrics in justice fairness)  
- Balanced accuracy differences  
- Predictive parity  
- Calibration error  
- Group-wise risk distribution divergence  

## 🧩 Pipeline Output  
Narrative emphasises:

- Ethical and legal implications  
- Known sensitivity of justice domain fairness  
- Proportionality and equal treatment  

---

# 7️⃣ Governance Domain

## 📘 Purpose  
To evaluate fairness in:

- Public-sector resource allocation models  
- Policy analytics tools  
- Eligibility scoring for government programs  
- Social support triage  

## 💡 Typical Bias Risks  
- Unequal eligibility predictions  
- Region-based misclassification  
- Age-linked fairness failures  
- Socioeconomic segregation effects  

## 📊 Metrics Used  
- Statistical parity metrics  
- Resource allocation disparities  
- Composite inclusion indicators  
- Region-wise risk differences  

## 🧩 Pipeline Output  
Narrative emphasises:

- Public-sector transparency  
- Equality-of-access principles  
- Policy fairness implications  

---

# 🧭 Cross-Domain Consistency

All domains share:

- Identical input schema expectations  
- The same JSON output structure  
- The same upload → detect → confirm → run → download workflow  
- Domain-specific fairness metrics (unique sets based on typical harms)  
- A standardised composite scoring system  
- Plain-language summaries aligned with the FDK™ book  

This enables users to run fairness diagnostics across multiple sectors while maintaining interpretability and consistency.

---

# 📬 Contact

For domain-specific questions or academic collaboration:

```text
info@ai-fairness.com
