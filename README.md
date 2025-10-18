# 🧭 Fairness Diagnostic  Kit (FDK™)

**Open-source toolkit for fairness diagnostics across seven key AI domains.**  
Developed to support the *Fairness Diagnostic Kit (FDK™)* framework described in the book  
**_The Fairness Diagnostic Kit: Tools for Auditing, Education, and Governance of Responsible AI_** (Tavakoli, 2025).

---

## 🌍 Overview

The **FDK™ Toolkit** enables non-technical professionals, regulators, and educators to test and interpret algorithmic fairness **without coding**.  
It offers *domain-specific APIs*, *automatic feature detection*, and *plain-language audit reports* in JSON and human-readable form.

Each domain API operates through the [AI Fairness Portal](https://www.ai-fairness.com)  
and can also be run locally for demonstration or research purposes.

---

## 🧩 Supported Domains

FDK™ currently supports seven domains, each with its own validated fairness metrics and pipelines:

| Domain | Folder | Description |
|:--|:--|:--|
| Healthcare | `/healthcare/` | Fairness auditing for diagnostic and clinical decision-support models |
| Finance | `/finance/` | Auditing for credit scoring, lending, and insurance models |
| Hiring | `/hiring/` | Detection of demographic and procedural bias in recruitment pipelines |
| Education | `/education/` | Fairness testing of grading, admissions, and adaptive learning systems |
| Justice | `/justice/` | Evaluation of algorithmic fairness in risk assessment and sentencing tools |
| Environment | `/environment/` | Fairness and sustainability diagnostics for environmental and resource models |
| General / Business | `/general/` | For multi-domain and mixed datasets |

---

## ⚙️ Repository Structure

Each domain folder includes: 
## 🧩 Repository Structure
Each domain folder includes:
- Python pipeline (.py)
- HTML interface (.html)
- Example synthetic dataset (.csv)
- Human-readable summary (.txt)
- JSON output (.json)

---

## 📚 Citation and Credits
If you use or reference this toolkit in your research, please cite:

**Tavakoli, H.** (2025). *The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI.* London: Apress.

**Repository:** [AI-Fairness-com/FDK-Toolkit](https://github.com/AI-Fairness-com/FDK-Toolkit)  
**Correspondence:** [info@ai-fairness.com](mailto:info@ai-fairness.com)

---

### 📖 BibTeX
```bibtex
@book{Tavakoli2025FDK,
  author    = {Hamid Tavakoli},
  title     = {The Fairness Diagnostic Kit (FDK™): Tools for Auditing, Education, and Governance of Responsible AI},
  year      = {2025},
  publisher = {Apress},
  address   = {London},
  url       = {https://github.com/AI-Fairness-com/FDK-Toolkit}
}


