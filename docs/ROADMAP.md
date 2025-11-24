# 🧭 FDK™ Development Roadmap

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This roadmap outlines planned milestones for the **Fairness Diagnostic Kit (FDK™)**.  
It supports academic transparency, version planning, and future reproducibility across all seven domains.

The timeline follows semantic versioning (v1.x, v2.x) and is aligned with reviewer recommendations and the FDK™ book (Tavakoli, 2025).

---

# 🚀 Version Milestones

## ✅ v1.0.0 — Initial Public Release  
**Status:** *Planned for immediate release*  
This version includes:

- Fully functional seven-domain fairness pipeline  
- Flask-based web interface  
- Synthetic datasets and narrative summaries  
- JSON audit reporting engine  
- Documentation bundle:
  - installation.md  
  - architecture.md  
  - domains.md  
  - example_usage.md  
  - disclaimer.md  
- Apache 2.0 licensing  
- Repository restructuring for clarity  

Deliverable: Initial GitHub Release + changelog.

---

# 📘 v1.1.0 — Unit Testing & Benchmark Validation  
**Goals:**

- Add comprehensive test suite:
  - Column detection tests  
  - Metric verification tests  
  - Pipeline consistency tests  
  - JSON schema validation  
- Validate pipelines against known benchmarks:
  - COMPAS (justice domain)  
  - UCI Adult (hiring/finance domain)  
- Introduce automated CI checks (optional)

**Outcome:** Scientific reproducibility and baseline fairness validity.

---

# 📊 v1.2.0 — Jupyter Notebook Demos  
**Goals:**

Create demonstration notebooks for all seven domains:

- Business  
- Education  
- Finance  
- Health  
- Hiring  
- Justice  
- Governance  

Each will show:

- Dataset loading  
- Pipeline execution  
- Key fairness metrics  
- Explanation of outputs  
- Simple visualisations  

**Outcome:** Educational materials for teaching, workshops, and academic demonstrations.

---

# 📁 v1.3.0 — Dataset Expansion (Real + Synthetic)  
**Goals:**

- Add curated open datasets:
  - UCI Adult  
  - COMPAS  
  - OpenML Finance datasets  
  - Public health outcome datasets (anonymised)  
- Strengthen synthetic datasets using domain-accurate distributions:
  - Region × Age × Ethnicity × Outcome  
  - Sector-specific behaviour modelling

**Outcome:** Richer, domain-faithful datasets that reflect real bias structures.

---

# 📈 v1.4.0 — Metric Visualisation Layer  
**Goals:**

- Introduce Python-based fairness visualisations (matplotlib only):
  - Group-wise distributions  
  - Error rate gaps  
  - Calibration curves  
  - Disparate impact plots  
- Optional: Visualisation tab in HTML interface (static images)

**Outcome:** Improved interpretability for non-technical stakeholders.

---

# 🧩 v1.5.0 — Metric Taxonomy Reference  
**Goals:**

- Add comprehensive metric documentation (formulas + narrative)  
- Provide cross-domain mapping of metrics  
- Include a downloadable PDF reference for teaching  

**Outcome:** Academic-grade metric glossary aligned with the FDK™ book.

---

# 🕊️ v2.0.0 — Governance & Educational Edition  
**Goals:**

- Fully integrated teaching mode:  
  - Simplified explanations  
  - Interactive fairness examples  
  - Domain comparison framework  
- Optional “Governance Pack”:
  - Policy templates  
  - Fairness reporting examples  
  - Sector-specific ethics notes  
- Optional REST API extension

**Outcome:** A stable, education- and governance-ready fairness suite.

---

# 🔍 Long-Term Vision (v3.x and Beyond)

- Multi-metric optimisation suggestions  
- Bias mitigation modules (pre-processing & post-processing)  
- Cross-domain fairness parity explorer  
- Internationalisation (reduce English-only limitations)  
- Comprehensive fairness benchmark suite  
- Integration with wider fairness toolkits (AIF360, Fairlearn, OpenML)

---

# 📬 Contact

For collaboration, ideas, or feature requests:

```text
info@ai-fairness.com
