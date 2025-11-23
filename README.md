# 🌐 **FDK™ — Fairness Diagnostic Kit**  
### *Multi-Domain AI Fairness Auditing Toolkit (2025)*  
Automated Bias Detection • Domain-Specific Pipelines • Apache-2.0 Licensed

---

## 📊 **What is FDK™?**
FDK™ is a multi-domain fairness auditing framework designed to analyse AI-driven decisions across seven high-risk application areas.  
It supports automated dataset validation, domain-specific fairness metrics (36–56 per domain), composite bias scoring, and complete audit reporting.

FDK™ is built around the theoretical framework documented in the book:  
**_Fairness Diagnostic Kit (FDK™)_ — 2025 Edition.**

---

# 🗂️ **Domains Supported**
A unified toolkit with seven domain-specific fairness pipelines.

| Domain | Folder | Icon |
|--------|--------|-------|
| **Business** | `/Business/` | 🏢 |
| **Education** | `/Education/` | 🎓 |
| **Finance** | `/Finance/` | 💷 |
| **Health** | `/Health/` | 🏥 |
| **Hiring** | `/Hiring/` | 👥 |
| **Justice** | `/Justice/` | ⚖️ |
| **Governance** | `/Governance/` | 🏛️ |

Each domain has:
- Upload page  
- Auto-mapping page  
- Result page with fairness summary  
- Pipeline using domain-tailored metrics  
- JSON report generation  

---

# ⚙️ **Core Features**

### 🔍 Automated Column Detection
FDK™ detects:
- Sensitive attribute(s)  
- Outcome label (`y_true`)  
- Model predictions (`y_pred`)  
- Probability scores (`y_prob`, if available)

### 📐 36–56 Fairness Metrics per Domain
Including:
- Group fairness  
- Error rate fairness  
- Calibration  
- Distributional fairness  
- Drift metrics  
- Domain-specific metrics  
- Composite bias score (0–1)  
- Severity classification: LOW • MEDIUM • HIGH

### 📄 Automated JSON + Natural-Language Reports
Reports include:
- Metric breakdown  
- Score-weighted summary  
- Interpretative analysis  
- Domain-tailored recommendations  

---

# 📁 **Repository Structure**

```
FDK-Toolkit/
│
├── Business/
├── Education/
├── Finance/
├── Health/
├── Hiring/
├── Justice/
├── Governance/
│
├── app.py
├── requirements.txt
├── render.yaml
├── .python-version
│
├── LICENSE      (Apache-2.0)
├── NOTICE
└── README.md
```

---

# 🧩 **System Architecture**

```
User Upload (CSV)
        ↓
Column Auto-Mapping
        ↓
Domain-Specific Fairness Pipeline
        ↓
Metric Computation (36–56 metrics)
        ↓
Composite Bias Score + Severity Classification
        ↓
Natural-Language Summary Generation
        ↓
Downloadable JSON Report
```

---

# 🚀 **Installation**

### 📌 Requirements
- Python **3.10.x**

### 📦 Install Packages
```bash
pip install -r requirements.txt
```

### ▶️ Run the Toolkit
```bash
python app.py
```

---

# 🧪 **Usage Example**

### 1️⃣ Upload Dataset  
Go to:
```
/business-upload
/education-upload
/finance-upload
/health-upload
/hiring-upload
/justice-upload
/governance-upload
```

### 2️⃣ Confirm Auto-Detected Mappings  
FDK™ proposes `group`, `y_true`, `y_pred`, and optional `y_prob`.

### 3️⃣ Run Domain Audit  
Produces:
- JSON report  
- Human-readable fairness summary  
- Category-wise metric tables  
- Severity classification  

### 4️⃣ Download Results  
Each report is timestamped and stored in the appropriate domain folder.

---

# 📘 **Documentation Roadmap (per Faria’s Review)**

### ✔️ Included Now
- Installation instructions  
- Architecture explanation  
- Domain descriptions  
- Use-case overview  
- Licence (Apache-2.0)  
- Citation entry  
- Roadmap  

### ⏳ To Be Added (v1.0.1 – v1.0.2)
- 7 Jupyter notebooks (one per domain)  
- Example real + synthetic datasets  
- Unit tests + benchmark validation  
- Expanded API documentation  
- Versioned changelog  

---

# ⚖️ **Legal Disclaimer**

FDK™ is a research toolkit for fairness assessment.  
It **does not** provide legal, financial, healthcare, or regulatory advice.  
Users must ensure all datasets are **anonymised** and free from personal identifiers.  
The authors accept no liability for how results are used in practice.

Full disclaimer available at the `/legal-disclaimer/` route.

---

# 📄 **Licence**
Software is released under:

### **Apache License 2.0**  
(See `LICENSE` and `NOTICE` at repo root)

Book content is covered under **CC BY-NC-SA 4.0** (non-software licence).

---

# 🔬 **Citation**

```
@software{Tavakoli2025FDK,
  author = {Hamid Tavakoli},
  title  = {Fairness Diagnostic Kit (FDK™)},
  year   = {2025},
  url    = {https://github.com/.../FDK-Toolkit}
}
```

---

# 🛣️ **Roadmap**
- Domain notebooks (Business → Governance)  
- Dataset library (synthetic + open datasets)  
- Metric expansion (towards 80+ metrics)  
- Automated tests (pytest suite)  
- Optional PyPI distribution  
- Interactive dashboard layer  

---

# © 2025 Hamid Tavakoli • Optics AI Ltd
