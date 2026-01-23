# 📝 FDK™ Changelog

This changelog documents all notable changes to the **Fairness Diagnostic Kit (FDK™)**.  
It follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH).

---

## 📌 v2.0.0 — Complete Unified Intelligent System & Transparency Framework  
**Date:** 2024-01-23  
**Status:** Released

### 🚨 Critical Methodological Fix
- **Resolved target column discrepancy** in Justice domain validation study
- **Ensured apples-to-apples comparison** by using identical target columns (`two_year_recid`) for pre/post tests
- **Validated BiasClean v2.7 improvement**: 15.2% fairness improvement confirmed with consistent methodology

### ✨ Added
- **7-Domain Unified Intelligent System**: All domains now use `FDK.py`'s `intelligent_target_selection()`
- **Complete Metadata Tracking**: Every JSON report includes:
  - `target_column_used`: Actual target column used in calculations
  - `prediction_column_used`: Prediction column used
  - `test_type`: Pre/Post implementation test type
  - `intelligent_suggestion`: FDK's intelligent suggestion
  - `user_override_applied`: Whether user manually selected column
  - `column_mapping`: Complete mapping of standard to original columns
  - `timestamp`: ISO 8601 timestamp
  - `fdk_version`: Version identifier
- **Enhanced Transparency**: Full audit trail of all columns and decisions

### 🔧 Improved
- **Justice Domain Priority Fix**: `two_year_recid` now prioritized over `is_recid` for pre-implementation tests
- **Unified Parameter System**: Consistent `target_column` and `target_column_fallback` parameter names
- **HTML Interface**: All 7 domains feature intelligent selection with visual feedback
- **Error Prevention**: System detects and prevents methodological inconsistencies
- **Backward Compatibility**: Maintains all existing API contracts

### 🎯 Key Features
1. **Methodological Integrity**: Ensures valid comparisons with consistent target columns
2. **Full Transparency**: Every calculation documented in JSON metadata
3. **User Control**: Intelligent suggestions with manual override capability
4. **Domain Consistency**: Same intelligent system across all 7 domains
5. **Self-Documenting**: Automatic audit trails for scientific reproducibility

### 🔄 Files Modified
- `FDK.py` - Enhanced intelligent selection with justice domain priority fix
- `Justice/fdk_justice.py` - Complete rewrite with unified intelligent system and metadata
- `Business/fdk_business.py` - Updated to unified system
- `Education/fdk_education.py` - Updated to unified system  
- `Finance/fdk_finance.py` - Updated to unified system
- `Health/fdk_health.py` - Updated to unified system
- `Hiring/fdk_hiring.py` - Updated to unified system
- `Governance/fdk_governance.py` - Updated to unified system
- All 7 domain HTML upload templates with enhanced intelligent selection UI

### 🧪 Validation Results
- ✅ **Justice Domain**: 15.2% fairness improvement verified with consistent target columns
- ✅ **All Domains**: Intelligent selection operational with user override
- ✅ **Metadata**: Complete audit trail in all JSON reports
- ✅ **Backward Compatibility**: Existing workflows fully preserved
- ✅ **Transparency**: Every audit documents exact methodology used

---

## 📌 v1.1.0 — Universal Intelligent Target Selection System (Justice Domain Pilot)  
**Date:** 2026-01-23  
**Status:** Superseded by v2.0.0

### ✨ Added (Initial Pilot)
- **Intelligent Target Selection Prototype** for Justice domain
- **Domain auto-detection** from column patterns
- **Test type awareness** (pre vs. post-implementation)
- **New API endpoint**: `/api/intelligent-target`

### 🔧 Improved
- **Enhanced `/api/detect-columns` endpoint** with test type parameter
- **Justice domain UX** with test type selector
- **Core FDK.py intelligence functions**

### ⚠️ Note
*This was a pilot release for Justice domain only. The system has been fully expanded to all 7 domains in v2.0.0.*

---

## 📌 v1.0.0 — Initial Public Release  
**Date:** 2025-12-15  
**Status:** Published (first stable release)

### ✨ Added
- Seven fully functional domain modules:
  - Business  
  - Education  
  - Finance  
  - Health  
  - Hiring  
  - Justice  
  - Governance  
- Flask-based multi-domain web interface  
- Automatic column detection  
- Domain-specific pipelines with 36–56 fairness metrics  
- Composite bias scoring engine  
- Plain-language narrative summary generation  
- JSON report generator and download endpoint  
- Synthetic example datasets  
- Complete documentation suite
- Apache-2.0 licensing + NOTICE file  

### 🔧 Improved
- Consistent naming conventions across domain blueprints  
- Unified JSON schema for audit reports  
- Stable folder structure for scalability

### 📁 Repository
https://github.com/AI-Fairness-com/FDK-Toolkit

---

## 📌 v1.2.0 — Unit Tests & Benchmark Validation *(In Progress)*

### ✨ Planned Additions
- Comprehensive test suite:
  - Column detection tests  
  - Metric correctness tests  
  - Pipeline consistency tests  
  - JSON schema validation
- Benchmark dataset validation:
  - COMPAS (justice)  
  - UCI Adult (hiring/finance)
  - Medical datasets (health)
- Intelligent selection verification tests

### 🎯 Expected Outcome
Enhanced reproducibility, scientific integrity, and deployment confidence.

---

## 📌 v1.3.0 — Jupyter Notebook Demos *(Planned)*

### ✨ Planned Additions
- Interactive notebooks for all seven domains  
- Step-by-step execution guides
- Metric interpretation tutorials  
- Intelligent selection demonstrations
- Best practices documentation

---

## 📌 v1.4.0 — Dataset Expansion *(Planned)*

### ✨ Planned Additions
- Curated open datasets in `/datasets/`  
- Domain-accurate synthetic examples
- Pre/post implementation benchmark sets
- Dataset documentation and citation guides

---

## 📌 v1.5.0 — Visualization Tools *(Planned)*

### ✨ Planned Additions
- Matplotlib-based fairness visualization
- Group disparity charts
- Error distribution plots
- Calibration curves
- Interactive metric dashboards

---

## 📌 v2.1.0 — Advanced Analytics & API *(Planned)*

### ✨ Planned Additions
- Batch processing capabilities
- Comparative fairness analysis across multiple datasets
- REST API for programmatic access
- Advanced intelligent selection with explainable AI
- Integration with popular ML frameworks

---

## 🔮 Current Development Focus

### Immediate Priorities
- Complete test suite development (v1.2.0)
- Benchmark validation across all domains
- Performance optimization for large datasets

### Research & Development
- Advanced fairness visualization techniques
- Real-time fairness monitoring capabilities
- Integration with regulatory compliance frameworks
- Educational materials for fairness auditing

### Community & Ecosystem
- Contributor guidelines and documentation
- Plugin architecture for custom metrics
- Community dataset contributions
- Academic collaboration framework

---

## 📬 Contact

For versioning questions or release notes:

info@ai-fairness.com
