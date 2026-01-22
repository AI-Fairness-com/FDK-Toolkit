# 📝 FDK™ Changelog

This changelog documents all notable changes to the **Fairness Diagnostic Kit (FDK™)**.  
It follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH).

---

## 📌 v1.1.0 — Universal Intelligent Target Selection System  
**Date:** 2026-01-23  
**Status:** Released

### ✨ Added
- **Universal Intelligent Target Selection System** for all 7 domains:
  - Domain auto-detection from column patterns (justice, health, education, hiring, finance, business, governance)
  - Intelligent target column selection based on test type (pre vs. post-implementation)
  - Domain-specific priority rules with customized logic for each domain
  - 7 new intelligent selection functions in `FDK.py`
- **New API endpoint**: `/api/intelligent-target`
  - Returns recommended target column with reasoning
  - Provides domain detection and intelligent selection
  - Includes all columns and universal suggestions for backward compatibility
- **Enhanced user interface**:
  - Test type selector (Pre/Post Implementation)
  - Real-time intelligent column analysis
  - Context-aware hints and explanations
  - Auto-filled target columns
  - Maintains manual selection as fallback

### 🔧 Improved
- **Enhanced `/api/detect-columns` endpoint**:
  - Now accepts `test_type` parameter (pre_implementation/post_implementation)
  - Returns intelligent suggestions based on test type
  - Maintains full backward compatibility with existing workflows
- **Justice domain UX overhaul**:
  - Replaced manual dropdown with intelligent test type selection
  - Added visual distinction between pre/post implementation tests
  - Improved user guidance with real-time feedback
- **Core FDK.py intelligence**:
  - Added `detect_domain_from_columns()` function
  - Added `intelligent_target_selection()` with domain-specific rules
  - Added binary column verification functions
  - Enhanced fallback mechanisms with multi-level intelligence

### 🎯 Key Features
1. **Universal System**: One implementation works across all 7 domains
2. **Test-Type Awareness**: Different logic for baseline (pre) vs. corrected model (post) testing
3. **Domain-Specific Rules**: Custom priority columns for each domain and test type
4. **Intelligent Fallbacks**: Multi-level fallback when domain rules don't match
5. **Backward Compatible**: Maintains existing API responses and workflows
6. **Enhanced UX**: Simplified interface with intelligent defaults

### 🔄 Files Modified
- `FDK.py` - Added universal intelligent selection system (7 new functions, 2 enhanced endpoints)
- `upload_justice.html` - Updated with test type selector and intelligent UI
- `fdk_justice.py` - Integrated with intelligent target selection
- *(Ready for rollout to other 6 domains: business, education, finance, health, hiring, governance)*

### 🧪 Testing Status
- ✅ Justice domain fully tested with COMPAS dataset
- ✅ API endpoints validated with pre/post implementation tests
- ✅ Backward compatibility confirmed
- ✅ Ready for rollout to remaining domains

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
- Automatic column detection (group attributes, outcomes, predictions)  
- Domain-specific pipelines with 36–56 fairness metrics  
- Composite bias scoring engine  
- Plain-language narrative summary generation  
- JSON report generator and download endpoint  
- Synthetic example datasets  
- Documentation suite:
  - `installation.md`  
  - `architecture.md`  
  - `domains.md`  
  - `example_usage.md`  
  - `disclaimer.md`  
  - `roadmap.md`  
- Apache-2.0 licensing + NOTICE file  
- Repository restructuring for clarity

### 🔧 Improved
- Consistent naming conventions across domain blueprints  
- Unified JSON schema for audit reports  
- Stable folder structure for future scalability

### 📁 Repository
https://github.com/AI-Fairness-com/FDK-Toolkit

---

## 📌 v1.2.0 — Unit Tests & Benchmark Validation *(Planned)*

### ✨ Planned Additions
- Full test suite:
  - Column detection tests  
  - Metric correctness tests  
  - Pipeline consistency tests  
  - JSON schema checks  
- Validation using benchmark datasets:
  - COMPAS (justice)  
  - UCI Adult (hiring/finance)
- Intelligent target selection tests across all 7 domains

### 🎯 Expected Outcome
Improved reproducibility, scientific integrity, and reviewer confidence.

---

## 📌 v1.3.0 — Jupyter Notebook Demos *(Planned)*

### ✨ Planned Additions
- Demonstration notebooks for all seven domains  
- Sample execution, metric inspection, and interpretability walkthroughs  
- Intelligent target selection demonstration

---

## 📌 v1.4.0 — Dataset Expansion *(Planned)*

### ✨ Planned Additions
- Real open datasets added to `/datasets/`  
- Stronger domain-accurate synthetic examples  
- Pre/post implementation example datasets

---

## 📌 v1.5.0 — Visualisation Tools *(Planned)*

### ✨ Planned Additions
- Matplotlib-based group fairness plots  
- Error-gap charts  
- Calibration visualisations  
- Intelligent selection visualization

---

## 📌 v2.0.0 — Governance & Educational Edition *(Planned)*

### ✨ Planned Additions
- Integrated teaching mode  
- Educational examples  
- Policy alignment templates  
- Optional REST API extension  
- Advanced intelligent selection with explainable AI

---

## 🔮 Future Development

### Immediate Next Steps
- Rollout intelligent target selection to remaining 6 domains (business, education, finance, health, hiring, governance)
- Add domain-specific visual themes for test type selectors
- Enhanced error handling for edge cases in intelligent selection

### Planned Enhancements
- Batch processing capabilities
- Comparative fairness analysis across multiple datasets
- Integration with popular ML frameworks
- Advanced visualization dashboard
- API key authentication for enterprise use
- Extended intelligent selection with feature importance analysis

---

## 📬 Contact

For versioning questions or release notes:

```text
info@ai-fairness.com
