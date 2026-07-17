# FDK Toolkit: Fairness Diagnostic Kit

**No-code AI fairness auditing for professionals.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-methodology.light-yellow)](docs/methodology.md)

## What is FDK Toolkit?

The Fairness Diagnostic Kit (FDK™) is the first absolute no-code framework for auditing algorithmic fairness. It allows anyone—regardless of programming background—to test, interpret, and document fairness in AI-driven decision systems.

Source: FDK Manuscript, Preface, p. 7

> *"FDK™ is the first absolute no-code framework for auditing algorithmic fairness. It allows anyone, regardless of programming background, to test, interpret, and document fairness in AI-driven decision systems."*

## Quick Start

### 1. Visit the website
Go to [www.ai-fairness.com](https://www.ai-fairness.com)

### 2. Select your domain
Choose from seven domains: Healthcare, Justice, Finance, Education, Hiring, Business, Governance

### 3. Upload your dataset (CSV format)
- Include predictions/outcomes
- Include protected attributes (gender, ethnicity, age, etc.)
- Remove personally identifiable information

### 4. Run the audit
Click "Run Audit" and receive two outputs:
- **Professional JSON report** – detailed metrics for auditors
- **Public summary** – plain-language explanation

## Worked Examples

| Domain | Dataset | Source |
|--------|---------|--------|
| Justice | COMPAS recidivism | [examples/justice_COMPAS.ipynb](examples/justice_COMPAS.ipynb) |
| Hiring | OpenIntro Resume Callback | [examples/hiring_resume.ipynb](examples/hiring_resume.ipynb) |
| Education | OULAD student success | [examples/education_OULAD.ipynb](examples/education_OULAD.ipynb) |
| Finance | German Credit Dataset | [examples/finance_credit.ipynb](examples/finance_credit.ipynb) |
| Healthcare | MIMIC-IV ICU admissions | [examples/healthcare_MIMIC.ipynb](examples/healthcare_MIMIC.ipynb) |

## Documentation

- [**Fairness Metrics Methodology & Citation Registry**](FDK_METHODOLOGY.md) – The complete scientific audit of all 158 fairness metrics: source, formula, and peer-reviewed citation for every one, plus the full record of defects found and fixed prior to launch. Companion data: [full citations spreadsheet](FDK_158_metrics_citations.csv).
- [Methodology: Fairness Metrics Explained](docs/methodology.md) – Mathematical formulas and plain-language analogies
- [Comparison with AIF360, Fairlearn, Aequitas](docs/comparison.md) – How FDK differs from existing toolkits
- [Legal and Ethical Disclaimer](DISCLAIMER.md) – Important: FDK is not a legal compliance tool

## Domain APIs

| Domain | Key Metrics |
|--------|-------------|
| Healthcare | Equal opportunity, calibration, false negative rate parity |
| Justice | Equalized odds, disparate impact, error rate balance |
| Finance | Statistical parity, predictive parity, calibration |
| Education | Consistency, equal opportunity, worst-group accuracy |
| Hiring | Demographic parity, false negative rate, counterfactual fairness |
| Business | Group fairness, segmentation parity, temporal fairness |
| Governance | Conditional demographic disparity, transparency index |

*Full metric lists: Appendix B of the FDK manuscript (pp. 351–365)*

## Output Format

### Professional Report (JSON)
Contains all computed metrics, subgroup comparisons, and confidence intervals.

### Public Summary (Plain Language)
Written in accessible language for non-technical stakeholders.

Example (Source: Chapter 10.3, p. 151):
> *"The disparate impact ratio (0.5625) suggests that one group is predicted to re-offend at roughly half the rate of another, falling well below the commonly accepted 0.8 threshold for fairness."*

## Important Limitations

**FDK is a diagnostic tool, not a legal compliance tool.** Please read [DISCLAIMER.md](DISCLAIMER.md) before use.

Source: Chapter 1.1, p. 29

## License

- Software: Apache License 2.0
- Documentation: CC BY-NC-SA 4.0

## Repository

[github.com/AI-Fairness-com/FDK-Toolkit](https://github.com/AI-Fairness-com/FDK-Toolkit)

## Reference

Tavakoli, H. (2026). *The AI Fairness Diagnostic Kit: From Principle to Practice in No-Code AI Fairness Auditing*.
