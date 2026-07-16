# FDK Toolkit vs Existing Fairness Frameworks

This document positions the FDK Toolkit relative to existing open-source fairness libraries. All claims are sourced directly from the FDK manuscript (Tavakoli, 2026).

## Summary Comparison Matrix

| Dimension | FDK Toolkit | AIF360 (IBM) | Fairlearn (Microsoft) | Aequitas (UChicago) |
|-----------|-------------|--------------|----------------------|---------------------|
| **Primary focus** | Governance & audit evidence | Research & algorithm zoo | ML pipeline integration | Risk assessment dashboard |
| **Target user** | Auditors, PMs, clinicians, educators | Data scientists | ML engineers | Policy analysts |
| **Coding required?** | No (no-code interface) | Yes (Python) | Yes (Python) | Minimal (config only) |
| **Plain-language output** | Yes (dual reports) | No | No | Partial |
| **Domain-specific bundles** | Yes (7 domains) | No (general) | No (general) | No (general) |
| **Causal fairness** | Partial (correlation-based proxy metrics; not DoWhy-integrated) | No | No | No |
| **Individual fairness** | Yes (consistency, counterfactual) | Limited | No | No |

*Source: Compiled from Chapter 2.5, 2.6, 4.4, and Appendix D (p. 398)*

---

## Detailed Analysis

### AI Fairness 360 (AIF360)

**What it covers (Source: Chapter 2.6, p. 50–51):**
- Nearly 70 metrics for bias detection
- Mitigation algorithms spanning pre-processing, in-processing, post-processing
- Widely cited as a benchmark reference

**Where it falls short (Source: Chapter 2.6, p. 51–52):**

> *"The toolkit demands a high level of statistical and coding competence, rendering it inaccessible to most professionals who encounter artificial intelligence in operational settings."*

> *"Individual fairness, the principle that similar individuals should receive similar outcomes, is not substantively addressed. Nor does the toolkit encompass causal fairness."*

> *"AIF360 does not produce plain-language summaries, which limits its usefulness for non-specialist decision-makers such as clinicians, teachers, or regulators."*

> *"The extensive metric coverage in AIF360 may encourage organizations to approach fairness as a checklist exercise. Compliance with statistical tests can foster false reassurance."*

### Fairlearn (Microsoft)

**What it covers (Source: Chapter 2.5, p. 50):**
- Group fairness metrics
- Mitigation strategies
- Designed for integration into supervised learning tasks

**Where it falls short (Source: Chapter 2.5, p. 50):**

> *"The need for coding knowledge again restricts access for non-specialists."*

> *"Its functionality is narrower than AIF360's, focusing on classification and regression."*

> *"The absence of plain-language reporting features also makes Fairlearn's findings harder for non-technical stakeholders to interpret."*

### Aequitas (University of Chicago)

**What it covers (Source: Chapter 2.5, p. 49–50):**
- Group fairness metrics through a dashboard interface
- Accessible statistical comparisons
- Deployed in criminal justice and municipal services

**Where it falls short (Source: Chapter 2.5, p. 49–50):**

> *"Aequitas has a limited scope; its metrics focus on group-level disparities, such as statistical parity, and it lacks seamless integration into machine learning development pipelines."*

> *"Compared to AIF360, Aequitas has a smaller user base and less assured long-term sustainability."*

---

## FDK's Unique Strengths

As articulated in the manuscript (Source: Chapter 4.4, p. 74; Chapter 4.5, p. 76–77):

| Strength | Description |
|----------|-------------|
| **Governance-first workflow** | Fairness testing embedded in decision-making, not just technical metrics |
| **No-code accessibility** | Usable by clinicians, educators, HR managers, and regulators without programming |
| **Domain modularity** | Seven domain-specific APIs (Healthcare, Justice, Finance, Education, Hiring, Business, Governance) |
| **Plain-language reporting** | Dual outputs: professional JSON reports + public-facing plain-language summaries |
| **Audit-first governance** | Evidence packs, versioned reports, and reproducible audit trails |
| **Complementary positioning** | Designed as a front-end triage layer for AIF360, Fairlearn, and other specialist toolkits |

---

## FDK as Complementary, Not Competitive

**Direct statement (Source: Chapter 4.4, p. 74):**

> *"FDK™ is designed to expand, not replace, the reach of existing fairness libraries. Its main role is to serve as a front-end and translation layer, so non-specialists can perform initial fairness checks before expert analysis."*

**Interoperability (Source: Chapter 4.4, p. 74):**

> *"FDK™ ensures compatibility with specialist pipelines through standardized data formats, exportable configurations, API connectors, and reproducible seed states. These features ensure that fairness tests conducted in FDK™ can be seamlessly transferred into AIF360, causal inference frameworks, or MLOps environments for in-depth analysis."*

**Division of labor (Source: Chapter 4.4, p. 75):**

> *"Causal inference, policy-impact simulations, advanced statistical corrections, and legal compliance assessments require specialist expertise. FDK™ does not perform these analyses; instead, it flags scenarios that warrant further specialist review."*

---

## Summary

| Framework | Best for | Limitations |
|-----------|----------|-------------|
| **AIF360** | Researchers needing comprehensive metric catalog | Requires coding; no plain-language; no causal/individual fairness |
| **Fairlearn** | ML engineers integrating fairness into pipelines | Requires coding; narrow scope; no plain-language |
| **Aequitas** | Policy analysts needing dashboard visualization | Group metrics only; limited integration |
| **FDK Toolkit** | Non-technical professionals needing governance-ready audits | Triage-level; defers deep analysis to specialist tools |

---

## References

- Bellamy, R.K.E., et al. (2019). AI Fairness 360. *IBM Journal of Research and Development*.
- Bird, S., et al. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. *Microsoft Research*.
- Saleiro, P., et al. (2018). Aequitas: A bias and fairness audit toolkit. *arXiv:1811.05577*.
- Tavakoli, H. (2026). *The AI Fairness Diagnostic Kit*. Chapter 2.5, 2.6, 4.4, Appendix D.
