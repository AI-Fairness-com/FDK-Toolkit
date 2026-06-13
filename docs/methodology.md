# Methodology: Fairness Metrics in the FDK Toolkit

This document provides the mathematical and methodological foundations for the fairness metrics implemented in the FDK Toolkit. All formulas and explanations are sourced directly from the FDK manuscript (Tavakoli, 2026).

## Table of Contents

1. [Subgroup Disparity Calculation](#subgroup-disparity-calculation)
2. [Calibration Metrics](#calibration-metrics)
3. [Robustness Testing](#robustness-testing)
4. [Composite Bias Score](#composite-bias-score)
5. [Metric Trade-offs and Impossibility Results](#metric-trade-offs-and-impossibility-results)
6. [References to Source](#references-to-source)

---

## Subgroup Disparity Calculation

### Demographic Parity (Statistical Parity)

**Formula (Source: Chapter 7.3.1, p. 101):**

\[SPD = SR_{unpriv} - SR_{priv}\]

Where:
- \(SR_g\) = selection rate for group \(g\)
- \(SR_g = \frac{\#\text{predicted positive in group } g}{N_g}\)

**Plain-language analogy (Source: Chapter 7.3.1, p. 101):**
> "Imagine two classrooms sitting the same exam. If both classrooms are equally capable, you would expect their pass rates to be similar. A consistent disparity in pass rates between classrooms suggests a potential bias within the assessment system."

### Equal Opportunity

**Formula (Source: Chapter 7.3.2, p. 102–103):**

\[TPR_{priv} = TPR_{unpriv}\]

Where:
- \(TPR_g = \frac{TP_g}{TP_g + FN_g}\) (true positive rate)

**Plain-language analogy (Source: Chapter 7.3.2, p. 102):**
> "Think of a medical test. If two patients both have the disease, they should have the same chance of being correctly diagnosed, no matter their group."

### Equalized Odds

**Formula (Source: Chapter 7.3.3, p. 103):**

\[TPR_{priv} = TPR_{unpriv} \quad \text{and} \quad FPR_{priv} = FPR_{unpriv}\]

Where:
- \(TPR_g = \frac{TP_g}{TP_g + FN_g}\)
- \(FPR_g = \frac{FP_g}{FP_g + TN_g}\)

**Plain-language analogy (Source: Chapter 7.3.3, p. 103):**
> "Picture a referee in a football match. A fair referee makes accurate calls for both teams, not only in spotting goals (true positives) but also in avoiding mistakes like wrongly disallowing a goal (false negatives) or awarding one that never happened (false positives)."

### Predictive Parity (Predictive Value Equality)

**Formula (Source: Chapter 7.3.4, p. 104):**

\[PPV_g = \frac{TP_g}{TP_g + FP_g}\]

Predictive parity holds when \(PPV\) is equal across groups.

**Plain-language analogy (Source: Chapter 7.3.4, p. 104):**
> "Imagine two weather forecasters who both predict rain. If one forecaster is usually correct for City A but often wrong for City B, their predictions are less reliable. Predictive parity requires that when the system predicts a positive outcome, its correctness is equally reliable across groups."

---

## Calibration Metrics

### Slice AUC Difference (AUC-over-Threshold Disparity)

**Formula (Source: Chapter 7.6.2, p. 113–114):**

\[AUC_{slice} = \frac{\int TPR_g(t) \, dFPR_g(t)}{\int TPR_{ref}(t) \, dFPR_{ref}(t)}\]

Where \(TPR_g(t)\) is the true positive rate for group \(g\) at threshold \(t\), compared with a reference group.

**Plain-language analogy (Source: Chapter 7.6.2, p. 113):**
> "Think of a race where competitors are compared not only at the finish line but at every checkpoint. A system may look fair on average yet reveal hidden gaps at particular decision thresholds. This metric checks for disparities that appear only at specific operating points."

**Why it matters (Source: Chapter 7.6.2, p. 114):**
> "Slice AUC difference uncovers disparities that average metrics miss. A model might appear balanced overall but fail badly for some groups at specific cut-offs, such as loan approval thresholds or medical test sensitivities."

### Worst-Group Calibration Gap

**Formula (Source: Appendix C, p. 390):**

\[WorstCalGap = \max_{g \in G} \sup_{s \in [0,1]} \left| \Pr(Y = 1 \mid \hat{p} = s, g) - s \right|\]

**Plain-language analogy (Source: Appendix C, p. 390):**
> "The most miscalibrated thermometer among a set."

---

## Robustness Testing

### Worst-Group Accuracy

**Formula (Source: Chapter 7.6.1, p. 113):**

\[WorstGroupAcc = \min_{g \in G} Accuracy_g\]

Where:
- \(Accuracy_g = \frac{TP_g + TN_g}{N_g}\)

**Plain-language analogy (Source: Chapter 7.6.1, p. 113):**
> "Imagine assessing a school not by the average grades of all classrooms but by the lowest-performing one. A system is only as fair as the experience of its most disadvantaged group."

### Validation-Holdout Robustness Score

**Formula (Source: Appendix C, p. 390):**

\[Robustness = 1 - \frac{|m_{val} - m_{train}|}{|m_{train}| + \tau}\]

**Why it matters (Source: Appendix C, p. 390):**
> "Detects fairness overfitting; ensures generalization."

---

## Composite Bias Score

**Definition (Source: Appendix F, p. 406):**

The Composite Bias Score is a weighted aggregate of multiple fairness metrics. It is calculated during the baseline bias detection phase of the audit pipeline.

**Interpretation thresholds (Source: Chapter 14.5, p. 194):**
- Values \(< 0.05\): generally fair performance (LOW_BIAS range)
- Values \(> 0.10\): significant disparities (HIGH_BIAS range)

**Example from COMPAS dataset (Source: Appendix F, p. 406):**
- Initial Bias Score: 0.2213 (HIGH_BIAS)
- Final Bias Score after mitigation: 0.1591
- Bias Reduction: 28.1%

---

## Metric Trade-offs and Impossibility Results

### Core Principle (Source: Chapter 7.7, p. 114)

> "Fairness metrics provide powerful tools for diagnosing bias, but they rarely point in the same direction. In practice, different metrics can conflict with one another, and satisfying one may worsen another."

### Trade-off Examples by Domain

**Healthcare (Source: Chapter 14.7, p. 195–196):**

| Trade-off | Explanation |
|-----------|-------------|
| Demographic parity vs. clinical necessity | Forcing equal ICU admission prediction rates across groups may conflict with genuine differences in clinical risk. |
| Equal opportunity vs. base rate differences | If actual ICU admission rates differ substantially between groups, equal true positive rates may be clinically inappropriate. |
| Individual fairness vs. group fairness | Adjusting outcomes to balance group-level metrics may treat two similar patients from different groups differently. |

**Finance (Source: Chapter 13.7, p. 182):**

| Trade-off | Explanation |
|-----------|-------------|
| Demographic parity vs. accuracy | Forcing equal approval rates across groups typically increases false positives for lower-risk groups or false negatives for higher-risk groups. |
| Equalized odds vs. calibration | A model calibrated to predict actual risk accurately for all groups may violate equalized odds if base rates differ. |

**General statement (Source: Chapter 17.1, p. 188):**

> "These metrics often conflict. Achieving equal opportunity may harm predictive parity ... such tensions, called impossibility results, show fairness is multi-dimensional, not a design flaw. The FDK makes these trade-offs clear, so clinicians and policymakers can judge ethical priorities directly."

---

## References to Source

All content in this document is derived from:

| Topic | Source |
|-------|--------|
| Group fairness metrics (SPD, Equal Opportunity, Equalized Odds, Predictive Parity) | Chapter 7.3, pp. 101–104 |
| Individual fairness (Consistency, Counterfactual) | Chapter 7.4, pp. 105–106 |
| Causal fairness | Chapter 7.5, pp. 106–112 |
| Robustness metrics (Worst-Group Accuracy, Slice AUC) | Chapter 7.6, pp. 113–114 |
| Metric trade-offs | Chapter 7.7, p. 114; Chapter 13.7, p. 182; Chapter 14.7, pp. 195–196; Chapter 17.1, p. 188 |
| Composite Bias Score | Chapter 14.5, p. 194; Appendix F, p. 406 |
| Validation-Holdout Robustness Score | Appendix C, p. 390 |
| Worst-Group Calibration Gap | Appendix C, p. 390 |

**Full reference:** Tavakoli, H. (2026). *The AI Fairness Diagnostic Kit: From Principle to Practice in No-Code AI Fairness Auditing*. Available at: https://github.com/AI-Fairness-com/FDK-Toolkit