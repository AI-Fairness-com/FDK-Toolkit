"""
Intelligent target-column selection, shared by FDK.py (local dev entry
point), app.py (production entry point), and every domain blueprint
(Justice, Business, Education, Finance, Health, Hiring, Governance).

Deliberately has NO Flask dependency and imports no domain blueprint --
that's the whole point of this being a separate file. Previously this
logic lived duplicated in BOTH FDK.py and app.py independently, which had
already caused real drift between them (app.py had picked up a
prediction-indicator-term exclusion and safer no-blind-fallback behavior
that FDK.py's copy never got). This file uses app.py's more advanced
versions as the single source of truth going forward, plus
_suggest_target_column, which only FDK.py needed and app.py never had.
"""

# ================================================================
# UNIVERSAL INTELLIGENT TARGET SELECTION SYSTEM
# ================================================================

def is_binary_column(series):
    """Check if a pandas Series contains binary 0/1 values"""
    try:
        unique_vals = series.dropna().unique()
        return len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})
    except:
        return False

def find_first_binary_column(columns, df):
    """Find first binary (0/1) column in dataset"""
    for col in columns:
        if is_binary_column(df[col]):
            return col
    return None

def find_bias_corrected_columns(columns, df):
    """Find columns that indicate bias-corrected targets"""
    bias_corrected_patterns = [
        'svm_fair_target', 'biasclean_target', 'fair_target', 'corrected_target',
        'debiased_target', 'mitigated_target', 'fairness_corrected', 'post_correction'
    ]
    for pattern in bias_corrected_patterns:
        for col in columns:
            if pattern in col.lower():
                if is_binary_column(df[col]):
                    return col
    return None

def detect_domain_from_columns(columns):
    """Auto-detect which of 7 FDK domains based on column name patterns"""
    domain_patterns = {
        'justice': ['recid', 'bail', 'sentenc', 'parole', 'defendant'],
        'health': ['mortality', 'readmission', 'readmit', 'complication', 'patient', 'diagnos'],
        'education': ['admission', 'dropout', 'graduation', 'student'],
        'hiring': ['hired', 'selected', 'offer_accepted', 'callback', 'applicant'],
        'finance': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk', 'loan'],
        'business': ['churn', 'conversion', 'purchase', 'customer'],
        'governance': ['approved', 'granted', 'permitted', 'constituent'],
    }
    col_string = ' '.join(str(c).lower() for c in columns)
    for domain, keywords in domain_patterns.items():
        if any(keyword in col_string for keyword in keywords):
            return domain
    return None

def general_intelligent_selection(df, test_type):
    """General intelligent target selection when no domain rules match"""
    columns = df.columns.tolist()

    if test_type == 'post_implementation':
        bias_corrected = find_bias_corrected_columns(columns, df)
        if bias_corrected:
            return bias_corrected

    target_keywords = ['target', 'outcome', 'label', 'y_true', 'decision', 'result',
                      'callback', 'hired', 'selected', 'approved', 'default',
                      'churn', 'admission', 'recid', 'mortality']

    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in target_keywords):
            if test_type in ['pre_implementation', 'post_implementation']:
                if is_binary_column(df[col]):
                    return col
            else:
                return col

    if test_type in ['pre_implementation', 'post_implementation']:
        semantic_patterns = ['outcome', 'result', 'status', 'decision', 'flag']
        for pattern in semantic_patterns:
            for col in columns:
                col_lower = str(col).lower()
                if pattern in col_lower and is_binary_column(df[col]):
                    return col

    return None

def intelligent_target_selection(df, test_type, domain_hint=None):
    """Intelligently select target column based on test type and domain"""
    columns = df.columns.tolist()
    domain = domain_hint or detect_domain_from_columns(columns)

    domain_rules = {
        'justice': {
            'pre_implementation': ['two_year_recid', 'is_recid', 'recidivism'],
            'post_implementation': ['two_year_recid', 'is_recid', 'recidivism'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'health': {
            'pre_implementation': ['mortality', 'readmission', 'readmit', 'complication'],
            'post_implementation': ['mortality', 'readmission', 'readmit', 'complication'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'education': {
            'pre_implementation': ['admission', 'dropout', 'graduation'],
            'post_implementation': ['admission', 'dropout', 'graduation'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'hiring': {
            'pre_implementation': ['hired', 'selected', 'offer_accepted', 'callback'],
            'post_implementation': ['hired', 'selected', 'offer_accepted', 'callback'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'finance': {
            'pre_implementation': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk'],
            'post_implementation': ['default', 'approved', 'loan_status', 'creditrisk', 'credit_risk'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'business': {
            'pre_implementation': ['churn', 'conversion', 'purchase', 'subscribed'],
            'post_implementation': ['churn', 'conversion', 'purchase', 'subscribed'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        },
        'governance': {
            'pre_implementation': ['approved', 'granted', 'permitted', 'pubcov'],
            'post_implementation': ['approved', 'granted', 'permitted', 'pubcov'],
            'fallback': lambda cols: None  # no semantic match -- let caller's own detection take over
        }
    }

    if domain in domain_rules:
        priority_list = domain_rules[domain].get(test_type, [])
        # Column names containing these terms are predictions/recommendations,
        # never genuine ground truth -- exclude them even when they also
        # match a domain keyword (e.g. "predicted_recidivism" contains
        # "recidivism" but is a model output, not the real outcome).
        prediction_indicator_terms = ['predicted', 'prediction', 'recommend', 'recommended',
                                       'forecast', 'estimate', 'model_output', 'model_score']
        for col_pattern in priority_list:
            for actual_col in columns:
                col_lower = actual_col.lower()
                if col_pattern.lower() in col_lower:
                    if any(term in col_lower for term in prediction_indicator_terms):
                        continue
                    if test_type in ['pre_implementation', 'post_implementation']:
                        if is_binary_column(df[actual_col]):
                            return actual_col
                    else:
                        return actual_col

        fallback_func = domain_rules[domain].get('fallback')
        if fallback_func:
            fallback_result = fallback_func(columns)
            if fallback_result:
                return fallback_result

    return general_intelligent_selection(df, test_type)

def _suggest_target_column(columns):
    """
    Conservative heuristic: returns a suggested target column if found, else last column.
    Only used by FDK.py's /api/detect-columns route -- app.py never had an
    equivalent, so this one function comes from FDK.py's original rather
    than app.py.
    """
    if not columns:
        return None

    preferred = [
        "target", "label", "y", "y_true", "outcome", "class", "decision",
        "ground_truth", "gt", "approved", "hired", "hire", "selected", "callback"
    ]
    lower_map = {c.lower(): c for c in columns if isinstance(c, str)}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    return columns[-1]
