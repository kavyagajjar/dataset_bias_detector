"""
Fairness and bias metrics for dataset auditing.

This module provides statistical measures for quantifying bias and fairness
in datasets, including representation metrics, label disparity metrics,
and information-theoretic measures.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr


def statistical_parity_difference(
    data: pd.DataFrame,
    protected_attr: str,
    target_column: str,
    positive_label: Any = 1,
    privileged_group: Optional[Any] = None,
    unprivileged_group: Optional[Any] = None,
) -> dict[str, float]:
    """
    Calculate Statistical Parity Difference (SPD).

    SPD = P(Y=1|A=unprivileged) - P(Y=1|A=privileged)

    A value of 0 indicates perfect parity. Negative values indicate
    the unprivileged group has lower positive outcome rates.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    protected_attr : str
        Column name of the protected attribute.
    target_column : str
        Column name of the target/label.
    positive_label : Any
        The value considered positive outcome.
    privileged_group : Any, optional
        Value of the privileged group. If None, uses group with highest rate.
    unprivileged_group : Any, optional
        Value of the unprivileged group. If None, uses group with lowest rate.

    Returns
    -------
    dict
        Dictionary with SPD value and group details.
    """
    # Calculate positive rate for each group
    group_rates = data.groupby(protected_attr)[target_column].apply(
        lambda x: (x == positive_label).mean()
    )

    if privileged_group is None:
        privileged_group = group_rates.idxmax()
    if unprivileged_group is None:
        unprivileged_group = group_rates.idxmin()

    priv_rate = group_rates.get(privileged_group, 0)
    unpriv_rate = group_rates.get(unprivileged_group, 0)

    spd = unpriv_rate - priv_rate

    return {
        "spd": spd,
        "privileged_group": privileged_group,
        "privileged_rate": priv_rate,
        "unprivileged_group": unprivileged_group,
        "unprivileged_rate": unpriv_rate,
        "all_rates": group_rates.to_dict(),
    }


def disparate_impact_ratio(
    data: pd.DataFrame,
    protected_attr: str,
    target_column: str,
    positive_label: Any = 1,
    privileged_group: Optional[Any] = None,
    unprivileged_group: Optional[Any] = None,
) -> dict[str, float]:
    """
    Calculate Disparate Impact Ratio (DIR).

    DIR = P(Y=1|A=unprivileged) / P(Y=1|A=privileged)

    The 80% rule (DIR < 0.8) is commonly used to identify adverse impact.
    A value of 1 indicates perfect parity.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    protected_attr : str
        Column name of the protected attribute.
    target_column : str
        Column name of the target/label.
    positive_label : Any
        The value considered positive outcome.
    privileged_group : Any, optional
        Value of the privileged group.
    unprivileged_group : Any, optional
        Value of the unprivileged group.

    Returns
    -------
    dict
        Dictionary with DIR value and group details.
    """
    spd_result = statistical_parity_difference(
        data, protected_attr, target_column, positive_label,
        privileged_group, unprivileged_group
    )

    priv_rate = spd_result["privileged_rate"]
    unpriv_rate = spd_result["unprivileged_rate"]

    # Avoid division by zero
    if priv_rate == 0:
        dir_value = float('inf') if unpriv_rate > 0 else 1.0
    else:
        dir_value = unpriv_rate / priv_rate

    return {
        "dir": float(dir_value),
        "passes_80_percent_rule": bool(dir_value >= 0.8),
        **spd_result,
    }


def class_imbalance_ratio(
    data: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Calculate class imbalance ratio.

    Ratio = count(majority) / count(minority)

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    column : str
        Column to check for imbalance.

    Returns
    -------
    dict
        Dictionary with imbalance metrics.
    """
    value_counts = data[column].value_counts()

    if len(value_counts) < 2:
        return {
            "ratio": 1.0,
            "majority_class": value_counts.index[0] if len(value_counts) > 0 else None,
            "minority_class": None,
            "n_classes": len(value_counts),
            "distribution": value_counts.to_dict(),
        }

    majority_class = value_counts.index[0]
    minority_class = value_counts.index[-1]

    ratio = value_counts.iloc[0] / value_counts.iloc[-1]

    return {
        "ratio": ratio,
        "majority_class": majority_class,
        "majority_count": int(value_counts.iloc[0]),
        "minority_class": minority_class,
        "minority_count": int(value_counts.iloc[-1]),
        "n_classes": len(value_counts),
        "distribution": value_counts.to_dict(),
        "normalized_distribution": (value_counts / len(data)).to_dict(),
    }


def normalized_entropy(
    data: pd.DataFrame,
    column: str,
) -> float:
    """
    Calculate normalized entropy of a distribution.

    H_norm = -sum(p_i * log(p_i)) / log(n)

    Returns a value between 0 and 1:
    - 0: maximum imbalance (all in one class)
    - 1: perfect balance (uniform distribution)

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    column : str
        Column to calculate entropy for.

    Returns
    -------
    float
        Normalized entropy value.
    """
    value_counts = data[column].value_counts()
    n_classes = len(value_counts)

    if n_classes <= 1:
        return 0.0

    proportions = value_counts / len(data)

    # Calculate entropy
    entropy = -np.sum(proportions * np.log(proportions))

    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log(n_classes)

    return entropy / max_entropy


def kl_divergence(
    observed: dict[str, float],
    expected: dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate KL divergence between observed and expected distributions.

    D_KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Parameters
    ----------
    observed : dict
        Observed distribution {category: proportion}.
    expected : dict
        Expected/reference distribution {category: proportion}.
    epsilon : float
        Small value to avoid log(0).

    Returns
    -------
    float
        KL divergence value.
    """
    # Align distributions
    all_keys = set(observed.keys()) | set(expected.keys())

    p = np.array([observed.get(k, epsilon) for k in all_keys])
    q = np.array([expected.get(k, epsilon) for k in all_keys])

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Calculate KL divergence
    return float(np.sum(rel_entr(p, q)))


def group_label_rates(
    data: pd.DataFrame,
    protected_attr: str,
    target_column: str,
    positive_label: Any = 1,
) -> dict[str, dict[str, float]]:
    """
    Calculate label rates for each group in a protected attribute.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    protected_attr : str
        Column name of the protected attribute.
    target_column : str
        Column name of the target/label.
    positive_label : Any
        The value considered positive outcome.

    Returns
    -------
    dict
        Dictionary with rates and statistics for each group.
    """
    results = {}

    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group]

        positive_count = (group_data[target_column] == positive_label).sum()
        total_count = len(group_data)
        rate = positive_count / total_count if total_count > 0 else 0

        results[str(group)] = {
            "positive_rate": rate,
            "positive_count": int(positive_count),
            "total_count": int(total_count),
            "proportion_of_dataset": total_count / len(data),
        }

    # Calculate disparity metrics
    rates = [r["positive_rate"] for r in results.values()]
    if rates:
        results["_summary"] = {
            "max_rate": max(rates),
            "min_rate": min(rates),
            "rate_difference": max(rates) - min(rates),
            "rate_ratio": min(rates) / max(rates) if max(rates) > 0 else 0,
        }

    return results


def conditional_demographic_parity(
    data: pd.DataFrame,
    protected_attr: str,
    target_column: str,
    conditioning_features: list[str],
    positive_label: Any = 1,
) -> dict[str, Any]:
    """
    Calculate Conditional Demographic Parity.

    Checks if demographic parity holds within strata defined by
    conditioning features (controlling for legitimate factors).

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    protected_attr : str
        Column name of the protected attribute.
    target_column : str
        Column name of the target/label.
    conditioning_features : list[str]
        Features to condition on (legitimate factors).
    positive_label : Any
        The value considered positive outcome.

    Returns
    -------
    dict
        Dictionary with parity metrics for each stratum.
    """
    results = {
        "strata": {},
        "overall_disparity": 0.0,
    }

    # Group by conditioning features
    if not conditioning_features:
        return results

    strata = data.groupby(conditioning_features)

    disparities = []

    for stratum_key, stratum_data in strata:
        if len(stratum_data) < 10:  # Skip small strata
            continue

        stratum_name = str(stratum_key)

        # Calculate SPD within stratum
        try:
            spd_result = statistical_parity_difference(
                stratum_data, protected_attr, target_column, positive_label
            )

            results["strata"][stratum_name] = {
                "spd": spd_result["spd"],
                "n_samples": len(stratum_data),
                "group_rates": spd_result["all_rates"],
            }

            disparities.append(abs(spd_result["spd"]))
        except Exception:
            continue

    if disparities:
        results["overall_disparity"] = np.mean(disparities)
        results["max_disparity"] = max(disparities)
        results["n_strata_analyzed"] = len(disparities)

    return results


def equalized_odds_difference(
    data: pd.DataFrame,
    protected_attr: str,
    target_column: str,
    prediction_column: str,
    positive_label: Any = 1,
) -> dict[str, float]:
    """
    Calculate Equalized Odds Difference.

    Measures difference in True Positive Rate (TPR) and False Positive Rate (FPR)
    between protected groups.

    Note: This requires both actual labels AND predictions, so it's typically
    used for model evaluation, not raw data auditing. Included for completeness.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset with both labels and predictions.
    protected_attr : str
        Column name of the protected attribute.
    target_column : str
        Column name of the actual target/label.
    prediction_column : str
        Column name of the model predictions.
    positive_label : Any
        The value considered positive outcome.

    Returns
    -------
    dict
        Dictionary with TPR and FPR differences by group.
    """
    results = {"groups": {}}

    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group]

        # True positives, false positives, etc.
        actual_pos = group_data[target_column] == positive_label
        actual_neg = ~actual_pos
        pred_pos = group_data[prediction_column] == positive_label

        tp = (actual_pos & pred_pos).sum()
        fp = (actual_neg & pred_pos).sum()
        fn = (actual_pos & ~pred_pos).sum()
        tn = (actual_neg & ~pred_pos).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results["groups"][str(group)] = {
            "tpr": tpr,
            "fpr": fpr,
            "n_samples": len(group_data),
        }

    # Calculate differences
    tprs = [g["tpr"] for g in results["groups"].values()]
    fprs = [g["fpr"] for g in results["groups"].values()]

    results["tpr_difference"] = max(tprs) - min(tprs) if tprs else 0
    results["fpr_difference"] = max(fprs) - min(fprs) if fprs else 0
    results["equalized_odds_difference"] = max(
        results["tpr_difference"],
        results["fpr_difference"]
    )

    return results


def chi_squared_test(
    data: pd.DataFrame,
    attr1: str,
    attr2: str,
) -> dict[str, float]:
    """
    Perform chi-squared test of independence between two categorical variables.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    attr1 : str
        First categorical column.
    attr2 : str
        Second categorical column.

    Returns
    -------
    dict
        Test statistic, p-value, and interpretation.
    """
    contingency = pd.crosstab(data[attr1], data[attr2])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V for effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v,
        "is_significant": p_value < 0.05,
        "effect_size": "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large",
    }


def mutual_information(
    data: pd.DataFrame,
    feature: str,
    protected_attr: str,
    n_bins: int = 10,
) -> float:
    """
    Calculate mutual information between a feature and protected attribute.

    Higher values indicate the feature contains more information about
    the protected attribute (potential proxy).

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    feature : str
        Feature column to test.
    protected_attr : str
        Protected attribute column.
    n_bins : int
        Number of bins for continuous features.

    Returns
    -------
    float
        Normalized mutual information (0-1).
    """
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.preprocessing import LabelEncoder

    # Handle missing values
    mask = data[[feature, protected_attr]].notna().all(axis=1)
    feature_data = data.loc[mask, feature].copy()
    protected_data = data.loc[mask, protected_attr].copy()

    if len(feature_data) == 0:
        return 0.0

    # Discretize continuous features
    if feature_data.dtype in ['float64', 'float32', 'int64', 'int32']:
        try:
            feature_data = pd.cut(feature_data, bins=n_bins, labels=False)
        except Exception:
            return 0.0

    # Encode categorical features
    le_feature = LabelEncoder()
    le_protected = LabelEncoder()

    try:
        feature_encoded = le_feature.fit_transform(feature_data.astype(str))
        protected_encoded = le_protected.fit_transform(protected_data.astype(str))
    except Exception:
        return 0.0

    return float(normalized_mutual_info_score(feature_encoded, protected_encoded))
