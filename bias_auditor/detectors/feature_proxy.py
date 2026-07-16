"""
Feature Proxy Detector.

Detects features that may serve as proxies for protected attributes.
"""

import numpy as np
import pandas as pd
from scipy import stats

from bias_auditor.core.config import AuditConfig
from bias_auditor.core.report import BiasCategory, BiasFindings, BiasSeverity
from bias_auditor.metrics.fairness import chi_squared_test, mutual_information


class FeatureProxyDetector:
    """
    Detector for proxy variables that encode protected attributes.

    Checks for:
    - High correlation with protected attributes
    - High mutual information with protected attributes
    - Known proxy patterns (zip code, name patterns, etc.)
    """

    # Known proxy patterns - features commonly associated with protected attributes
    KNOWN_PROXY_PATTERNS = {
        "location": ["zip", "postal", "address", "neighborhood", "district", "county", "region"],
        "name": ["name", "surname", "first_name", "last_name"],
        "education": ["school", "university", "college", "degree"],
        "employment": ["employer", "company", "job_title", "occupation"],
        "financial": ["income", "salary", "credit", "wealth", "assets"],
    }

    def __init__(self, config: AuditConfig):
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, data: pd.DataFrame) -> list[BiasFindings]:
        """
        Detect proxy variables in the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to analyze.

        Returns
        -------
        list[BiasFindings]
            List of detected bias findings.
        """
        findings = []

        # Get non-protected feature columns
        feature_columns = [
            col
            for col in data.columns
            if col not in self.config.protected_attributes
            and col != self.config.target_column
            and col not in self.config.exclude_columns
        ]

        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue

            for feature in feature_columns:
                # Check correlation/association
                findings.extend(self._check_association(data, feature, attr))

                # Check mutual information
                findings.extend(self._check_mutual_information(data, feature, attr))

            # Check known proxy patterns
            findings.extend(self._check_known_patterns(data, attr, feature_columns))

        return findings

    def _check_association(
        self, data: pd.DataFrame, feature: str, protected_attr: str
    ) -> list[BiasFindings]:
        """Check correlation or association between feature and protected attribute."""
        findings = []

        # Handle missing values
        valid_mask = data[[feature, protected_attr]].notna().all(axis=1)
        if valid_mask.sum() < 30:
            return findings

        feature_data = data.loc[valid_mask, feature]
        protected_data = data.loc[valid_mask, protected_attr]

        # Determine data types and appropriate test
        feature_is_numeric = feature_data.dtype in ["float64", "float32", "int64", "int32"]
        protected_is_numeric = protected_data.dtype in ["float64", "float32", "int64", "int32"]

        correlation = None
        p_value = None
        test_type = None

        try:
            if feature_is_numeric and protected_is_numeric:
                # Pearson correlation
                correlation, p_value = stats.pearsonr(feature_data, protected_data)
                test_type = "pearson"
            elif feature_is_numeric and not protected_is_numeric:
                # Point-biserial or ANOVA-based correlation
                groups = [feature_data[protected_data == g].values for g in protected_data.unique()]
                groups = [g for g in groups if len(g) > 1]

                if len(groups) >= 2:
                    if len(protected_data.unique()) == 2:
                        # Point-biserial
                        correlation, p_value = stats.pointbiserialr(
                            pd.factorize(protected_data)[0], feature_data
                        )
                        test_type = "point_biserial"
                    else:
                        # ANOVA F-test -> convert to correlation-like measure
                        f_stat, p_value = stats.f_oneway(*groups)
                        # Eta-squared as effect size
                        ss_between = sum(
                            len(g) * (np.mean(g) - np.mean(feature_data)) ** 2 for g in groups
                        )
                        ss_total = np.sum((feature_data - np.mean(feature_data)) ** 2)
                        correlation = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0
                        test_type = "eta"
            else:
                # Chi-squared test for categorical variables
                chi_result = chi_squared_test(data.loc[valid_mask], feature, protected_attr)
                correlation = chi_result["cramers_v"]
                p_value = chi_result["p_value"]
                test_type = "cramers_v"
        except Exception:
            return findings

        if correlation is None:
            return findings

        abs_corr = abs(correlation)

        if abs_corr > self.thresholds.proxy_correlation_critical:
            findings.append(
                BiasFindings(
                    category=BiasCategory.FEATURE_PROXY,
                    severity=BiasSeverity.CRITICAL,
                    title=f"Strong proxy detected: '{feature}' for '{protected_attr}'",
                    description=(
                        f"Feature '{feature}' has a strong association with protected "
                        f"attribute '{protected_attr}' ({test_type}={correlation:.3f}). "
                        f"Using this feature may indirectly discriminate based on "
                        f"'{protected_attr}' even if the protected attribute is removed."
                    ),
                    affected_attribute=protected_attr,
                    affected_groups=[feature],
                    metrics={
                        "correlation": correlation,
                        "abs_correlation": abs_corr,
                        "p_value": p_value,
                        "test_type": test_type,
                        "threshold": self.thresholds.proxy_correlation_critical,
                    },
                    remediation_suggestions=[
                        f"Consider removing '{feature}' from training features",
                        "Apply fairness-aware feature selection",
                        f"Transform '{feature}' to reduce correlation with '{protected_attr}'",
                        "Use adversarial debiasing during training",
                        "Apply disparate impact remover preprocessing",
                    ],
                    evidence={
                        "n_samples": int(valid_mask.sum()),
                    },
                )
            )
        elif abs_corr > self.thresholds.proxy_correlation_warning:
            findings.append(
                BiasFindings(
                    category=BiasCategory.FEATURE_PROXY,
                    severity=BiasSeverity.WARNING,
                    title=f"Moderate proxy detected: '{feature}' for '{protected_attr}'",
                    description=(
                        f"Feature '{feature}' has moderate association with "
                        f"'{protected_attr}' ({test_type}={correlation:.3f}). "
                        f"This could introduce indirect discrimination."
                    ),
                    affected_attribute=protected_attr,
                    affected_groups=[feature],
                    metrics={
                        "correlation": correlation,
                        "abs_correlation": abs_corr,
                        "p_value": p_value,
                        "test_type": test_type,
                    },
                    remediation_suggestions=[
                        "Monitor model decisions related to this feature",
                        "Consider fairness testing with and without this feature",
                    ],
                    evidence={
                        "n_samples": int(valid_mask.sum()),
                    },
                )
            )

        return findings

    def _check_mutual_information(
        self, data: pd.DataFrame, feature: str, protected_attr: str
    ) -> list[BiasFindings]:
        """Check mutual information between feature and protected attribute."""
        findings = []

        try:
            mi = mutual_information(data, feature, protected_attr)
        except Exception:
            return findings

        if mi > self.thresholds.proxy_mutual_info_critical:
            findings.append(
                BiasFindings(
                    category=BiasCategory.FEATURE_PROXY,
                    severity=BiasSeverity.CRITICAL,
                    title=f"High mutual information: '{feature}' ↔ '{protected_attr}'",
                    description=(
                        f"Feature '{feature}' shares significant information with "
                        f"protected attribute '{protected_attr}' (MI={mi:.3f}). "
                        f"This feature can effectively predict group membership."
                    ),
                    affected_attribute=protected_attr,
                    affected_groups=[feature],
                    metrics={
                        "mutual_information": mi,
                        "threshold": self.thresholds.proxy_mutual_info_critical,
                    },
                    remediation_suggestions=[
                        f"Evaluate necessity of '{feature}' for prediction task",
                        "Use mutual information-based feature selection",
                        "Consider privacy-preserving transformations",
                    ],
                )
            )
        elif mi > self.thresholds.proxy_mutual_info_warning:
            findings.append(
                BiasFindings(
                    category=BiasCategory.FEATURE_PROXY,
                    severity=BiasSeverity.INFO,
                    title=f"Elevated mutual information: '{feature}' ↔ '{protected_attr}'",
                    description=(
                        f"Feature '{feature}' has elevated mutual information with "
                        f"'{protected_attr}' (MI={mi:.3f})."
                    ),
                    affected_attribute=protected_attr,
                    affected_groups=[feature],
                    metrics={
                        "mutual_information": mi,
                    },
                    remediation_suggestions=[
                        "Monitor for fairness implications",
                    ],
                )
            )

        return findings

    def _check_known_patterns(
        self, data: pd.DataFrame, protected_attr: str, feature_columns: list[str]
    ) -> list[BiasFindings]:
        """Check for known proxy patterns in feature names."""
        findings = []

        for category, patterns in self.KNOWN_PROXY_PATTERNS.items():
            for feature in feature_columns:
                feature_lower = feature.lower()

                for pattern in patterns:
                    if pattern in feature_lower:
                        # Check actual association
                        try:
                            mi = mutual_information(data, feature, protected_attr)
                        except Exception:
                            mi = None

                        if mi is not None and mi > 0.1:
                            findings.append(
                                BiasFindings(
                                    category=BiasCategory.FEATURE_PROXY,
                                    severity=BiasSeverity.WARNING,
                                    title=f"Known proxy pattern: '{feature}'",
                                    description=(
                                        f"Feature '{feature}' matches known proxy pattern "
                                        f"'{category}' and has association with '{protected_attr}' "
                                        f"(MI={mi:.3f}). {self._get_proxy_explanation(category)}"
                                    ),
                                    affected_attribute=protected_attr,
                                    affected_groups=[feature],
                                    metrics={
                                        "mutual_information": mi,
                                        "pattern_category": category,
                                        "matched_pattern": pattern,
                                    },
                                    remediation_suggestions=self._get_proxy_remediation(
                                        category, feature
                                    ),
                                )
                            )
                        break

        return findings

    def _get_proxy_explanation(self, category: str) -> str:
        """Get explanation for why a category is a potential proxy."""
        explanations = {
            "location": (
                "Location data often correlates with race/ethnicity due to "
                "historical segregation patterns."
            ),
            "name": ("Names can indicate gender, ethnicity, or national origin."),
            "education": (
                "Educational institutions may correlate with socioeconomic "
                "status and race due to systemic inequalities."
            ),
            "employment": (
                "Employer and job information can correlate with protected "
                "attributes due to occupational segregation."
            ),
            "financial": (
                "Financial features often correlate with race and gender "
                "due to historical wealth gaps and pay disparities."
            ),
        }
        return explanations.get(category, "")

    def _get_proxy_remediation(self, category: str, feature: str) -> list[str]:
        """Get category-specific remediation suggestions."""
        base_suggestions = [
            f"Evaluate whether '{feature}' is necessary for the prediction task",
            "Test model fairness with and without this feature",
        ]

        category_suggestions = {
            "location": [
                "Consider using region-level aggregation instead of precise location",
                "Apply geographic fairness constraints",
            ],
            "name": [
                "Remove name fields unless essential for the task",
                "Use name only for deduplication, not as a feature",
            ],
            "education": [
                "Consider using degree level instead of specific institutions",
                "Apply disparate impact analysis on education-related features",
            ],
            "employment": [
                "Use job category instead of specific employer names",
                "Apply industry-level aggregation",
            ],
            "financial": [
                "Consider relative rather than absolute financial measures",
                "Apply binning or quantile normalization",
            ],
        }

        return base_suggestions + category_suggestions.get(category, [])
