"""
Label Bias Detector.

Detects systematic differences in labeling across protected groups.
"""

import numpy as np
import pandas as pd
from scipy import stats

from bias_auditor.core.config import AuditConfig
from bias_auditor.core.report import BiasCategory, BiasFindings, BiasSeverity
from bias_auditor.metrics.fairness import (
    disparate_impact_ratio,
    group_label_rates,
    statistical_parity_difference,
)


class LabelBiasDetector:
    """
    Detector for label/outcome biases in datasets.

    Checks for:
    - Disparate impact (80% rule)
    - Statistical parity differences
    - Label rate disparities across groups
    - Conditional parity violations
    """

    def __init__(self, config: AuditConfig):
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, data: pd.DataFrame) -> list[BiasFindings]:
        """
        Detect label biases in the dataset.

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

        if not self.config.target_column:
            return findings

        if self.config.target_column not in data.columns:
            return findings

        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue

            # Check disparate impact
            findings.extend(self._check_disparate_impact(data, attr))

            # Check statistical parity
            findings.extend(self._check_statistical_parity(data, attr))

            # Check for significant label rate differences
            findings.extend(self._check_label_rate_significance(data, attr))

        # Check intersectional label bias
        if self.config.compute_intersectional:
            findings.extend(self._check_intersectional_label_bias(data))

        return findings

    def _check_disparate_impact(
        self,
        data: pd.DataFrame,
        attr: str
    ) -> list[BiasFindings]:
        """Check for disparate impact (80% rule violation)."""
        findings = []

        dir_result = disparate_impact_ratio(
            data,
            attr,
            self.config.target_column,
            self.config.positive_label
        )

        dir_value = dir_result["dir"]

        # Handle edge cases
        if dir_value == float('inf') or np.isnan(dir_value):
            return findings

        if dir_value < self.thresholds.disparate_impact_critical:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.CRITICAL,
                title=f"Disparate impact detected for '{attr}'",
                description=(
                    f"The positive outcome rate for '{dir_result['unprivileged_group']}' "
                    f"is only {dir_value:.1%} of the rate for '{dir_result['privileged_group']}'. "
                    f"This violates the 80% rule (threshold: {self.thresholds.disparate_impact_critical:.0%}), "
                    f"indicating potential adverse impact."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(dir_result["privileged_group"]),
                    str(dir_result["unprivileged_group"]),
                ],
                metrics={
                    "disparate_impact_ratio": dir_value,
                    "privileged_rate": dir_result["privileged_rate"],
                    "unprivileged_rate": dir_result["unprivileged_rate"],
                    "threshold": self.thresholds.disparate_impact_critical,
                },
                remediation_suggestions=[
                    "Review labeling criteria for potential bias",
                    "Audit historical decisions that generated labels",
                    "Consider re-labeling a sample with blind review",
                    "Apply fairness constraints during model training",
                    "Use label smoothing or threshold adjustment",
                ],
                evidence={
                    "all_group_rates": dir_result["all_rates"],
                },
            ))
        elif dir_value < self.thresholds.disparate_impact_warning:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.WARNING,
                title=f"Moderate disparate impact for '{attr}'",
                description=(
                    f"The positive outcome rate for '{dir_result['unprivileged_group']}' "
                    f"is {dir_value:.1%} of the rate for '{dir_result['privileged_group']}'. "
                    f"This is close to the 80% threshold."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(dir_result["privileged_group"]),
                    str(dir_result["unprivileged_group"]),
                ],
                metrics={
                    "disparate_impact_ratio": dir_value,
                    "privileged_rate": dir_result["privileged_rate"],
                    "unprivileged_rate": dir_result["unprivileged_rate"],
                },
                remediation_suggestions=[
                    "Monitor this metric during model development",
                    "Review labeling process for potential biases",
                ],
                evidence={
                    "all_group_rates": dir_result["all_rates"],
                },
            ))

        return findings

    def _check_statistical_parity(
        self,
        data: pd.DataFrame,
        attr: str
    ) -> list[BiasFindings]:
        """Check statistical parity difference."""
        findings = []

        spd_result = statistical_parity_difference(
            data,
            attr,
            self.config.target_column,
            self.config.positive_label
        )

        spd = abs(spd_result["spd"])

        if spd > self.thresholds.statistical_parity_critical:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.CRITICAL,
                title=f"Large statistical parity difference for '{attr}'",
                description=(
                    f"The positive outcome rates differ by {spd:.1%} between groups. "
                    f"'{spd_result['privileged_group']}' has a {spd_result['privileged_rate']:.1%} rate "
                    f"while '{spd_result['unprivileged_group']}' has {spd_result['unprivileged_rate']:.1%}."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(spd_result["privileged_group"]),
                    str(spd_result["unprivileged_group"]),
                ],
                metrics={
                    "statistical_parity_difference": spd_result["spd"],
                    "abs_spd": spd,
                    "privileged_rate": spd_result["privileged_rate"],
                    "unprivileged_rate": spd_result["unprivileged_rate"],
                },
                remediation_suggestions=[
                    "Investigate root cause of label disparity",
                    "Check if disparity reflects real-world bias or data collection issues",
                    "Consider fairness-aware preprocessing or in-processing techniques",
                ],
                evidence={
                    "all_rates": spd_result["all_rates"],
                },
            ))
        elif spd > self.thresholds.statistical_parity_warning:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.WARNING,
                title=f"Moderate statistical parity difference for '{attr}'",
                description=(
                    f"Positive outcome rates differ by {spd:.1%} between groups."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(spd_result["privileged_group"]),
                    str(spd_result["unprivileged_group"]),
                ],
                metrics={
                    "statistical_parity_difference": spd_result["spd"],
                    "abs_spd": spd,
                },
                remediation_suggestions=[
                    "Monitor this metric during model development",
                ],
                evidence={
                    "all_rates": spd_result["all_rates"],
                },
            ))

        return findings

    def _check_label_rate_significance(
        self,
        data: pd.DataFrame,
        attr: str
    ) -> list[BiasFindings]:
        """Check if label rate differences are statistically significant."""
        findings = []

        # Get label rates by group
        rates = group_label_rates(
            data, attr, self.config.target_column, self.config.positive_label
        )

        # Perform chi-squared test
        contingency = pd.crosstab(
            data[attr],
            data[self.config.target_column] == self.config.positive_label
        )

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return findings

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except Exception:
            return findings

        # Calculate Cramér's V for effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        if p_value < 0.001 and cramers_v > 0.3:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.CRITICAL,
                title=f"Highly significant label association with '{attr}'",
                description=(
                    f"There is a strong, statistically significant association "
                    f"between '{attr}' and the outcome (χ²={chi2:.1f}, p<0.001, "
                    f"Cramér's V={cramers_v:.3f}). This suggests labels may be "
                    f"systematically influenced by group membership."
                ),
                affected_attribute=attr,
                affected_groups=list(contingency.index),
                metrics={
                    "chi_squared": chi2,
                    "p_value": p_value,
                    "cramers_v": cramers_v,
                    "degrees_of_freedom": dof,
                },
                remediation_suggestions=[
                    "Audit labeling process for systematic bias",
                    "Consider blind labeling protocols",
                    "Use multiple independent labelers",
                    "Apply calibration techniques post-training",
                ],
                evidence={
                    "contingency_table": contingency.to_dict(),
                    "group_rates": {k: v for k, v in rates.items() if k != "_summary"},
                },
            ))
        elif p_value < 0.05 and cramers_v > 0.1:
            findings.append(BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.INFO,
                title=f"Significant label association with '{attr}'",
                description=(
                    f"There is a statistically significant association between "
                    f"'{attr}' and the outcome (p={p_value:.4f}, Cramér's V={cramers_v:.3f})."
                ),
                affected_attribute=attr,
                affected_groups=list(contingency.index),
                metrics={
                    "chi_squared": chi2,
                    "p_value": p_value,
                    "cramers_v": cramers_v,
                },
                remediation_suggestions=[
                    "Monitor label distribution across groups",
                ],
                evidence={
                    "group_rates": {k: v for k, v in rates.items() if k != "_summary"},
                },
            ))

        return findings

    def _check_intersectional_label_bias(
        self,
        data: pd.DataFrame
    ) -> list[BiasFindings]:
        """Check for intersectional label biases."""
        findings = []

        if len(self.config.protected_attributes) < 2:
            return findings

        # Check pairs of protected attributes
        attrs = [a for a in self.config.protected_attributes if a in data.columns]

        for i, attr1 in enumerate(attrs):
            for attr2 in attrs[i+1:]:
                # Create intersection
                intersection = data[attr1].astype(str) + "_" + data[attr2].astype(str)

                # Calculate label rates for each intersection
                intersection_rates = data.groupby(intersection)[
                    self.config.target_column
                ].apply(lambda x: (x == self.config.positive_label).mean())

                if len(intersection_rates) < 2:
                    continue

                max_rate = intersection_rates.max()
                min_rate = intersection_rates.min()
                disparity = max_rate - min_rate

                if disparity > self.thresholds.label_rate_disparity_critical:
                    max_group = intersection_rates.idxmax()
                    min_group = intersection_rates.idxmin()

                    findings.append(BiasFindings(
                        category=BiasCategory.INTERSECTIONAL,
                        severity=BiasSeverity.WARNING,
                        title=f"Intersectional label disparity: {attr1} × {attr2}",
                        description=(
                            f"Large disparity in positive outcome rates across "
                            f"intersectional groups. '{max_group}' has {max_rate:.1%} "
                            f"while '{min_group}' has only {min_rate:.1%} "
                            f"(difference: {disparity:.1%})."
                        ),
                        affected_attribute=f"{attr1} × {attr2}",
                        affected_groups=[max_group, min_group],
                        metrics={
                            "max_rate": max_rate,
                            "min_rate": min_rate,
                            "disparity": disparity,
                            "max_group": max_group,
                            "min_group": min_group,
                        },
                        remediation_suggestions=[
                            "Investigate intersectional bias sources",
                            "Consider intersectional fairness constraints",
                            "Audit labels for intersectional groups separately",
                        ],
                        evidence={
                            "intersection_rates": intersection_rates.to_dict(),
                        },
                    ))

        return findings
