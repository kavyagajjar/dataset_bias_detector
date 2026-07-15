"""
Representation Bias Detector.

Detects underrepresentation, overrepresentation, and distribution skew
in protected attributes.
"""

from typing import Optional
import numpy as np
import pandas as pd

from bias_auditor.core.config import AuditConfig
from bias_auditor.core.report import BiasFindings, BiasSeverity, BiasCategory
from bias_auditor.metrics.fairness import (
    class_imbalance_ratio,
    normalized_entropy,
    kl_divergence,
)


class RepresentationDetector:
    """
    Detector for representation biases in datasets.
    
    Checks for:
    - Underrepresented groups (below minimum threshold)
    - Severe class imbalance
    - Distribution skew vs. reference population
    - Simpson's paradox indicators
    """
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.thresholds = config.thresholds
    
    def detect(self, data: pd.DataFrame) -> list[BiasFindings]:
        """
        Detect representation biases in the dataset.
        
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
        
        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue
            
            # Check group proportions
            findings.extend(self._check_group_proportions(data, attr))
            
            # Check class imbalance
            findings.extend(self._check_imbalance(data, attr))
            
            # Check against reference distribution if provided
            if attr in self.config.reference_distributions:
                findings.extend(self._check_reference_distribution(data, attr))
            
            # Check for intersectional underrepresentation
            if self.config.compute_intersectional:
                findings.extend(self._check_intersectional(data, attr))
        
        return findings
    
    def _check_group_proportions(
        self, 
        data: pd.DataFrame, 
        attr: str
    ) -> list[BiasFindings]:
        """Check if any group falls below minimum proportion thresholds."""
        findings = []
        
        distribution = data[attr].value_counts(normalize=True)
        
        for group, proportion in distribution.items():
            if proportion < self.thresholds.min_group_proportion_critical:
                findings.append(BiasFindings(
                    category=BiasCategory.REPRESENTATION,
                    severity=BiasSeverity.CRITICAL,
                    title=f"Severely underrepresented group: {group}",
                    description=(
                        f"Group '{group}' in '{attr}' represents only "
                        f"{proportion:.1%} of the dataset, below the critical "
                        f"threshold of {self.thresholds.min_group_proportion_critical:.1%}. "
                        f"This can lead to poor model performance and unfair outcomes "
                        f"for this group."
                    ),
                    affected_attribute=attr,
                    affected_groups=[str(group)],
                    metrics={
                        "group_proportion": proportion,
                        "threshold": self.thresholds.min_group_proportion_critical,
                        "group_count": int(distribution[group] * len(data)),
                    },
                    remediation_suggestions=[
                        f"Collect more data for group '{group}'",
                        "Use oversampling techniques (SMOTE, ADASYN)",
                        "Apply class weights during model training",
                        "Consider stratified sampling for train/test splits",
                    ],
                    evidence={"full_distribution": distribution.to_dict()},
                ))
            elif proportion < self.thresholds.min_group_proportion_warning:
                findings.append(BiasFindings(
                    category=BiasCategory.REPRESENTATION,
                    severity=BiasSeverity.WARNING,
                    title=f"Underrepresented group: {group}",
                    description=(
                        f"Group '{group}' in '{attr}' represents only "
                        f"{proportion:.1%} of the dataset. Consider collecting "
                        f"more data or using resampling techniques."
                    ),
                    affected_attribute=attr,
                    affected_groups=[str(group)],
                    metrics={
                        "group_proportion": proportion,
                        "threshold": self.thresholds.min_group_proportion_warning,
                        "group_count": int(distribution[group] * len(data)),
                    },
                    remediation_suggestions=[
                        f"Consider collecting more data for group '{group}'",
                        "Monitor model performance across groups",
                    ],
                    evidence={"full_distribution": distribution.to_dict()},
                ))
        
        return findings
    
    def _check_imbalance(
        self, 
        data: pd.DataFrame, 
        attr: str
    ) -> list[BiasFindings]:
        """Check overall class imbalance ratio."""
        findings = []
        
        imbalance_result = class_imbalance_ratio(data, attr)
        ratio = imbalance_result["ratio"]
        entropy = normalized_entropy(data, attr)
        
        if ratio > self.thresholds.imbalance_ratio_critical:
            findings.append(BiasFindings(
                category=BiasCategory.REPRESENTATION,
                severity=BiasSeverity.CRITICAL,
                title=f"Severe class imbalance in '{attr}'",
                description=(
                    f"The majority group '{imbalance_result['majority_class']}' is "
                    f"{ratio:.1f}x larger than the minority group "
                    f"'{imbalance_result['minority_class']}'. This severe imbalance "
                    f"can cause models to ignore minority groups entirely."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(imbalance_result["majority_class"]),
                    str(imbalance_result["minority_class"]),
                ],
                metrics={
                    "imbalance_ratio": ratio,
                    "normalized_entropy": entropy,
                    "majority_count": imbalance_result["majority_count"],
                    "minority_count": imbalance_result["minority_count"],
                },
                remediation_suggestions=[
                    "Use SMOTE or ADASYN for synthetic minority oversampling",
                    "Apply class weights inversely proportional to frequency",
                    "Use undersampling techniques like Tomek links or ENN",
                    "Consider ensemble methods designed for imbalanced data",
                    "Stratify all data splits to maintain class ratios",
                ],
                evidence={"distribution": imbalance_result["distribution"]},
            ))
        elif ratio > self.thresholds.imbalance_ratio_warning:
            findings.append(BiasFindings(
                category=BiasCategory.REPRESENTATION,
                severity=BiasSeverity.WARNING,
                title=f"Moderate class imbalance in '{attr}'",
                description=(
                    f"The majority group is {ratio:.1f}x larger than the minority group. "
                    f"Consider using resampling or weighted training."
                ),
                affected_attribute=attr,
                affected_groups=[
                    str(imbalance_result["majority_class"]),
                    str(imbalance_result["minority_class"]),
                ],
                metrics={
                    "imbalance_ratio": ratio,
                    "normalized_entropy": entropy,
                },
                remediation_suggestions=[
                    "Apply class weights during training",
                    "Use stratified sampling for data splits",
                ],
                evidence={"distribution": imbalance_result["distribution"]},
            ))
        
        return findings
    
    def _check_reference_distribution(
        self, 
        data: pd.DataFrame, 
        attr: str
    ) -> list[BiasFindings]:
        """Compare distribution against reference population."""
        findings = []
        
        observed = data[attr].value_counts(normalize=True).to_dict()
        expected = self.config.reference_distributions[attr]
        
        # Calculate KL divergence
        kl_div = kl_divergence(observed, expected)
        
        # Find most divergent groups
        divergences = {}
        for group in set(observed.keys()) | set(expected.keys()):
            obs_rate = observed.get(group, 0)
            exp_rate = expected.get(group, 0)
            divergences[group] = obs_rate - exp_rate
        
        most_under = min(divergences.items(), key=lambda x: x[1])
        most_over = max(divergences.items(), key=lambda x: x[1])
        
        if kl_div > 0.5:  # Significant divergence
            findings.append(BiasFindings(
                category=BiasCategory.REPRESENTATION,
                severity=BiasSeverity.WARNING,
                title=f"Distribution mismatch for '{attr}'",
                description=(
                    f"The dataset distribution differs significantly from the "
                    f"reference population (KL divergence: {kl_div:.3f}). "
                    f"Group '{most_under[0]}' is most underrepresented "
                    f"({most_under[1]:+.1%} vs. reference)."
                ),
                affected_attribute=attr,
                affected_groups=list(divergences.keys()),
                metrics={
                    "kl_divergence": kl_div,
                    "most_underrepresented": most_under[0],
                    "underrepresentation_gap": most_under[1],
                    "most_overrepresented": most_over[0],
                    "overrepresentation_gap": most_over[1],
                },
                remediation_suggestions=[
                    "Resample to match reference population distribution",
                    "Apply importance weighting based on reference distribution",
                    f"Collect more data for underrepresented group '{most_under[0]}'",
                ],
                evidence={
                    "observed": observed,
                    "expected": expected,
                    "divergences": divergences,
                },
            ))
        
        return findings
    
    def _check_intersectional(
        self, 
        data: pd.DataFrame, 
        primary_attr: str
    ) -> list[BiasFindings]:
        """Check for intersectional underrepresentation."""
        findings = []
        
        other_attrs = [
            a for a in self.config.protected_attributes
            if a != primary_attr and a in data.columns
        ]
        
        if not other_attrs:
            return findings
        
        for other_attr in other_attrs[:self.config.max_intersectional_depth - 1]:
            # Create intersection column
            intersection = data[primary_attr].astype(str) + "_" + data[other_attr].astype(str)
            intersection_counts = intersection.value_counts()
            
            # Find very small intersections
            small_groups = intersection_counts[
                intersection_counts < self.config.min_group_size
            ]
            
            if len(small_groups) > 0:
                total_small = small_groups.sum()
                pct_small = total_small / len(data)
                
                if pct_small > 0.01:  # More than 1% in small intersections
                    findings.append(BiasFindings(
                        category=BiasCategory.INTERSECTIONAL,
                        severity=BiasSeverity.INFO,
                        title=f"Intersectional underrepresentation: {primary_attr} × {other_attr}",
                        description=(
                            f"{len(small_groups)} intersectional groups have fewer than "
                            f"{self.config.min_group_size} samples. These small groups "
                            f"({pct_small:.1%} of data) may not have reliable model predictions."
                        ),
                        affected_attribute=f"{primary_attr} × {other_attr}",
                        affected_groups=small_groups.index.tolist()[:10],  # Top 10
                        metrics={
                            "n_small_intersections": len(small_groups),
                            "total_in_small": int(total_small),
                            "pct_in_small": pct_small,
                            "min_group_size_threshold": self.config.min_group_size,
                        },
                        remediation_suggestions=[
                            "Consider merging small intersectional groups",
                            "Collect more data for underrepresented intersections",
                            "Use hierarchical/multilevel models",
                        ],
                        evidence={
                            "small_groups": small_groups.to_dict(),
                        },
                    ))
        
        return findings
