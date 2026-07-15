"""
Missing Data Bias Detector.

Detects non-random missingness patterns that may indicate bias.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from bias_auditor.core.config import AuditConfig
from bias_auditor.core.report import BiasFindings, BiasSeverity, BiasCategory


class MissingDataDetector:
    """
    Detector for missing data biases.
    
    Checks for:
    - Differential missing rates across protected groups
    - Patterns suggesting MNAR (Missing Not At Random)
    - Missing data in protected attributes themselves
    - Correlation between missingness and outcomes
    """
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.thresholds = config.thresholds
    
    def detect(self, data: pd.DataFrame) -> list[BiasFindings]:
        """
        Detect missing data biases.
        
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
        
        # Check overall missing data patterns
        findings.extend(self._check_overall_missing(data))
        
        # Check missing in protected attributes
        findings.extend(self._check_protected_attr_missing(data))
        
        # Check differential missingness by group
        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue
            findings.extend(self._check_differential_missing(data, attr))
        
        # Check missingness correlation with target
        if self.config.target_column and self.config.target_column in data.columns:
            findings.extend(self._check_missing_outcome_correlation(data))
        
        return findings
    
    def _check_overall_missing(self, data: pd.DataFrame) -> list[BiasFindings]:
        """Check overall missing data patterns."""
        findings = []
        
        missing_rates = data.isnull().mean()
        high_missing = missing_rates[missing_rates > 0.3]
        
        if len(high_missing) > 0:
            findings.append(BiasFindings(
                category=BiasCategory.MISSING_DATA,
                severity=BiasSeverity.INFO,
                title="High missing data rates detected",
                description=(
                    f"{len(high_missing)} columns have >30% missing values. "
                    f"Highest: '{high_missing.idxmax()}' ({high_missing.max():.1%}). "
                    f"High missing rates may introduce bias if not handled properly."
                ),
                affected_attribute="multiple",
                affected_groups=high_missing.index.tolist(),
                metrics={
                    "n_high_missing_columns": len(high_missing),
                    "max_missing_rate": high_missing.max(),
                    "max_missing_column": high_missing.idxmax(),
                },
                remediation_suggestions=[
                    "Investigate why data is missing (MCAR, MAR, MNAR)",
                    "Consider multiple imputation for MAR data",
                    "Document missingness assumptions",
                    "Test model sensitivity to imputation method",
                ],
                evidence={
                    "missing_rates": high_missing.to_dict(),
                },
            ))
        
        return findings
    
    def _check_protected_attr_missing(self, data: pd.DataFrame) -> list[BiasFindings]:
        """Check for missing values in protected attributes."""
        findings = []
        
        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue
            
            missing_rate = data[attr].isnull().mean()
            
            if missing_rate > 0.1:
                findings.append(BiasFindings(
                    category=BiasCategory.MISSING_DATA,
                    severity=BiasSeverity.WARNING,
                    title=f"Missing values in protected attribute '{attr}'",
                    description=(
                        f"{missing_rate:.1%} of values are missing in protected "
                        f"attribute '{attr}'. This limits fairness analysis and "
                        f"may indicate systematic data collection issues."
                    ),
                    affected_attribute=attr,
                    affected_groups=["missing"],
                    metrics={
                        "missing_rate": missing_rate,
                        "missing_count": int(data[attr].isnull().sum()),
                        "total_count": len(data),
                    },
                    remediation_suggestions=[
                        "Investigate why protected attribute data is missing",
                        "Consider if missingness correlates with the attribute itself",
                        "Avoid imputing protected attributes without careful consideration",
                        "Analyze missing cases separately if possible",
                    ],
                ))
            elif missing_rate > 0:
                findings.append(BiasFindings(
                    category=BiasCategory.MISSING_DATA,
                    severity=BiasSeverity.INFO,
                    title=f"Some missing values in protected attribute '{attr}'",
                    description=(
                        f"{missing_rate:.1%} of values are missing in '{attr}'."
                    ),
                    affected_attribute=attr,
                    affected_groups=["missing"],
                    metrics={
                        "missing_rate": missing_rate,
                        "missing_count": int(data[attr].isnull().sum()),
                    },
                    remediation_suggestions=[
                        "Document how missing protected attributes will be handled",
                    ],
                ))
        
        return findings
    
    def _check_differential_missing(
        self, 
        data: pd.DataFrame, 
        protected_attr: str
    ) -> list[BiasFindings]:
        """Check for differential missing rates across protected groups."""
        findings = []
        
        # Get non-null rows for the protected attribute
        valid_mask = data[protected_attr].notna()
        valid_data = data[valid_mask]
        
        if len(valid_data) < 100:
            return findings
        
        # Check missing rates for each feature by group
        feature_cols = [
            col for col in data.columns
            if col not in self.config.protected_attributes
            and col != self.config.target_column
        ]
        
        differential_missing = {}
        
        for feature in feature_cols:
            if data[feature].isnull().sum() == 0:
                continue
            
            # Calculate missing rate by group
            group_missing = valid_data.groupby(protected_attr)[feature].apply(
                lambda x: x.isnull().mean()
            )
            
            if len(group_missing) < 2:
                continue
            
            max_rate = group_missing.max()
            min_rate = group_missing.min()
            disparity = max_rate - min_rate
            
            if disparity > self.thresholds.missing_rate_disparity_warning:
                differential_missing[feature] = {
                    "disparity": disparity,
                    "max_rate": max_rate,
                    "min_rate": min_rate,
                    "max_group": group_missing.idxmax(),
                    "min_group": group_missing.idxmin(),
                    "all_rates": group_missing.to_dict(),
                }
        
        # Report critical disparities
        critical_features = {
            k: v for k, v in differential_missing.items()
            if v["disparity"] > self.thresholds.missing_rate_disparity_critical
        }
        
        if critical_features:
            worst_feature = max(critical_features.items(), key=lambda x: x[1]["disparity"])
            
            findings.append(BiasFindings(
                category=BiasCategory.MISSING_DATA,
                severity=BiasSeverity.CRITICAL,
                title=f"Severe differential missingness by '{protected_attr}'",
                description=(
                    f"{len(critical_features)} features have severely different "
                    f"missing rates across '{protected_attr}' groups. "
                    f"Worst: '{worst_feature[0]}' with {worst_feature[1]['disparity']:.1%} "
                    f"difference ({worst_feature[1]['max_group']}: {worst_feature[1]['max_rate']:.1%} "
                    f"vs {worst_feature[1]['min_group']}: {worst_feature[1]['min_rate']:.1%})."
                ),
                affected_attribute=protected_attr,
                affected_groups=list(critical_features.keys()),
                metrics={
                    "n_affected_features": len(critical_features),
                    "max_disparity": worst_feature[1]["disparity"],
                    "worst_feature": worst_feature[0],
                },
                remediation_suggestions=[
                    "Investigate root cause of differential missingness",
                    "This pattern suggests MNAR (Missing Not At Random) data",
                    "Consider group-specific imputation strategies",
                    "Document bias risk if imputing uniformly",
                    "Test model sensitivity to different imputation approaches by group",
                ],
                evidence={
                    "critical_features": critical_features,
                },
            ))
        
        # Report warning-level disparities
        warning_features = {
            k: v for k, v in differential_missing.items()
            if self.thresholds.missing_rate_disparity_warning < v["disparity"] <= self.thresholds.missing_rate_disparity_critical
        }
        
        if warning_features and not critical_features:
            findings.append(BiasFindings(
                category=BiasCategory.MISSING_DATA,
                severity=BiasSeverity.WARNING,
                title=f"Moderate differential missingness by '{protected_attr}'",
                description=(
                    f"{len(warning_features)} features have moderately different "
                    f"missing rates across '{protected_attr}' groups."
                ),
                affected_attribute=protected_attr,
                affected_groups=list(warning_features.keys()),
                metrics={
                    "n_affected_features": len(warning_features),
                },
                remediation_suggestions=[
                    "Monitor missing data patterns during model development",
                    "Consider stratified imputation",
                ],
                evidence={
                    "warning_features": warning_features,
                },
            ))
        
        return findings
    
    def _check_missing_outcome_correlation(self, data: pd.DataFrame) -> list[BiasFindings]:
        """Check if missingness correlates with outcomes."""
        findings = []
        
        target = self.config.target_column
        positive_label = self.config.positive_label
        
        # For each feature, check if missingness correlates with outcome
        problematic_features = {}
        
        for col in data.columns:
            if col == target or data[col].isnull().sum() == 0:
                continue
            
            # Create missingness indicator
            is_missing = data[col].isnull()
            
            # Check correlation with outcome
            valid_mask = data[target].notna()
            
            if valid_mask.sum() < 100:
                continue
            
            # Outcome rates for missing vs non-missing
            try:
                missing_outcome_rate = (
                    data.loc[is_missing & valid_mask, target] == positive_label
                ).mean()
                present_outcome_rate = (
                    data.loc[~is_missing & valid_mask, target] == positive_label
                ).mean()
            except Exception:
                continue
            
            disparity = abs(missing_outcome_rate - present_outcome_rate)
            
            if disparity > 0.1 and is_missing.sum() > 30:
                # Statistical test
                contingency = pd.crosstab(
                    is_missing[valid_mask],
                    data.loc[valid_mask, target] == positive_label
                )
                
                if contingency.shape == (2, 2):
                    try:
                        _, p_value = stats.fisher_exact(contingency)
                    except Exception:
                        p_value = 1.0
                    
                    if p_value < 0.05:
                        problematic_features[col] = {
                            "disparity": disparity,
                            "missing_outcome_rate": missing_outcome_rate,
                            "present_outcome_rate": present_outcome_rate,
                            "p_value": p_value,
                            "n_missing": int(is_missing.sum()),
                        }
        
        if problematic_features:
            worst = max(problematic_features.items(), key=lambda x: x[1]["disparity"])
            
            findings.append(BiasFindings(
                category=BiasCategory.MISSING_DATA,
                severity=BiasSeverity.WARNING,
                title="Missingness correlates with outcome",
                description=(
                    f"{len(problematic_features)} features have missing values that "
                    f"correlate with the outcome '{target}'. Worst: '{worst[0]}' "
                    f"(missing samples have {worst[1]['missing_outcome_rate']:.1%} "
                    f"positive rate vs {worst[1]['present_outcome_rate']:.1%} for "
                    f"non-missing samples). This suggests MNAR data that could bias models."
                ),
                affected_attribute=target,
                affected_groups=list(problematic_features.keys()),
                metrics={
                    "n_problematic_features": len(problematic_features),
                    "worst_feature": worst[0],
                    "worst_disparity": worst[1]["disparity"],
                },
                remediation_suggestions=[
                    "Missing data is likely MNAR (Missing Not At Random)",
                    "Consider modeling missingness explicitly",
                    "Use Heckman selection or pattern-mixture models",
                    "Test model predictions on missing vs non-missing subsets",
                    "Document potential bias from missingness handling",
                ],
                evidence={
                    "problematic_features": problematic_features,
                },
            ))
        
        return findings
