"""
Dataset Bias Auditor
====================

Detect representation biases, label biases, and fairness issues in datasets
*before* training — with actionable remediation suggestions.

Example usage:
    >>> from bias_auditor import BiasAuditor
    >>> auditor = BiasAuditor(
    ...     protected_attributes=['gender', 'race'],
    ...     target_column='approved'
    ... )
    >>> report = auditor.audit(df)
    >>> report.summary()
    >>> report.to_html('audit_report.html')

With visualizations:
    >>> from bias_auditor.visualizations import BiasVisualizer
    >>> viz = BiasVisualizer()
    >>> chart = viz.label_rates_by_group(df, 'gender', 'approved')

With MLOps integration:
    >>> from bias_auditor.integrations import MLflowIntegration
    >>> mlflow_int = MLflowIntegration()
    >>> mlflow_int.log_audit(report)
"""

from bias_auditor.core.auditor import BiasAuditor
from bias_auditor.core.config import AuditConfig, BiasThresholds, LLMConfig
from bias_auditor.core.report import AuditReport, BiasCategory, BiasFindings, BiasSeverity

__version__ = "0.1.0"
__all__ = [
    "BiasAuditor",
    "AuditConfig",
    "AuditReport",
    "BiasFindings",
    "BiasThresholds",
    "BiasSeverity",
    "BiasCategory",
    "LLMConfig",
]
