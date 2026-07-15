"""Report data structures for bias audit results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json


class BiasSeverity(str, Enum):
    """Severity levels for detected biases."""
    
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    
    def __lt__(self, other: "BiasSeverity") -> bool:
        order = [BiasSeverity.NONE, BiasSeverity.INFO, BiasSeverity.WARNING, BiasSeverity.CRITICAL]
        return order.index(self) < order.index(other)
    
    def __le__(self, other: "BiasSeverity") -> bool:
        return self == other or self < other


class BiasCategory(str, Enum):
    """Categories of bias types."""
    
    REPRESENTATION = "representation"
    LABEL = "label"
    FEATURE_PROXY = "feature_proxy"
    MISSING_DATA = "missing_data"
    TEXT = "text"
    INTERSECTIONAL = "intersectional"


@dataclass
class BiasFindings:
    """
    Individual bias finding with details and recommendations.
    """
    
    category: BiasCategory
    severity: BiasSeverity
    title: str
    description: str
    
    # Affected attributes/columns
    affected_attribute: str
    affected_groups: list[str] = field(default_factory=list)
    
    # Quantitative metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Remediation
    remediation_suggestions: list[str] = field(default_factory=list)
    remediation_code: Optional[str] = None
    
    # LLM-generated explanation (if available)
    llm_explanation: Optional[str] = None
    
    # Supporting evidence
    evidence: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert finding to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_attribute": self.affected_attribute,
            "affected_groups": self.affected_groups,
            "metrics": self.metrics,
            "remediation_suggestions": self.remediation_suggestions,
            "remediation_code": self.remediation_code,
            "llm_explanation": self.llm_explanation,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BiasFindings":
        """Create finding from dictionary."""
        return cls(
            category=BiasCategory(data["category"]),
            severity=BiasSeverity(data["severity"]),
            title=data["title"],
            description=data["description"],
            affected_attribute=data["affected_attribute"],
            affected_groups=data.get("affected_groups", []),
            metrics=data.get("metrics", {}),
            remediation_suggestions=data.get("remediation_suggestions", []),
            remediation_code=data.get("remediation_code"),
            llm_explanation=data.get("llm_explanation"),
            evidence=data.get("evidence", {}),
        )


@dataclass
class DatasetProfile:
    """Basic dataset statistics and profile."""
    
    n_rows: int
    n_columns: int
    column_types: dict[str, str]
    protected_attribute_distributions: dict[str, dict[str, float]]
    target_distribution: Optional[dict[str, float]] = None
    missing_rates: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert profile to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "column_types": self.column_types,
            "protected_attribute_distributions": self.protected_attribute_distributions,
            "target_distribution": self.target_distribution,
            "missing_rates": self.missing_rates,
        }


@dataclass
class AuditReport:
    """
    Complete bias audit report with findings and recommendations.
    """
    
    # Metadata
    audit_id: str
    audit_timestamp: datetime
    dataset_name: Optional[str] = None
    
    # Dataset profile
    profile: Optional[DatasetProfile] = None
    
    # Findings by category
    findings: list[BiasFindings] = field(default_factory=list)
    
    # Overall scores
    overall_bias_score: float = 0.0  # 0-1, higher = more bias
    category_scores: dict[str, float] = field(default_factory=dict)
    
    # Configuration used
    config_summary: dict = field(default_factory=dict)
    
    # Visualizations (file paths or base64 encoded)
    visualizations: dict[str, str] = field(default_factory=dict)
    
    # LLM-generated content
    executive_summary: Optional[str] = None
    detailed_narrative: Optional[str] = None
    
    def add_finding(self, finding: BiasFindings) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)
        self._update_scores()
    
    def _update_scores(self) -> None:
        """Update overall and category scores based on findings."""
        if not self.findings:
            self.overall_bias_score = 0.0
            return
        
        # Score mapping
        severity_scores = {
            BiasSeverity.NONE: 0.0,
            BiasSeverity.INFO: 0.1,
            BiasSeverity.WARNING: 0.5,
            BiasSeverity.CRITICAL: 1.0,
        }
        
        # Calculate category scores
        category_findings: dict[str, list[float]] = {}
        for finding in self.findings:
            cat = finding.category.value
            if cat not in category_findings:
                category_findings[cat] = []
            category_findings[cat].append(severity_scores[finding.severity])
        
        self.category_scores = {
            cat: max(scores) for cat, scores in category_findings.items()
        }
        
        # Overall score is the max severity found
        self.overall_bias_score = max(self.category_scores.values()) if self.category_scores else 0.0
    
    @property
    def critical_findings(self) -> list[BiasFindings]:
        """Get all critical severity findings."""
        return [f for f in self.findings if f.severity == BiasSeverity.CRITICAL]
    
    @property
    def warning_findings(self) -> list[BiasFindings]:
        """Get all warning severity findings."""
        return [f for f in self.findings if f.severity == BiasSeverity.WARNING]
    
    @property
    def has_critical_bias(self) -> bool:
        """Check if any critical bias was detected."""
        return len(self.critical_findings) > 0
    
    @property
    def findings_by_category(self) -> dict[BiasCategory, list[BiasFindings]]:
        """Group findings by category."""
        result: dict[BiasCategory, list[BiasFindings]] = {}
        for finding in self.findings:
            if finding.category not in result:
                result[finding.category] = []
            result[finding.category].append(finding)
        return result
    
    def summary(self, style: str = "rich") -> str:
        """
        Generate a text summary of the audit.
        
        Parameters
        ----------
        style : str
            Output style: 'rich' (formatted), 'plain' (simple text), 'markdown'
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DATASET BIAS AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"Audit ID: {self.audit_id}")
        lines.append(f"Timestamp: {self.audit_timestamp.isoformat()}")
        if self.dataset_name:
            lines.append(f"Dataset: {self.dataset_name}")
        lines.append("")
        
        # Overall score
        score_label = "LOW" if self.overall_bias_score < 0.3 else \
                     "MODERATE" if self.overall_bias_score < 0.6 else "HIGH"
        lines.append(f"Overall Bias Score: {self.overall_bias_score:.2f} ({score_label})")
        lines.append("")
        
        # Findings summary
        lines.append("FINDINGS SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Critical: {len(self.critical_findings)}")
        lines.append(f"  Warning:  {len(self.warning_findings)}")
        lines.append(f"  Info:     {len([f for f in self.findings if f.severity == BiasSeverity.INFO])}")
        lines.append("")
        
        # Critical findings detail
        if self.critical_findings:
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 40)
            for finding in self.critical_findings:
                lines.append(f"  [{finding.category.value.upper()}] {finding.title}")
                lines.append(f"    {finding.description}")
                if finding.remediation_suggestions:
                    lines.append(f"    Fix: {finding.remediation_suggestions[0]}")
                lines.append("")
        
        # Category breakdown
        lines.append("CATEGORY SCORES")
        lines.append("-" * 40)
        for cat, score in self.category_scores.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {cat:20} [{bar}] {score:.2f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "audit_id": self.audit_id,
            "audit_timestamp": self.audit_timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "profile": self.profile.to_dict() if self.profile else None,
            "findings": [f.to_dict() for f in self.findings],
            "overall_bias_score": self.overall_bias_score,
            "category_scores": self.category_scores,
            "config_summary": self.config_summary,
            "executive_summary": self.executive_summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_html(self, output_path: Optional[str] = None) -> str:
        """
        Generate HTML report.
        
        Parameters
        ----------
        output_path : str, optional
            If provided, save HTML to this path.
        
        Returns
        -------
        str
            HTML content
        """
        # Import here to avoid circular dependency
        from bias_auditor.report.html_generator import generate_html_report
        
        html = generate_html_report(self)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(html)
        
        return html
    
    def remediation_plan(self) -> str:
        """Generate a prioritized remediation plan."""
        lines = []
        lines.append("REMEDIATION PLAN")
        lines.append("=" * 60)
        lines.append("")
        
        # Sort by severity
        sorted_findings = sorted(self.findings, key=lambda f: f.severity, reverse=True)
        
        priority = 1
        for finding in sorted_findings:
            if finding.severity in [BiasSeverity.CRITICAL, BiasSeverity.WARNING]:
                lines.append(f"{priority}. [{finding.severity.value.upper()}] {finding.title}")
                lines.append(f"   Category: {finding.category.value}")
                lines.append(f"   Affected: {finding.affected_attribute}")
                if finding.remediation_suggestions:
                    lines.append("   Actions:")
                    for suggestion in finding.remediation_suggestions:
                        lines.append(f"     - {suggestion}")
                if finding.remediation_code:
                    lines.append("   Code:")
                    for line in finding.remediation_code.split("\n")[:5]:
                        lines.append(f"     {line}")
                    if len(finding.remediation_code.split("\n")) > 5:
                        lines.append("     ...")
                lines.append("")
                priority += 1
        
        if priority == 1:
            lines.append("No critical or warning issues found. Dataset passes bias audit!")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"AuditReport(id={self.audit_id}, "
                f"findings={len(self.findings)}, "
                f"score={self.overall_bias_score:.2f})")
