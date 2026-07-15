"""LLM-powered explanation and code generation for bias findings."""

from typing import Optional
import json

from bias_auditor.core.config import LLMConfig
from bias_auditor.core.report import AuditReport, BiasFindings
from bias_auditor.llm.base import get_llm_provider
from bias_auditor.llm.prompts import (
    BIAS_ANALYST_SYSTEM,
    CODE_GENERATOR_SYSTEM,
    EXPLAIN_FINDING_TEMPLATE,
    EXECUTIVE_SUMMARY_TEMPLATE,
    REMEDIATION_CODE_TEMPLATE,
    PROXY_EXPLANATION_TEMPLATE,
    QA_TEMPLATE,
)


class BiasExplainer:
    """
    LLM-powered explainer for bias findings.
    
    Provides:
    - Natural language explanations of findings
    - Executive summaries for stakeholders
    - Remediation code generation
    - Interactive Q&A about the audit
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = get_llm_provider(config)
        
        if self.provider is None:
            raise ValueError("LLM provider is required for explanations")
    
    def explain_finding(self, finding: BiasFindings) -> str:
        """
        Generate a detailed explanation of a bias finding.
        
        Parameters
        ----------
        finding : BiasFindings
            The finding to explain.
        
        Returns
        -------
        str
            Natural language explanation.
        """
        if not self.provider.is_available():
            return self._fallback_explanation(finding)
        
        # Format metrics for prompt
        metrics_str = "\n".join(f"- {k}: {v}" for k, v in finding.metrics.items())
        groups_str = ", ".join(finding.affected_groups) if finding.affected_groups else "N/A"
        
        prompt = EXPLAIN_FINDING_TEMPLATE.format(
            title=finding.title,
            category=finding.category.value,
            severity=finding.severity.value,
            description=finding.description,
            metrics=metrics_str,
            affected_groups=groups_str,
        )
        
        try:
            response = self.provider.complete(prompt, BIAS_ANALYST_SYSTEM)
            return response.content
        except Exception as e:
            return self._fallback_explanation(finding)
    
    def generate_summary(self, report: AuditReport) -> str:
        """
        Generate an executive summary of the audit report.
        
        Parameters
        ----------
        report : AuditReport
            The complete audit report.
        
        Returns
        -------
        str
            Executive summary suitable for stakeholders.
        """
        if not self.provider.is_available():
            return self._fallback_summary(report)
        
        # Format category scores
        scores_str = "\n".join(
            f"- {cat}: {score:.2f}" 
            for cat, score in report.category_scores.items()
        )
        
        # Format critical findings
        critical_str = "\n".join(
            f"- [{f.category.value}] {f.title}: {f.description[:200]}"
            for f in report.critical_findings[:5]
        ) or "None"
        
        prompt = EXECUTIVE_SUMMARY_TEMPLATE.format(
            dataset_name=report.dataset_name or "Unknown Dataset",
            total_findings=len(report.findings),
            critical_count=len(report.critical_findings),
            warning_count=len(report.warning_findings),
            category_scores=scores_str,
            critical_findings=critical_str,
        )
        
        try:
            response = self.provider.complete(prompt, BIAS_ANALYST_SYSTEM)
            return response.content
        except Exception:
            return self._fallback_summary(report)
    
    def generate_code(
        self,
        report: AuditReport,
        finding_index: Optional[int] = None,
    ) -> str:
        """
        Generate remediation code for bias findings.
        
        Parameters
        ----------
        report : AuditReport
            The audit report.
        finding_index : int, optional
            Index of specific finding. If None, generates for all critical.
        
        Returns
        -------
        str
            Python code for remediation.
        """
        if not self.provider.is_available():
            return "# LLM not available for code generation"
        
        if finding_index is not None:
            findings = [report.findings[finding_index]]
        else:
            findings = report.critical_findings or report.warning_findings[:3]
        
        if not findings:
            return "# No findings requiring remediation code"
        
        all_code = []
        all_code.append("# Auto-generated bias remediation code")
        all_code.append("# Review and adapt before using in production")
        all_code.append("")
        all_code.append("import pandas as pd")
        all_code.append("import numpy as np")
        all_code.append("from sklearn.utils import resample")
        all_code.append("")
        
        for finding in findings:
            prompt = REMEDIATION_CODE_TEMPLATE.format(
                title=finding.title,
                category=finding.category.value,
                description=finding.description,
                affected_attribute=finding.affected_attribute,
                metrics=json.dumps(finding.metrics, indent=2),
            )
            
            try:
                response = self.provider.complete(prompt, CODE_GENERATOR_SYSTEM)
                
                # Extract code from response
                code = response.content
                
                # Clean up markdown code blocks if present
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                all_code.append(f"# === Remediation for: {finding.title} ===")
                all_code.append(code.strip())
                all_code.append("")
            except Exception as e:
                all_code.append(f"# Failed to generate code for: {finding.title}")
                all_code.append(f"# Error: {str(e)}")
                all_code.append("")
        
        return "\n".join(all_code)
    
    def answer_question(self, report: AuditReport, question: str) -> str:
        """
        Answer a question about the audit report.
        
        Parameters
        ----------
        report : AuditReport
            The audit report.
        question : str
            User's question.
        
        Returns
        -------
        str
            Answer based on the report.
        """
        if not self.provider.is_available():
            return "LLM not available. Please check your configuration."
        
        # Summarize findings for context
        findings_summary = "\n".join(
            f"- [{f.severity.value}] {f.title}: {f.description[:150]}..."
            for f in report.findings[:10]
        )
        
        prompt = QA_TEMPLATE.format(
            dataset_name=report.dataset_name or "Unknown",
            bias_score=f"{report.overall_bias_score:.2f}",
            critical_count=len(report.critical_findings),
            warning_count=len(report.warning_findings),
            findings_summary=findings_summary,
            question=question,
        )
        
        try:
            response = self.provider.complete(prompt, BIAS_ANALYST_SYSTEM)
            return response.content
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def explain_proxy(
        self,
        feature: str,
        protected_attr: str,
        correlation: float,
        mutual_info: float,
    ) -> str:
        """
        Explain why a feature may be a proxy.
        
        Parameters
        ----------
        feature : str
            The feature name.
        protected_attr : str
            The protected attribute.
        correlation : float
            Correlation value.
        mutual_info : float
            Mutual information value.
        
        Returns
        -------
        str
            Explanation of the proxy relationship.
        """
        if not self.provider.is_available():
            return f"Feature '{feature}' is correlated with '{protected_attr}'."
        
        prompt = PROXY_EXPLANATION_TEMPLATE.format(
            feature_name=feature,
            protected_attribute=protected_attr,
            correlation=f"{correlation:.3f}",
            mutual_information=f"{mutual_info:.3f}",
        )
        
        try:
            response = self.provider.complete(prompt, BIAS_ANALYST_SYSTEM)
            return response.content
        except Exception:
            return f"Feature '{feature}' shows correlation ({correlation:.3f}) with '{protected_attr}'."
    
    def _fallback_explanation(self, finding: BiasFindings) -> str:
        """Generate a basic explanation without LLM."""
        return (
            f"This finding indicates {finding.severity.value}-level {finding.category.value} bias "
            f"affecting '{finding.affected_attribute}'. {finding.description} "
            f"Consider the suggested remediation actions to address this issue."
        )
    
    def _fallback_summary(self, report: AuditReport) -> str:
        """Generate a basic summary without LLM."""
        return (
            f"Bias audit completed with {len(report.findings)} findings. "
            f"Critical issues: {len(report.critical_findings)}. "
            f"Warning issues: {len(report.warning_findings)}. "
            f"Overall bias score: {report.overall_bias_score:.2f}/1.0."
        )
