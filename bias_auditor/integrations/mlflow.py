"""MLflow integration for bias auditing."""

import os
import tempfile
from typing import Optional

from bias_auditor.core.report import AuditReport


class MLflowIntegration:
    """
    MLflow integration for logging bias audit results.

    Logs metrics, parameters, and artifacts to MLflow for experiment tracking.

    Example
    -------
    >>> from bias_auditor import BiasAuditor
    >>> from bias_auditor.integrations import MLflowIntegration
    >>>
    >>> auditor = BiasAuditor(protected_attributes=['gender'])
    >>> report = auditor.audit(df)
    >>>
    >>> mlflow_int = MLflowIntegration()
    >>> mlflow_int.log_audit(report, run_name="data_audit_v1")
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize MLflow integration.

        Parameters
        ----------
        tracking_uri : str, optional
            MLflow tracking server URI.
        experiment_name : str, optional
            Name of the MLflow experiment.
        """
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError(
                "mlflow not installed. Install with: pip install mlflow"
            ) from None

        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            self._mlflow.set_experiment(experiment_name)

    def log_audit(
        self,
        report: AuditReport,
        run_name: Optional[str] = None,
        log_html: bool = True,
        log_json: bool = True,
        tags: Optional[dict[str, str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """
        Log audit report to MLflow.

        Parameters
        ----------
        report : AuditReport
            The audit report to log.
        run_name : str, optional
            Name for the MLflow run.
        log_html : bool
            Whether to log HTML report as artifact.
        log_json : bool
            Whether to log JSON report as artifact.
        tags : dict, optional
            Additional tags to log.
        parent_run_id : str, optional
            Parent run ID for nested runs.

        Returns
        -------
        str
            The MLflow run ID.
        """
        run_context = {}
        if run_name:
            run_context["run_name"] = run_name
        if parent_run_id:
            run_context["nested"] = True
            # Start as child run
            with self._mlflow.start_run(**run_context) as run:
                return self._log_to_run(report, log_html, log_json, tags, run)
        else:
            with self._mlflow.start_run(**run_context) as run:
                return self._log_to_run(report, log_html, log_json, tags, run)

    def _log_to_run(
        self,
        report: AuditReport,
        log_html: bool,
        log_json: bool,
        tags: Optional[dict[str, str]],
        run,
    ) -> str:
        """Log audit data to the current run."""
        # Log parameters
        self._mlflow.log_param("audit_id", report.audit_id)
        self._mlflow.log_param("dataset_name", report.dataset_name or "unknown")
        self._mlflow.log_param("n_protected_attributes",
                               len(report.config_summary.get("protected_attributes", [])))

        # Log metrics
        self._mlflow.log_metric("bias_score", report.overall_bias_score)
        self._mlflow.log_metric("critical_findings", len(report.critical_findings))
        self._mlflow.log_metric("warning_findings", len(report.warning_findings))
        self._mlflow.log_metric("total_findings", len(report.findings))

        # Log category scores
        for category, score in report.category_scores.items():
            self._mlflow.log_metric(f"bias_{category}", score)

        # Log tags
        self._mlflow.set_tag("audit_timestamp", report.audit_timestamp.isoformat())
        self._mlflow.set_tag("has_critical_bias", str(report.has_critical_bias))

        if tags:
            for key, value in tags.items():
                self._mlflow.set_tag(key, value)

        # Log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            if log_json:
                json_path = os.path.join(tmpdir, "audit_report.json")
                with open(json_path, "w") as f:
                    f.write(report.to_json())
                self._mlflow.log_artifact(json_path)

            if log_html:
                html_path = os.path.join(tmpdir, "audit_report.html")
                report.to_html(html_path)
                self._mlflow.log_artifact(html_path)

            # Log findings summary
            findings_path = os.path.join(tmpdir, "findings_summary.txt")
            with open(findings_path, "w") as f:
                f.write(report.summary())
            self._mlflow.log_artifact(findings_path)

        return run.info.run_id

    def log_to_existing_run(
        self,
        report: AuditReport,
        run_id: str,
        prefix: str = "data_",
    ):
        """
        Log audit metrics to an existing MLflow run.

        Useful for adding bias metrics to a model training run.

        Parameters
        ----------
        report : AuditReport
            The audit report.
        run_id : str
            Existing MLflow run ID.
        prefix : str
            Prefix for metric names.
        """
        with self._mlflow.start_run(run_id=run_id):
            self._mlflow.log_metric(f"{prefix}bias_score", report.overall_bias_score)
            self._mlflow.log_metric(f"{prefix}critical_findings", len(report.critical_findings))

            for category, score in report.category_scores.items():
                self._mlflow.log_metric(f"{prefix}bias_{category}", score)

    def create_model_card_bias_section(self, report: AuditReport) -> str:
        """
        Generate bias section text for MLflow model card.

        Parameters
        ----------
        report : AuditReport
            The audit report.

        Returns
        -------
        str
            Markdown text for model card bias section.
        """
        lines = []
        lines.append("## Data Bias Assessment\n")
        lines.append(f"**Audit ID:** {report.audit_id}\n")
        lines.append(f"**Overall Bias Score:** {report.overall_bias_score:.2f}\n")
        lines.append(f"**Timestamp:** {report.audit_timestamp.isoformat()}\n")
        lines.append("")

        lines.append("### Findings Summary\n")
        lines.append(f"- Critical Issues: {len(report.critical_findings)}")
        lines.append(f"- Warnings: {len(report.warning_findings)}")
        lines.append(f"- Total Findings: {len(report.findings)}")
        lines.append("")

        if report.critical_findings:
            lines.append("### Critical Issues\n")
            for finding in report.critical_findings[:5]:
                lines.append(f"- **{finding.title}**: {finding.description[:200]}...")
            lines.append("")

        lines.append("### Category Scores\n")
        lines.append("| Category | Score |")
        lines.append("|----------|-------|")
        for cat, score in sorted(report.category_scores.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat.replace('_', ' ').title()} | {score:.2f} |")

        return "\n".join(lines)
