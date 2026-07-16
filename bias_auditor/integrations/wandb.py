"""Weights & Biases integration for bias auditing."""

import os
import tempfile
from typing import Optional

from bias_auditor.core.report import AuditReport


class WandbIntegration:
    """
    Weights & Biases integration for logging bias audit results.

    Logs metrics, summaries, and artifacts to W&B for experiment tracking.

    Example
    -------
    >>> from bias_auditor import BiasAuditor
    >>> from bias_auditor.integrations import WandbIntegration
    >>>
    >>> auditor = BiasAuditor(protected_attributes=['gender'])
    >>> report = auditor.audit(df)
    >>>
    >>> wandb_int = WandbIntegration(project="bias-audits")
    >>> wandb_int.log_audit(report)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize W&B integration.

        Parameters
        ----------
        project : str, optional
            W&B project name.
        entity : str, optional
            W&B entity (team/user).
        config : dict, optional
            Additional config to log.
        """
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb") from None

        self.project = project or "bias-auditor"
        self.entity = entity
        self.config = config or {}

    def log_audit(
        self,
        report: AuditReport,
        run_name: Optional[str] = None,
        log_html: bool = True,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        finish: bool = True,
    ) -> str:
        """
        Log audit report to Weights & Biases.

        Parameters
        ----------
        report : AuditReport
            The audit report to log.
        run_name : str, optional
            Name for the W&B run.
        log_html : bool
            Whether to log HTML report as artifact.
        tags : list[str], optional
            Tags for the run.
        notes : str, optional
            Notes for the run.
        finish : bool
            Whether to finish the run after logging.

        Returns
        -------
        str
            The W&B run ID.
        """
        # Build config
        config = {
            **self.config,
            "audit_id": report.audit_id,
            "dataset_name": report.dataset_name,
            "protected_attributes": report.config_summary.get("protected_attributes", []),
        }

        # Initialize run
        run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name or f"bias_audit_{report.audit_id}",
            config=config,
            tags=tags or ["bias-audit"],
            notes=notes,
        )

        # Log summary metrics
        self._wandb.summary["bias_score"] = report.overall_bias_score
        self._wandb.summary["critical_findings"] = len(report.critical_findings)
        self._wandb.summary["warning_findings"] = len(report.warning_findings)
        self._wandb.summary["total_findings"] = len(report.findings)
        self._wandb.summary["has_critical_bias"] = report.has_critical_bias

        # Log category scores
        for category, score in report.category_scores.items():
            self._wandb.summary[f"bias_{category}"] = score

        # Log metrics over time (if running multiple audits)
        self._wandb.log(
            {
                "bias_score": report.overall_bias_score,
                "critical_findings": len(report.critical_findings),
                "warning_findings": len(report.warning_findings),
                **{f"bias_{k}": v for k, v in report.category_scores.items()},
            }
        )

        # Create findings table
        if report.findings:
            findings_table = self._wandb.Table(
                columns=["Severity", "Category", "Title", "Description", "Attribute"]
            )
            for f in report.findings:
                findings_table.add_data(
                    f.severity.value,
                    f.category.value,
                    f.title,
                    f.description[:200],
                    f.affected_attribute,
                )
            self._wandb.log({"findings": findings_table})

        # Log category scores as bar chart
        category_data = [
            [cat.replace("_", " ").title(), score] for cat, score in report.category_scores.items()
        ]
        if category_data:
            table = self._wandb.Table(data=category_data, columns=["Category", "Score"])
            self._wandb.log(
                {
                    "category_scores": self._wandb.plot.bar(
                        table, "Category", "Score", title="Bias Score by Category"
                    )
                }
            )

        # Log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON report
            json_path = os.path.join(tmpdir, "audit_report.json")
            with open(json_path, "w") as f:
                f.write(report.to_json())

            artifact = self._wandb.Artifact(
                name=f"bias-report-{report.audit_id}",
                type="bias-audit",
                description=f"Bias audit report for {report.dataset_name or 'dataset'}",
            )
            artifact.add_file(json_path, name="report.json")

            # HTML report
            if log_html:
                html_path = os.path.join(tmpdir, "audit_report.html")
                report.to_html(html_path)
                artifact.add_file(html_path, name="report.html")

            self._wandb.log_artifact(artifact)

        run_id = run.id

        if finish:
            self._wandb.finish()

        return run_id

    def log_to_existing_run(
        self,
        report: AuditReport,
        prefix: str = "data_",
    ):
        """
        Log audit metrics to the current W&B run.

        Useful for adding bias metrics during model training.

        Parameters
        ----------
        report : AuditReport
            The audit report.
        prefix : str
            Prefix for metric names.
        """
        if self._wandb.run is None:
            raise RuntimeError("No active W&B run. Call wandb.init() first.")

        self._wandb.log(
            {
                f"{prefix}bias_score": report.overall_bias_score,
                f"{prefix}critical_findings": len(report.critical_findings),
                **{f"{prefix}bias_{k}": v for k, v in report.category_scores.items()},
            }
        )

        self._wandb.summary[f"{prefix}has_critical_bias"] = report.has_critical_bias

    def create_report_panel(self, report: AuditReport) -> dict:
        """
        Create W&B report panel configuration.

        Parameters
        ----------
        report : AuditReport
            The audit report.

        Returns
        -------
        dict
            Panel configuration for W&B reports.
        """
        return {
            "panel_type": "vega",
            "panel_config": {
                "title": "Bias Audit Summary",
                "description": f"Audit {report.audit_id} - Score: {report.overall_bias_score:.2f}",
                "metrics": [
                    "bias_score",
                    *[f"bias_{cat}" for cat in report.category_scores.keys()],
                ],
            },
        }

    def log_comparison(
        self,
        reports: list[AuditReport],
        names: Optional[list[str]] = None,
    ):
        """
        Log comparison of multiple audit reports.

        Parameters
        ----------
        reports : list[AuditReport]
            List of audit reports to compare.
        names : list[str], optional
            Names for each report (defaults to audit IDs).
        """
        if self._wandb.run is None:
            self._wandb.init(project=self.project, entity=self.entity)

        names = names or [r.audit_id for r in reports]

        # Create comparison table
        columns = ["Name", "Bias Score", "Critical", "Warnings", "Total"]
        all_categories = set()
        for r in reports:
            all_categories.update(r.category_scores.keys())
        columns.extend([f"{cat.replace('_', ' ').title()}" for cat in sorted(all_categories)])

        table = self._wandb.Table(columns=columns)

        for name, report in zip(names, reports):
            row = [
                name,
                report.overall_bias_score,
                len(report.critical_findings),
                len(report.warning_findings),
                len(report.findings),
            ]
            for cat in sorted(all_categories):
                row.append(report.category_scores.get(cat, 0))
            table.add_data(*row)

        self._wandb.log({"audit_comparison": table})
