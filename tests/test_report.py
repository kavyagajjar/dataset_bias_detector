"""Tests for report generation."""

import json
import os
import tempfile
from datetime import datetime

import pytest

from bias_auditor.core.report import (
    AuditReport,
    BiasCategory,
    BiasFindings,
    BiasSeverity,
    DatasetProfile,
)
from bias_auditor.report.html_generator import generate_html_report
from bias_auditor.report.json_export import export_dict, export_json


@pytest.fixture
def sample_finding():
    """Create a sample finding."""
    return BiasFindings(
        category=BiasCategory.REPRESENTATION,
        severity=BiasSeverity.CRITICAL,
        title="Severe underrepresentation",
        description="Group 'female' represents only 5% of the dataset.",
        affected_attribute="gender",
        affected_groups=["female"],
        metrics={
            "group_proportion": 0.05,
            "threshold": 0.10,
        },
        remediation_suggestions=[
            "Collect more data for underrepresented groups",
            "Use oversampling techniques",
        ],
    )


@pytest.fixture
def sample_report(sample_finding):
    """Create a sample audit report."""
    report = AuditReport(
        audit_id="test123",
        audit_timestamp=datetime.now(),
        dataset_name="test_dataset.csv",
    )

    report.add_finding(sample_finding)

    report.add_finding(
        BiasFindings(
            category=BiasCategory.LABEL,
            severity=BiasSeverity.WARNING,
            title="Moderate disparate impact",
            description="The positive rate ratio is 0.75.",
            affected_attribute="gender",
            affected_groups=["male", "female"],
            metrics={"disparate_impact_ratio": 0.75},
            remediation_suggestions=["Review labeling criteria"],
        )
    )

    report.add_finding(
        BiasFindings(
            category=BiasCategory.MISSING_DATA,
            severity=BiasSeverity.INFO,
            title="Some missing data",
            description="5% missing values in income column.",
            affected_attribute="income",
            metrics={"missing_rate": 0.05},
            remediation_suggestions=["Consider imputation"],
        )
    )

    return report


class TestBiasFindings:
    """Tests for BiasFindings dataclass."""

    def test_to_dict(self, sample_finding):
        """Finding should convert to dictionary."""
        d = sample_finding.to_dict()

        assert d["category"] == "representation"
        assert d["severity"] == "critical"
        assert d["title"] == "Severe underrepresentation"
        assert "metrics" in d
        assert "remediation_suggestions" in d

    def test_from_dict(self, sample_finding):
        """Finding should be reconstructible from dict."""
        d = sample_finding.to_dict()
        reconstructed = BiasFindings.from_dict(d)

        assert reconstructed.category == sample_finding.category
        assert reconstructed.severity == sample_finding.severity
        assert reconstructed.title == sample_finding.title

    def test_severity_comparison(self):
        """Severities should be comparable."""
        assert BiasSeverity.INFO < BiasSeverity.WARNING
        assert BiasSeverity.WARNING < BiasSeverity.CRITICAL
        assert BiasSeverity.NONE <= BiasSeverity.INFO


class TestAuditReport:
    """Tests for AuditReport."""

    def test_add_finding_updates_scores(self, sample_report):
        """Adding findings should update scores."""
        assert sample_report.overall_bias_score > 0
        assert len(sample_report.category_scores) > 0

    def test_critical_findings_property(self, sample_report):
        """Should correctly filter critical findings."""
        critical = sample_report.critical_findings

        assert len(critical) == 1
        assert all(f.severity == BiasSeverity.CRITICAL for f in critical)

    def test_warning_findings_property(self, sample_report):
        """Should correctly filter warning findings."""
        warnings = sample_report.warning_findings

        assert len(warnings) == 1
        assert all(f.severity == BiasSeverity.WARNING for f in warnings)

    def test_has_critical_bias(self, sample_report):
        """Should detect critical bias presence."""
        assert sample_report.has_critical_bias is True

        # Create report without critical findings
        clean_report = AuditReport(
            audit_id="clean",
            audit_timestamp=datetime.now(),
        )
        clean_report.add_finding(
            BiasFindings(
                category=BiasCategory.MISSING_DATA,
                severity=BiasSeverity.INFO,
                title="Minor issue",
                description="Minor description",
                affected_attribute="test",
            )
        )

        assert clean_report.has_critical_bias is False

    def test_findings_by_category(self, sample_report):
        """Should group findings by category."""
        by_cat = sample_report.findings_by_category

        assert BiasCategory.REPRESENTATION in by_cat
        assert BiasCategory.LABEL in by_cat
        assert len(by_cat[BiasCategory.REPRESENTATION]) == 1

    def test_summary_generation(self, sample_report):
        """Should generate text summary."""
        summary = sample_report.summary()

        assert "DATASET BIAS AUDIT REPORT" in summary
        assert "test123" in summary
        assert "Critical: 1" in summary

    def test_remediation_plan(self, sample_report):
        """Should generate remediation plan."""
        plan = sample_report.remediation_plan()

        assert "REMEDIATION PLAN" in plan
        assert "CRITICAL" in plan

    def test_to_json(self, sample_report):
        """Should convert to valid JSON."""
        json_str = sample_report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)

        assert parsed["audit_id"] == "test123"
        assert len(parsed["findings"]) == 3

    def test_to_dict(self, sample_report):
        """Should convert to dictionary."""
        d = sample_report.to_dict()

        assert isinstance(d, dict)
        assert "findings" in d
        assert "overall_bias_score" in d


class TestHTMLGenerator:
    """Tests for HTML report generation."""

    def test_generate_html(self, sample_report):
        """Should generate valid HTML."""
        html = generate_html_report(sample_report)

        assert "<!DOCTYPE html>" in html
        assert "Dataset Bias Audit Report" in html
        assert "test123" in html

    def test_html_contains_findings(self, sample_report):
        """HTML should contain all findings."""
        html = generate_html_report(sample_report)

        assert "Severe underrepresentation" in html
        assert "Moderate disparate impact" in html

    def test_html_contains_metrics(self, sample_report):
        """HTML should display metrics."""
        html = generate_html_report(sample_report)

        assert "0.05" in html or "5%" in html  # group_proportion

    def test_to_html_saves_file(self, sample_report):
        """to_html should save to file."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            sample_report.to_html(path)

            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)


class TestJSONExport:
    """Tests for JSON export utilities."""

    def test_export_dict(self, sample_report):
        """Should export as dictionary."""
        d = export_dict(sample_report)

        assert isinstance(d, dict)
        assert "audit_id" in d

    def test_export_json_string(self, sample_report):
        """Should export as JSON string."""
        json_str = export_json(sample_report)

        parsed = json.loads(json_str)
        assert parsed["audit_id"] == "test123"

    def test_export_json_to_file(self, sample_report):
        """Should save JSON to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            export_json(sample_report, output_path=path)

            assert os.path.exists(path)
            with open(path) as f:
                parsed = json.load(f)
            assert parsed["audit_id"] == "test123"
        finally:
            os.unlink(path)


class TestDatasetProfile:
    """Tests for DatasetProfile."""

    def test_profile_to_dict(self):
        """Profile should convert to dictionary."""
        profile = DatasetProfile(
            n_rows=1000,
            n_columns=10,
            column_types={"gender": "object", "age": "int64"},
            protected_attribute_distributions={"gender": {"male": 0.6, "female": 0.4}},
            missing_rates={"income": 0.05},
        )

        d = profile.to_dict()

        assert d["n_rows"] == 1000
        assert "gender" in d["column_types"]
        assert "gender" in d["protected_attribute_distributions"]


class TestEnhancedReportSections:
    """Tests for profile, group-stats, visualization, and config sections."""

    def _report_with_details(self, sample_report):
        sample_report.profile = DatasetProfile(
            n_rows=500,
            n_columns=6,
            column_types={"gender": "object", "approved": "int64"},
            protected_attribute_distributions={"gender": {"male": 0.7, "female": 0.3}},
            target_distribution={"1": 0.65, "0": 0.35},
            missing_rates={"income": 0.08, "gender": 0.0},
        )
        sample_report.group_stats = {
            "gender": {
                "groups": [
                    {"group": "male", "count": 350, "share": 0.7, "positive_rate": 0.72},
                    {"group": "female", "count": 150, "share": 0.3, "positive_rate": 0.49},
                ],
                "chi2_p_value": 0.0000012,
            }
        }
        sample_report.visualizations = {
            "distribution_gender": '<div class="plotly-graph-div">chart</div>',
        }
        sample_report.config_summary = {
            "protected_attributes": ["gender"],
            "target_column": "approved",
        }
        return sample_report

    def test_profile_section_rendered(self, sample_report):
        html = generate_html_report(self._report_with_details(sample_report))
        assert "Dataset Overview" in html
        assert "500" in html
        assert "Target Distribution" in html
        assert "Missing Data" in html

    def test_group_stats_table_rendered(self, sample_report):
        html = generate_html_report(self._report_with_details(sample_report))
        assert "Group Breakdown" in html
        assert "male" in html
        assert "72.0%" in html
        assert "Chi-square" in html
        assert "statistically significant" in html

    def test_visualizations_embedded(self, sample_report):
        html = generate_html_report(self._report_with_details(sample_report))
        assert "Visualizations" in html
        assert "plotly-graph-div" in html

    def test_config_appendix_rendered(self, sample_report):
        html = generate_html_report(self._report_with_details(sample_report))
        assert "Audit Configuration" in html
        assert "config-appendix" in html

    def test_sections_absent_when_no_data(self, sample_report):
        html = generate_html_report(sample_report)
        assert "Group Breakdown" not in html
        assert "Visualizations</h2>" not in html

    def test_auto_detection_notes_rendered(self, sample_report):
        report = self._report_with_details(sample_report)
        report.config_summary["auto_detection"] = {
            "notes": ["Detected protected attribute 'gender' (gender)."]
        }
        html = generate_html_report(report)
        assert "Auto-detection was used" in html
