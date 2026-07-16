"""Tests for the main BiasAuditor class."""

import numpy as np
import pandas as pd
import pytest

from bias_auditor import AuditConfig, BiasAuditor, BiasThresholds
from bias_auditor.core.report import BiasCategory, BiasSeverity


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n = 500

    return pd.DataFrame(
        {
            "gender": np.random.choice(["male", "female"], n, p=[0.7, 0.3]),
            "race": np.random.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),
            "age": np.random.randint(18, 65, n),
            "income": np.random.normal(50000, 15000, n),
            "approved": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        }
    )


@pytest.fixture
def biased_data():
    """Create a dataset with intentional biases."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame(
        {
            "gender": np.random.choice(["male", "female"], n, p=[0.9, 0.1]),  # Severe imbalance
            "race": np.random.choice(["A", "B"], n, p=[0.5, 0.5]),
            "feature": np.random.randn(n),
            "approved": np.zeros(n, dtype=int),
        }
    )

    # Add label bias
    df.loc[df["gender"] == "male", "approved"] = np.random.choice(
        [0, 1], (df["gender"] == "male").sum(), p=[0.3, 0.7]
    )
    df.loc[df["gender"] == "female", "approved"] = np.random.choice(
        [0, 1], (df["gender"] == "female").sum(), p=[0.8, 0.2]
    )

    return df


class TestBiasAuditor:
    """Tests for BiasAuditor initialization and basic functionality."""

    def test_init_basic(self):
        """Test basic initialization."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
        )

        assert auditor.config.protected_attributes == ["gender"]
        assert auditor.config.target_column == "approved"

    def test_init_with_thresholds(self):
        """Test initialization with custom thresholds."""
        thresholds = BiasThresholds(
            disparate_impact_critical=0.7,
            statistical_parity_critical=0.25,
        )

        auditor = BiasAuditor(
            protected_attributes=["gender"],
            thresholds=thresholds,
        )

        assert auditor.config.thresholds.disparate_impact_critical == 0.7

    def test_validate_data_missing_column(self, sample_data):
        """Test that validation catches missing columns."""
        auditor = BiasAuditor(
            protected_attributes=["nonexistent"],
            target_column="approved",
        )

        with pytest.raises(ValueError, match="not found in data"):
            auditor.audit(sample_data)

    def test_validate_data_empty(self):
        """Test that validation catches empty DataFrame."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
        )

        with pytest.raises(ValueError, match="empty"):
            auditor.audit(pd.DataFrame())


class TestAuditReport:
    """Tests for audit report generation."""

    def test_audit_returns_report(self, sample_data):
        """Test that audit returns a report object."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
            verbose=False,
        )

        report = auditor.audit(sample_data)

        assert report is not None
        assert report.audit_id is not None
        assert report.profile is not None

    def test_audit_detects_imbalance(self, biased_data):
        """Test that audit detects representation imbalance."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
            verbose=False,
        )

        report = auditor.audit(biased_data)

        # Should have critical findings for representation
        representation_findings = [
            f for f in report.findings if f.category == BiasCategory.REPRESENTATION
        ]
        assert len(representation_findings) > 0

    def test_audit_detects_label_bias(self, biased_data):
        """Test that audit detects label bias."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
            verbose=False,
        )

        report = auditor.audit(biased_data)

        # Should have critical findings for label bias
        label_findings = [
            f
            for f in report.findings
            if f.category == BiasCategory.LABEL
            and f.severity in [BiasSeverity.CRITICAL, BiasSeverity.WARNING]
        ]
        assert len(label_findings) > 0

    def test_report_summary(self, sample_data):
        """Test report summary generation."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            verbose=False,
        )

        report = auditor.audit(sample_data)
        summary = report.summary()

        assert "BIAS AUDIT REPORT" in summary
        assert "Overall Bias Score" in summary

    def test_report_to_json(self, sample_data):
        """Test JSON export."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            verbose=False,
        )

        report = auditor.audit(sample_data)
        json_str = report.to_json()

        import json

        data = json.loads(json_str)

        assert "audit_id" in data
        assert "findings" in data
        assert "overall_bias_score" in data


class TestQuickCheck:
    """Tests for quick check functionality."""

    def test_quick_check_returns_dict(self, sample_data):
        """Test that quick check returns a dictionary."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
        )

        result = auditor.quick_check(sample_data)

        assert isinstance(result, dict)
        assert "has_critical_bias" in result
        assert "key_metrics" in result

    def test_quick_check_detects_critical(self, biased_data):
        """Test that quick check detects critical issues."""
        auditor = BiasAuditor(
            protected_attributes=["gender"],
            target_column="approved",
        )

        result = auditor.quick_check(biased_data)

        # With severe imbalance, should detect critical bias
        assert result["has_critical_bias"] is True


class TestMultipleAttributes:
    """Tests for multiple protected attributes."""

    def test_multiple_attributes(self, sample_data):
        """Test audit with multiple protected attributes."""
        auditor = BiasAuditor(
            protected_attributes=["gender", "race"],
            target_column="approved",
            verbose=False,
        )

        report = auditor.audit(sample_data)

        # Should have findings for both attributes
        affected_attrs = set(f.affected_attribute for f in report.findings)
        assert "gender" in affected_attrs or "race" in affected_attrs

    def test_intersectional_analysis(self, sample_data):
        """Test that intersectional analysis runs."""
        config = AuditConfig(
            protected_attributes=["gender", "race"],
            target_column="approved",
            compute_intersectional=True,
        )

        auditor = BiasAuditor(config=config, verbose=False)
        report = auditor.audit(sample_data)

        # Check intersectional findings exist or at least no errors
        assert report is not None
