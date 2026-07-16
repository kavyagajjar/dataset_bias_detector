"""Tests for automatic protected-attribute and target detection."""

import numpy as np
import pandas as pd
import pytest

from bias_auditor import BiasAuditor
from bias_auditor.core.auto_detect import (
    auto_detect,
    bin_continuous_attribute,
    detect_protected_attributes,
    detect_target_column,
    infer_positive_label,
)


@pytest.fixture
def loan_data():
    """A realistic loan dataset with detectable columns."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "applicant_id": range(n),
            "gender": rng.choice(["male", "female"], n),
            "race": rng.choice(["white", "black", "asian"], n),
            "age": rng.integers(18, 80, n),
            "income": rng.integers(20000, 150000, n),
            "approved": rng.choice([0, 1], n),
        }
    )


class TestDetectProtectedAttributes:
    def test_detects_common_names(self, loan_data):
        matches = detect_protected_attributes(loan_data)
        assert matches.get("gender") == "gender"
        assert matches.get("race") == "race"
        assert matches.get("age") == "age"

    def test_ignores_unrelated_columns(self, loan_data):
        matches = detect_protected_attributes(loan_data)
        assert "income" not in matches
        assert "applicant_id" not in matches

    def test_detects_variants(self):
        df = pd.DataFrame(
            {
                "applicant_sex": ["m", "f"],
                "ethnicity": ["a", "b"],
                "zip_code": ["10001", "10002"],
                "marital_status": ["single", "married"],
            }
        )
        matches = detect_protected_attributes(df)
        assert len(matches) == 4


class TestDetectTargetColumn:
    def test_finds_binary_target(self, loan_data):
        assert detect_target_column(loan_data) == "approved"

    def test_skips_non_binary_candidate(self):
        df = pd.DataFrame({"outcome": [1, 2, 3, 4], "x": [1, 2, 3, 4]})
        assert detect_target_column(df) is None

    def test_priority_order(self):
        df = pd.DataFrame(
            {
                "approved": [0, 1, 0, 1],
                "label": [0, 1, 1, 0],
            }
        )
        # 'label' comes before 'approved' in the candidate list
        assert detect_target_column(df) == "label"

    def test_detects_promoted_and_satisfied(self):
        # Regression: these were missed in skill evals (yes/no promotions,
        # survey satisfaction) before being added to the candidate list
        df1 = pd.DataFrame({"promoted": ["yes", "no", "yes"]})
        assert detect_target_column(df1) == "promoted"
        df2 = pd.DataFrame({"satisfied": [0, 1, 1]})
        assert detect_target_column(df2) == "satisfied"

    def test_yes_no_positive_label(self):
        assert infer_positive_label(pd.Series(["yes", "no", "no"])) == "yes"


class TestInferPositiveLabel:
    def test_binary_numeric(self):
        assert infer_positive_label(pd.Series([0, 1, 1, 0])) == 1

    def test_string_hint(self):
        result = infer_positive_label(pd.Series(["approved", "denied", "denied"]))
        assert result == "approved"

    def test_boolean(self):
        assert (
            infer_positive_label(pd.Series([True, False, True])) is np.True_
            or infer_positive_label(pd.Series([True, False, True])) == True
        )  # noqa: E712


class TestBinContinuous:
    def test_age_binning(self, loan_data):
        binned = bin_continuous_attribute(loan_data, "age")
        assert binned.nunique() <= 5
        assert set(binned.unique()) <= {"<25", "25-34", "35-49", "50-64", "65+"}


class TestAutoDetect:
    def test_end_to_end(self, loan_data):
        data, result = auto_detect(loan_data)
        assert "gender" in result.protected_attributes
        assert "race" in result.protected_attributes
        assert "age_group" in result.protected_attributes
        assert result.derived_columns["age_group"] == "age"
        assert "age_group" in data.columns
        assert result.target_column == "approved"
        assert result.positive_label == 1
        assert result.notes

    def test_original_data_unmodified(self, loan_data):
        auto_detect(loan_data)
        assert "age_group" not in loan_data.columns

    def test_respects_known_target(self, loan_data):
        _, result = auto_detect(loan_data, target_column="approved")
        assert result.target_column == "approved"


class TestAuditorAutoDetect:
    def test_audit_with_auto_detect(self, loan_data):
        auditor = BiasAuditor(auto_detect=True, verbose=False)
        report = auditor.audit(loan_data)
        assert "gender" in auditor.config.protected_attributes
        assert auditor.config.target_column == "approved"
        assert "auto_detection" in report.config_summary
        assert report.group_stats  # group stats computed for detected attrs

    def test_audit_auto_detect_no_matches_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        auditor = BiasAuditor(auto_detect=True, verbose=False)
        with pytest.raises(ValueError, match="no protected attribute"):
            auditor.audit(df)
