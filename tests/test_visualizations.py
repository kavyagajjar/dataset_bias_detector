"""Tests for visualization module."""

import numpy as np
import pandas as pd
import pytest

try:
    from bias_auditor.visualizations import (
        BiasVisualizer,
        generate_all_visualizations,
        plot_category_scores,
        plot_group_distribution,
        plot_label_rates,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


@pytest.fixture
def sample_data():
    """Create sample dataset for visualization tests."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.6, 0.4]),
        'age_group': np.random.choice(['young', 'middle', 'senior'], n, p=[0.3, 0.4, 0.3]),
        'income': np.random.normal(50000, 15000, n),
        'approved': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })


@pytest.fixture
def category_scores():
    """Sample category scores for visualization."""
    return {
        'representation': 0.7,
        'label': 0.4,
        'feature_proxy': 0.2,
        'missing_data': 0.1,
    }


@pytest.mark.skipif(not HAS_VIZ, reason="Visualization dependencies not installed")
class TestBiasVisualizer:
    """Tests for BiasVisualizer class."""

    def test_group_distribution(self, sample_data):
        """Should generate group distribution chart."""
        viz = BiasVisualizer()
        result = viz.group_distribution(sample_data, 'gender')

        assert isinstance(result, str)
        assert len(result) > 0

    def test_group_distribution_with_reference(self, sample_data):
        """Should include reference distribution when provided."""
        viz = BiasVisualizer()
        reference = {'male': 0.5, 'female': 0.5}

        result = viz.group_distribution(
            sample_data, 'gender',
            reference_dist=reference
        )

        assert isinstance(result, str)

    def test_label_rates_by_group(self, sample_data):
        """Should generate label rates chart."""
        viz = BiasVisualizer()
        result = viz.label_rates_by_group(
            sample_data, 'gender', 'approved', positive_label=1
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_category_scores_radar(self, category_scores):
        """Should generate radar chart."""
        viz = BiasVisualizer()
        result = viz.category_scores_radar(category_scores)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_bias_heatmap(self, sample_data):
        """Should generate heatmap for multiple attributes."""
        viz = BiasVisualizer()
        result = viz.bias_heatmap(
            sample_data,
            protected_attrs=['gender', 'age_group'],
            target_column='approved'
        )

        assert isinstance(result, str)

    def test_findings_timeline(self, sample_data):
        """Should generate findings overview chart."""
        from bias_auditor.core.report import BiasCategory, BiasFindings, BiasSeverity

        findings = [
            BiasFindings(
                category=BiasCategory.REPRESENTATION,
                severity=BiasSeverity.CRITICAL,
                title="Test",
                description="Test",
                affected_attribute="gender"
            ),
            BiasFindings(
                category=BiasCategory.LABEL,
                severity=BiasSeverity.WARNING,
                title="Test2",
                description="Test2",
                affected_attribute="gender"
            ),
        ]

        viz = BiasVisualizer()
        result = viz.findings_timeline(findings)

        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_VIZ, reason="Visualization dependencies not installed")
class TestConvenienceFunctions:
    """Tests for convenience visualization functions."""

    def test_plot_group_distribution(self, sample_data):
        """Should work as convenience function."""
        result = plot_group_distribution(sample_data, 'gender')
        assert isinstance(result, str)

    def test_plot_label_rates(self, sample_data):
        """Should work as convenience function."""
        result = plot_label_rates(sample_data, 'gender', 'approved')
        assert isinstance(result, str)

    def test_plot_category_scores(self, category_scores):
        """Should work as convenience function."""
        result = plot_category_scores(category_scores)
        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_VIZ, reason="Visualization dependencies not installed")
class TestGenerateAllVisualizations:
    """Tests for generate_all_visualizations function."""

    def test_generates_multiple_charts(self, sample_data, category_scores):
        """Should generate multiple visualization types."""
        result = generate_all_visualizations(
            data=sample_data,
            protected_attrs=['gender', 'age_group'],
            target_column='approved',
            category_scores=category_scores,
        )

        assert isinstance(result, dict)
        assert len(result) > 0

        # Should have distribution charts
        assert any('distribution' in k for k in result.keys())

        # Should have label rate charts
        assert any('label_rates' in k for k in result.keys())

        # Should have radar chart
        assert 'category_radar' in result

    def test_handles_missing_target(self, sample_data):
        """Should work without target column."""
        result = generate_all_visualizations(
            data=sample_data,
            protected_attrs=['gender'],
            target_column=None,
        )

        assert isinstance(result, dict)
        # Should still have distribution charts
        assert any('distribution' in k for k in result.keys())
