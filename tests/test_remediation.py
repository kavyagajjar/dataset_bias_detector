"""Tests for remediation strategies."""

import numpy as np
import pandas as pd
import pytest

from bias_auditor.core.report import BiasCategory, BiasFindings, BiasSeverity
from bias_auditor.remediation.resampling import ResamplingRemediation
from bias_auditor.remediation.reweighting import ReweightingRemediation
from bias_auditor.remediation.strategies import (
    RemediationStrategy,
    StrategyType,
    get_all_strategies,
    get_remediation_strategies,
)


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset for testing."""
    np.random.seed(42)

    return pd.DataFrame({
        'gender': ['male'] * 900 + ['female'] * 100,
        'age': np.random.randint(18, 65, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'approved': np.random.choice([0, 1], 1000, p=[0.4, 0.6]),
    })


@pytest.fixture
def label_biased_data():
    """Create dataset with label bias."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.5, 0.5]),
        'feature': np.random.randn(n),
    })

    # Biased labels
    df['approved'] = 0
    df.loc[df['gender'] == 'male', 'approved'] = np.random.choice(
        [0, 1], (df['gender'] == 'male').sum(), p=[0.3, 0.7]
    )
    df.loc[df['gender'] == 'female', 'approved'] = np.random.choice(
        [0, 1], (df['gender'] == 'female').sum(), p=[0.7, 0.3]
    )

    return df


class TestResamplingRemediation:
    """Tests for ResamplingRemediation."""

    def test_random_oversample(self, imbalanced_data):
        """Random oversampling should balance groups."""
        resampler = ResamplingRemediation(random_state=42)

        resampled = resampler.oversample(
            imbalanced_data,
            protected_attr='gender',
            target_ratio=1.0,
            strategy='random'
        )

        # Check that minority group is now balanced
        counts = resampled['gender'].value_counts()
        assert counts['female'] >= counts['male'] * 0.9  # Within 10%

    def test_oversample_preserves_columns(self, imbalanced_data):
        """Resampling should preserve all columns."""
        resampler = ResamplingRemediation()

        resampled = resampler.oversample(imbalanced_data, 'gender')

        assert set(resampled.columns) == set(imbalanced_data.columns)

    def test_undersample(self, imbalanced_data):
        """Undersampling should reduce majority group."""
        resampler = ResamplingRemediation()

        undersampled = resampler.undersample(imbalanced_data, 'gender')

        counts = undersampled['gender'].value_counts()
        # Both groups should now be equal (size of minority)
        assert counts['male'] == counts['female']

    def test_random_state_reproducible(self, imbalanced_data):
        """Same random state should give same results."""
        resampler1 = ResamplingRemediation(random_state=42)
        resampler2 = ResamplingRemediation(random_state=42)

        result1 = resampler1.oversample(imbalanced_data, 'gender')
        result2 = resampler2.oversample(imbalanced_data, 'gender')

        pd.testing.assert_frame_equal(result1, result2)


class TestReweightingRemediation:
    """Tests for ReweightingRemediation."""

    def test_inverse_frequency_weights(self, imbalanced_data):
        """Inverse frequency weights should upweight minority."""
        reweighter = ReweightingRemediation()

        weights = reweighter.inverse_frequency_weights(imbalanced_data, 'gender')

        # Female samples should have higher weights
        female_weights = weights[imbalanced_data['gender'] == 'female']
        male_weights = weights[imbalanced_data['gender'] == 'male']

        assert female_weights.mean() > male_weights.mean()

    def test_balanced_weights(self, imbalanced_data):
        """Balanced weights should equalize group contributions."""
        reweighter = ReweightingRemediation()

        weights = reweighter.balanced_weights(imbalanced_data, 'gender')

        # Weighted sum for each group should be equal
        df_weighted = imbalanced_data.copy()
        df_weighted['weight'] = weights

        male_total = df_weighted[df_weighted['gender'] == 'male']['weight'].sum()
        female_total = df_weighted[df_weighted['gender'] == 'female']['weight'].sum()

        # Should be approximately equal
        assert abs(male_total - female_total) / max(male_total, female_total) < 0.01

    def test_label_balancing_weights(self, label_biased_data):
        """Label balancing should adjust for outcome disparities."""
        reweighter = ReweightingRemediation()

        weights = reweighter.label_balancing_weights(
            label_biased_data,
            protected_attr='gender',
            target_column='approved',
            positive_label=1
        )

        # All weights should be positive
        assert (weights > 0).all()
        # Weights should be roughly normalized
        assert abs(weights.mean() - 1.0) < 0.1

    def test_intersectional_weights(self, imbalanced_data):
        """Intersectional weights should handle multiple attributes."""
        # Add another attribute
        imbalanced_data['age_group'] = pd.cut(
            imbalanced_data['age'],
            bins=[0, 30, 50, 100],
            labels=['young', 'middle', 'senior']
        )

        reweighter = ReweightingRemediation()
        weights = reweighter.intersectional_weights(
            imbalanced_data,
            protected_attrs=['gender', 'age_group']
        )

        assert len(weights) == len(imbalanced_data)
        assert (weights > 0).all()


class TestStrategies:
    """Tests for remediation strategy functions."""

    def test_get_all_strategies(self):
        """Should return list of all strategies."""
        strategies = get_all_strategies()

        assert len(strategies) > 0
        assert all(isinstance(s, RemediationStrategy) for s in strategies)

    def test_get_strategies_for_representation(self):
        """Should return relevant strategies for representation bias."""
        finding = BiasFindings(
            category=BiasCategory.REPRESENTATION,
            severity=BiasSeverity.CRITICAL,
            title="Test finding",
            description="Test description",
            affected_attribute="gender",
        )

        strategies = get_remediation_strategies(finding)

        assert len(strategies) > 0
        # Should include resampling strategies
        assert any(s.strategy_type == StrategyType.RESAMPLING for s in strategies)

    def test_get_strategies_for_label_bias(self):
        """Should return relevant strategies for label bias."""
        finding = BiasFindings(
            category=BiasCategory.LABEL,
            severity=BiasSeverity.CRITICAL,
            title="Test finding",
            description="Test description",
            affected_attribute="gender",
        )

        strategies = get_remediation_strategies(finding)

        assert len(strategies) > 0
        # Should include labeling strategies
        assert any(s.strategy_type == StrategyType.LABELING for s in strategies)

    def test_strategies_sorted_by_effectiveness(self):
        """Strategies should be sorted by effectiveness."""
        finding = BiasFindings(
            category=BiasCategory.REPRESENTATION,
            severity=BiasSeverity.CRITICAL,
            title="Test finding",
            description="Test description",
            affected_attribute="gender",
        )

        strategies = get_remediation_strategies(finding)

        # High effectiveness should come first
        effectiveness_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(strategies) - 1):
            current = effectiveness_order.get(strategies[i].effectiveness, 3)
            next_val = effectiveness_order.get(strategies[i+1].effectiveness, 3)
            assert current <= next_val

    def test_strategy_to_dict(self):
        """Strategy should convert to dictionary."""
        strategies = get_all_strategies()

        for strategy in strategies[:3]:
            d = strategy.to_dict()
            assert 'type' in d
            assert 'name' in d
            assert 'description' in d
            assert 'complexity' in d
            assert 'effectiveness' in d
