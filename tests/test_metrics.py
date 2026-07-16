"""Tests for fairness metrics."""

import numpy as np
import pandas as pd
import pytest

from bias_auditor.metrics.fairness import (
    class_imbalance_ratio,
    disparate_impact_ratio,
    group_label_rates,
    kl_divergence,
    mutual_information,
    normalized_entropy,
    statistical_parity_difference,
)


@pytest.fixture
def balanced_data():
    """Create balanced test data."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.5, 0.5]),
        'approved': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    })


@pytest.fixture
def imbalanced_data():
    """Create imbalanced test data with clear bias."""
    np.random.seed(42)

    df = pd.DataFrame({
        'gender': ['male'] * 80 + ['female'] * 20,
        'approved': [1] * 60 + [0] * 20 + [1] * 4 + [0] * 16,
    })

    return df


class TestStatisticalParity:
    """Tests for statistical parity difference."""

    def test_perfect_parity(self, balanced_data):
        """Test SPD with roughly balanced data."""
        result = statistical_parity_difference(
            balanced_data, 'gender', 'approved', positive_label=1
        )

        # Should be close to 0 for balanced data
        assert abs(result['spd']) < 0.15

    def test_clear_disparity(self, imbalanced_data):
        """Test SPD with clear disparity."""
        result = statistical_parity_difference(
            imbalanced_data, 'gender', 'approved', positive_label=1
        )

        # Male: 60/80 = 0.75, Female: 4/20 = 0.2
        # SPD = 0.2 - 0.75 = -0.55
        assert result['spd'] < -0.4
        assert result['privileged_group'] == 'male'
        assert result['unprivileged_group'] == 'female'


class TestDisparateImpact:
    """Tests for disparate impact ratio."""

    def test_passes_80_rule(self, balanced_data):
        """Test DIR with balanced data passes 80% rule."""
        result = disparate_impact_ratio(
            balanced_data, 'gender', 'approved', positive_label=1
        )

        # Should be close to 1.0 for balanced data
        assert result['passes_80_percent_rule'] or abs(result['dir'] - 1.0) < 0.3

    def test_fails_80_rule(self, imbalanced_data):
        """Test DIR with imbalanced data fails 80% rule."""
        result = disparate_impact_ratio(
            imbalanced_data, 'gender', 'approved', positive_label=1
        )

        # DIR = 0.2 / 0.75 = 0.267 < 0.8
        assert result['dir'] < 0.4
        assert result['passes_80_percent_rule'] is False


class TestClassImbalance:
    """Tests for class imbalance ratio."""

    def test_balanced(self, balanced_data):
        """Test imbalance ratio for balanced data."""
        result = class_imbalance_ratio(balanced_data, 'gender')

        # Should be close to 1.0
        assert result['ratio'] < 1.3

    def test_imbalanced(self, imbalanced_data):
        """Test imbalance ratio for imbalanced data."""
        result = class_imbalance_ratio(imbalanced_data, 'gender')

        # 80/20 = 4.0
        assert result['ratio'] == 4.0
        assert result['majority_class'] == 'male'
        assert result['minority_class'] == 'female'


class TestNormalizedEntropy:
    """Tests for normalized entropy."""

    def test_uniform_distribution(self):
        """Test entropy for uniform distribution."""
        df = pd.DataFrame({'col': ['A', 'B', 'C', 'D'] * 25})
        entropy = normalized_entropy(df, 'col')

        # Should be close to 1.0 for uniform
        assert entropy > 0.99

    def test_skewed_distribution(self):
        """Test entropy for skewed distribution."""
        df = pd.DataFrame({'col': ['A'] * 90 + ['B'] * 10})
        entropy = normalized_entropy(df, 'col')

        # Should be low for skewed
        assert entropy < 0.6


class TestKLDivergence:
    """Tests for KL divergence."""

    def test_identical_distributions(self):
        """Test KL divergence for identical distributions."""
        dist = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        kl = kl_divergence(dist, dist)

        # Should be 0 for identical
        assert kl < 0.01

    def test_different_distributions(self):
        """Test KL divergence for different distributions."""
        observed = {'A': 0.8, 'B': 0.1, 'C': 0.1}
        expected = {'A': 0.33, 'B': 0.33, 'C': 0.34}

        kl = kl_divergence(observed, expected)

        # Should be positive for different
        assert kl > 0.3


class TestGroupLabelRates:
    """Tests for group label rates."""

    def test_rates_calculation(self, imbalanced_data):
        """Test group label rates calculation."""
        rates = group_label_rates(
            imbalanced_data, 'gender', 'approved', positive_label=1
        )

        assert 'male' in rates
        assert 'female' in rates
        assert rates['male']['positive_rate'] == 0.75
        assert rates['female']['positive_rate'] == 0.2


class TestMutualInformation:
    """Tests for mutual information."""

    def test_independent_variables(self):
        """Test MI for independent variables."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.choice(['X', 'Y'], 200),
            'protected': np.random.choice(['A', 'B'], 200),
        })

        mi = mutual_information(df, 'feature', 'protected')

        # Should be low for independent
        assert mi < 0.1

    def test_correlated_variables(self):
        """Test MI for correlated variables."""
        df = pd.DataFrame({
            'feature': ['X'] * 100 + ['Y'] * 100,
            'protected': ['A'] * 100 + ['B'] * 100,
        })

        mi = mutual_information(df, 'feature', 'protected')

        # Should be high (close to 1) for perfectly correlated
        assert mi > 0.9
