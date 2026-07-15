"""Tests for bias detectors."""

import pytest
import pandas as pd
import numpy as np

from bias_auditor.detectors.representation import RepresentationDetector
from bias_auditor.detectors.label_bias import LabelBiasDetector
from bias_auditor.detectors.feature_proxy import FeatureProxyDetector
from bias_auditor.detectors.missing_data import MissingDataDetector
from bias_auditor.core.config import AuditConfig, BiasThresholds
from bias_auditor.core.report import BiasSeverity, BiasCategory


@pytest.fixture
def balanced_data():
    """Create balanced dataset."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.5, 0.5]),
        'age_group': np.random.choice(['young', 'middle', 'senior'], n, p=[0.33, 0.34, 0.33]),
        'income': np.random.normal(50000, 15000, n),
        'approved': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })


@pytest.fixture
def imbalanced_data():
    """Create dataset with severe imbalance."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.95, 0.05]),  # Severe imbalance
        'race': np.random.choice(['A', 'B', 'C'], n, p=[0.8, 0.15, 0.05]),
        'feature': np.random.randn(n),
        'approved': np.zeros(n, dtype=int),
    })
    
    # Add label bias - males get approved much more
    df.loc[df['gender'] == 'male', 'approved'] = np.random.choice(
        [0, 1], (df['gender'] == 'male').sum(), p=[0.2, 0.8]
    )
    df.loc[df['gender'] == 'female', 'approved'] = np.random.choice(
        [0, 1], (df['gender'] == 'female').sum(), p=[0.9, 0.1]
    )
    
    return df


@pytest.fixture
def proxy_data():
    """Create dataset with proxy features."""
    np.random.seed(42)
    n = 500
    
    # Gender determines zip code (proxy relationship)
    gender = np.random.choice(['male', 'female'], n, p=[0.5, 0.5])
    zip_code = np.where(
        gender == 'male',
        np.random.choice(['10001', '10002', '10003'], n, p=[0.7, 0.2, 0.1]),
        np.random.choice(['10001', '10002', '10003'], n, p=[0.1, 0.3, 0.6])
    )
    
    return pd.DataFrame({
        'gender': gender,
        'zip_code': zip_code,
        'age': np.random.randint(18, 65, n),
        'approved': np.random.choice([0, 1], n),
    })


@pytest.fixture
def missing_data():
    """Create dataset with non-random missing values."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n, p=[0.5, 0.5]),
        'income': np.random.normal(50000, 15000, n),
        'credit_score': np.random.randint(300, 850, n),
        'approved': np.random.choice([0, 1], n),
    })
    
    # Missing income more likely for females (differential missingness)
    female_mask = df['gender'] == 'female'
    missing_prob = np.where(female_mask, 0.3, 0.05)  # 30% vs 5%
    missing_mask = np.random.random(n) < missing_prob
    df.loc[missing_mask, 'income'] = np.nan
    
    return df


class TestRepresentationDetector:
    """Tests for RepresentationDetector."""
    
    def test_balanced_no_findings(self, balanced_data):
        """Balanced data should have few/no critical findings."""
        config = AuditConfig(protected_attributes=['gender', 'age_group'])
        detector = RepresentationDetector(config)
        
        findings = detector.detect(balanced_data)
        critical = [f for f in findings if f.severity == BiasSeverity.CRITICAL]
        
        assert len(critical) == 0
    
    def test_severe_imbalance_detected(self, imbalanced_data):
        """Severe imbalance should be detected as critical."""
        config = AuditConfig(protected_attributes=['gender'])
        detector = RepresentationDetector(config)
        
        findings = detector.detect(imbalanced_data)
        critical = [f for f in findings if f.severity == BiasSeverity.CRITICAL]
        
        # Should detect severe underrepresentation of females
        assert len(critical) >= 1
        assert any('female' in str(f.affected_groups).lower() for f in critical)
    
    def test_imbalance_ratio_calculated(self, imbalanced_data):
        """Should correctly calculate imbalance ratio."""
        config = AuditConfig(protected_attributes=['gender'])
        detector = RepresentationDetector(config)
        
        findings = detector.detect(imbalanced_data)
        imbalance_findings = [f for f in findings if 'imbalance' in f.title.lower()]
        
        assert len(imbalance_findings) >= 1
        # 95:5 ratio should be flagged
        assert any(f.metrics.get('imbalance_ratio', 0) > 5 for f in imbalance_findings)
    
    def test_remediation_suggestions_provided(self, imbalanced_data):
        """Critical findings should include remediation suggestions."""
        config = AuditConfig(protected_attributes=['gender'])
        detector = RepresentationDetector(config)
        
        findings = detector.detect(imbalanced_data)
        critical = [f for f in findings if f.severity == BiasSeverity.CRITICAL]
        
        for finding in critical:
            assert len(finding.remediation_suggestions) > 0


class TestLabelBiasDetector:
    """Tests for LabelBiasDetector."""
    
    def test_disparate_impact_detected(self, imbalanced_data):
        """Should detect disparate impact in labels."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved',
            positive_label=1
        )
        detector = LabelBiasDetector(config)
        
        findings = detector.detect(imbalanced_data)
        di_findings = [f for f in findings if 'disparate impact' in f.title.lower()]
        
        assert len(di_findings) >= 1
    
    def test_statistical_parity_calculated(self, imbalanced_data):
        """Should calculate statistical parity difference."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved',
            positive_label=1
        )
        detector = LabelBiasDetector(config)
        
        findings = detector.detect(imbalanced_data)
        spd_findings = [f for f in findings if 'statistical parity' in f.title.lower()]
        
        assert len(spd_findings) >= 1
        # Male 80% vs Female 10% = 70% difference
        assert any(abs(f.metrics.get('abs_spd', 0)) > 0.5 for f in spd_findings)
    
    def test_no_label_bias_without_target(self, balanced_data):
        """Should return no findings without target column."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column=None  # No target
        )
        detector = LabelBiasDetector(config)
        
        findings = detector.detect(balanced_data)
        
        assert len(findings) == 0


class TestFeatureProxyDetector:
    """Tests for FeatureProxyDetector."""
    
    def test_proxy_detected(self, proxy_data):
        """Should detect zip_code as proxy for gender."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved'
        )
        detector = FeatureProxyDetector(config)
        
        findings = detector.detect(proxy_data)
        proxy_findings = [f for f in findings if f.category == BiasCategory.FEATURE_PROXY]
        
        # Should find zip_code as potential proxy
        assert len(proxy_findings) >= 1
        assert any('zip' in str(f.affected_groups).lower() for f in proxy_findings)
    
    def test_known_pattern_flagged(self, proxy_data):
        """Should flag known proxy patterns like zip codes."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved'
        )
        detector = FeatureProxyDetector(config)
        
        findings = detector.detect(proxy_data)
        pattern_findings = [f for f in findings if 'pattern' in f.title.lower()]
        
        # zip_code matches 'zip' pattern
        assert len(pattern_findings) >= 1


class TestMissingDataDetector:
    """Tests for MissingDataDetector."""
    
    def test_differential_missing_detected(self, missing_data):
        """Should detect differential missing rates."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved'
        )
        detector = MissingDataDetector(config)
        
        findings = detector.detect(missing_data)
        diff_findings = [f for f in findings if 'differential' in f.title.lower()]
        
        # Should detect income has differential missingness by gender
        assert len(diff_findings) >= 1
    
    def test_high_missing_flagged(self, missing_data):
        """Should flag columns with high missing rates."""
        # Add column with >30% missing
        data = missing_data.copy()
        data['high_missing'] = np.nan
        data.loc[data.index[:150], 'high_missing'] = 1.0  # 70% missing
        
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved'
        )
        detector = MissingDataDetector(config)
        
        findings = detector.detect(data)
        high_missing = [f for f in findings if 'high missing' in f.title.lower()]
        
        assert len(high_missing) >= 1
    
    def test_no_missing_no_findings(self, balanced_data):
        """Complete data should have no missing data findings."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved'
        )
        detector = MissingDataDetector(config)
        
        findings = detector.detect(balanced_data)
        # May have informational findings but no critical
        critical = [f for f in findings if f.severity == BiasSeverity.CRITICAL]
        
        assert len(critical) == 0


class TestIntegration:
    """Integration tests across detectors."""
    
    def test_all_detectors_run(self, imbalanced_data):
        """All detectors should run without error."""
        config = AuditConfig(
            protected_attributes=['gender', 'race'],
            target_column='approved',
            positive_label=1
        )
        
        detectors = [
            RepresentationDetector(config),
            LabelBiasDetector(config),
            FeatureProxyDetector(config),
            MissingDataDetector(config),
        ]
        
        all_findings = []
        for detector in detectors:
            findings = detector.detect(imbalanced_data)
            all_findings.extend(findings)
        
        # Should have multiple findings across categories
        categories = set(f.category for f in all_findings)
        assert len(categories) >= 2
    
    def test_findings_have_required_fields(self, imbalanced_data):
        """All findings should have required fields populated."""
        config = AuditConfig(
            protected_attributes=['gender'],
            target_column='approved',
            positive_label=1
        )
        
        detector = RepresentationDetector(config)
        findings = detector.detect(imbalanced_data)
        
        for finding in findings:
            assert finding.category is not None
            assert finding.severity is not None
            assert finding.title
            assert finding.description
            assert finding.affected_attribute
