"""Remediation strategy definitions and recommendations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from bias_auditor.core.report import BiasCategory, BiasFindings


class StrategyType(str, Enum):
    """Types of remediation strategies."""

    RESAMPLING = "resampling"
    REWEIGHTING = "reweighting"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_COLLECTION = "data_collection"
    LABELING = "labeling"


@dataclass
class RemediationStrategy:
    """A remediation strategy for addressing bias."""

    strategy_type: StrategyType
    name: str
    description: str
    applicable_categories: list[BiasCategory]
    implementation_complexity: str  # "low", "medium", "high"
    effectiveness: str  # "low", "medium", "high"
    code_template: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.strategy_type.value,
            "name": self.name,
            "description": self.description,
            "applicable_categories": [c.value for c in self.applicable_categories],
            "complexity": self.implementation_complexity,
            "effectiveness": self.effectiveness,
        }


# Pre-defined remediation strategies
STRATEGIES = [
    # Resampling strategies
    RemediationStrategy(
        strategy_type=StrategyType.RESAMPLING,
        name="Random Oversampling",
        description="Duplicate samples from minority groups to balance representation.",
        applicable_categories=[BiasCategory.REPRESENTATION, BiasCategory.LABEL],
        implementation_complexity="low",
        effectiveness="medium",
        code_template="""
from sklearn.utils import resample

def oversample_minority(df, protected_attr, target_ratio=1.0):
    '''Oversample minority groups to achieve target ratio.'''
    majority_group = df[protected_attr].value_counts().idxmax()
    majority_size = df[df[protected_attr] == majority_group].shape[0]

    balanced_dfs = [df[df[protected_attr] == majority_group]]

    for group in df[protected_attr].unique():
        if group != majority_group:
            group_df = df[df[protected_attr] == group]
            target_size = int(majority_size * target_ratio)
            oversampled = resample(group_df, n_samples=target_size, random_state=42)
            balanced_dfs.append(oversampled)

    return pd.concat(balanced_dfs, ignore_index=True)
""",
    ),
    RemediationStrategy(
        strategy_type=StrategyType.RESAMPLING,
        name="SMOTE",
        description="Synthetic Minority Over-sampling Technique - creates synthetic samples.",
        applicable_categories=[BiasCategory.REPRESENTATION, BiasCategory.LABEL],
        implementation_complexity="medium",
        effectiveness="high",
        code_template="""
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, protected_attr_col):
    '''Apply SMOTE for synthetic oversampling.'''
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
""",
    ),
    RemediationStrategy(
        strategy_type=StrategyType.RESAMPLING,
        name="Random Undersampling",
        description="Remove samples from majority groups to balance representation.",
        applicable_categories=[BiasCategory.REPRESENTATION],
        implementation_complexity="low",
        effectiveness="medium",
        code_template="""
from sklearn.utils import resample

def undersample_majority(df, protected_attr):
    '''Undersample majority groups to match minority.'''
    minority_size = df[protected_attr].value_counts().min()

    balanced_dfs = []
    for group in df[protected_attr].unique():
        group_df = df[df[protected_attr] == group]
        undersampled = resample(group_df, n_samples=minority_size, random_state=42)
        balanced_dfs.append(undersampled)

    return pd.concat(balanced_dfs, ignore_index=True)
""",
    ),

    # Reweighting strategies
    RemediationStrategy(
        strategy_type=StrategyType.REWEIGHTING,
        name="Inverse Frequency Weighting",
        description="Assign higher weights to underrepresented groups.",
        applicable_categories=[BiasCategory.REPRESENTATION, BiasCategory.LABEL],
        implementation_complexity="low",
        effectiveness="medium",
        code_template="""
def compute_sample_weights(df, protected_attr):
    '''Compute inverse frequency weights.'''
    counts = df[protected_attr].value_counts()
    total = len(df)
    weights = df[protected_attr].map(lambda x: total / (len(counts) * counts[x]))
    return weights
""",
    ),
    RemediationStrategy(
        strategy_type=StrategyType.REWEIGHTING,
        name="Equalized Odds Reweighting",
        description="Compute weights to achieve equalized odds across groups.",
        applicable_categories=[BiasCategory.LABEL],
        implementation_complexity="high",
        effectiveness="high",
        code_template="""
from aif360.algorithms.preprocessing import Reweighing

def equalized_reweighting(df, protected_attr, target_col):
    '''Apply fairness-aware reweighting.'''
    # Requires AIF360 library
    # See: https://aif360.readthedocs.io/
    pass
""",
    ),

    # Preprocessing strategies
    RemediationStrategy(
        strategy_type=StrategyType.PREPROCESSING,
        name="Disparate Impact Remover",
        description="Transform features to remove correlation with protected attributes.",
        applicable_categories=[BiasCategory.FEATURE_PROXY],
        implementation_complexity="medium",
        effectiveness="high",
        code_template="""
from aif360.algorithms.preprocessing import DisparateImpactRemover

def remove_disparate_impact(df, protected_attr, features):
    '''Remove disparate impact from features.'''
    # Requires AIF360 library
    di_remover = DisparateImpactRemover(repair_level=1.0)
    # Transform features to reduce proxy effect
    pass
""",
    ),
    RemediationStrategy(
        strategy_type=StrategyType.PREPROCESSING,
        name="Missing Data Imputation by Group",
        description="Impute missing values separately for each protected group.",
        applicable_categories=[BiasCategory.MISSING_DATA],
        implementation_complexity="medium",
        effectiveness="medium",
        code_template="""
from sklearn.impute import SimpleImputer

def impute_by_group(df, protected_attr, columns_to_impute):
    '''Impute missing values separately per group.'''
    result = df.copy()
    for group in df[protected_attr].unique():
        mask = df[protected_attr] == group
        imputer = SimpleImputer(strategy='median')
        result.loc[mask, columns_to_impute] = imputer.fit_transform(
            df.loc[mask, columns_to_impute]
        )
    return result
""",
    ),

    # Feature engineering strategies
    RemediationStrategy(
        strategy_type=StrategyType.FEATURE_ENGINEERING,
        name="Remove Proxy Features",
        description="Remove features highly correlated with protected attributes.",
        applicable_categories=[BiasCategory.FEATURE_PROXY],
        implementation_complexity="low",
        effectiveness="medium",
        code_template="""
def remove_proxy_features(df, proxy_features):
    '''Remove identified proxy features.'''
    return df.drop(columns=proxy_features)
""",
    ),
    RemediationStrategy(
        strategy_type=StrategyType.FEATURE_ENGINEERING,
        name="Binning/Discretization",
        description="Reduce granularity of proxy features through binning.",
        applicable_categories=[BiasCategory.FEATURE_PROXY],
        implementation_complexity="low",
        effectiveness="low",
        code_template="""
def bin_proxy_feature(df, feature, n_bins=5):
    '''Reduce proxy effect through discretization.'''
    df[f'{feature}_binned'] = pd.qcut(df[feature], q=n_bins, labels=False)
    return df.drop(columns=[feature])
""",
    ),

    # Data collection strategies
    RemediationStrategy(
        strategy_type=StrategyType.DATA_COLLECTION,
        name="Stratified Data Collection",
        description="Collect additional data to balance group representation.",
        applicable_categories=[BiasCategory.REPRESENTATION, BiasCategory.INTERSECTIONAL],
        implementation_complexity="high",
        effectiveness="high",
        code_template=None,  # Not automatable
    ),

    # Labeling strategies
    RemediationStrategy(
        strategy_type=StrategyType.LABELING,
        name="Blind Labeling Protocol",
        description="Remove protected attribute information during labeling.",
        applicable_categories=[BiasCategory.LABEL],
        implementation_complexity="medium",
        effectiveness="high",
        code_template=None,  # Process change, not code
    ),
    RemediationStrategy(
        strategy_type=StrategyType.LABELING,
        name="Multi-Annotator Agreement",
        description="Use multiple annotators and require consensus.",
        applicable_categories=[BiasCategory.LABEL],
        implementation_complexity="medium",
        effectiveness="medium",
        code_template=None,
    ),
]


def get_remediation_strategies(finding: BiasFindings) -> list[RemediationStrategy]:
    """
    Get applicable remediation strategies for a finding.

    Parameters
    ----------
    finding : BiasFindings
        The bias finding to remediate.

    Returns
    -------
    list[RemediationStrategy]
        Applicable strategies, sorted by effectiveness.
    """
    applicable = [
        s for s in STRATEGIES
        if finding.category in s.applicable_categories
    ]

    # Sort by effectiveness
    effectiveness_order = {"high": 0, "medium": 1, "low": 2}
    applicable.sort(key=lambda s: effectiveness_order.get(s.effectiveness, 3))

    return applicable


def get_all_strategies() -> list[RemediationStrategy]:
    """Get all available remediation strategies."""
    return STRATEGIES.copy()
