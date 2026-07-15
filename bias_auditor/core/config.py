"""Configuration and threshold settings for bias auditing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"
    NONE = "none"


@dataclass
class BiasThresholds:
    """
    Thresholds for determining bias severity levels.
    
    Based on industry standards and legal guidelines (e.g., 80% rule for disparate impact).
    """
    
    # Disparate Impact Ratio thresholds
    # < 0.8 is typically considered adverse impact (EEOC 80% rule)
    disparate_impact_warning: float = 0.9
    disparate_impact_critical: float = 0.8
    
    # Statistical Parity Difference thresholds
    # |SPD| > 0.1 often indicates significant disparity
    statistical_parity_warning: float = 0.1
    statistical_parity_critical: float = 0.2
    
    # Class imbalance ratio thresholds
    # Ratio of majority to minority class
    imbalance_ratio_warning: float = 3.0
    imbalance_ratio_critical: float = 5.0
    
    # Representation thresholds (minimum group proportion)
    min_group_proportion_warning: float = 0.1
    min_group_proportion_critical: float = 0.05
    
    # Missing data disparity thresholds
    # Difference in missing rates between groups
    missing_rate_disparity_warning: float = 0.1
    missing_rate_disparity_critical: float = 0.2
    
    # Proxy variable detection thresholds
    # Correlation with protected attribute
    proxy_correlation_warning: float = 0.5
    proxy_correlation_critical: float = 0.7
    
    # Mutual information threshold for proxy detection
    proxy_mutual_info_warning: float = 0.3
    proxy_mutual_info_critical: float = 0.5
    
    # Label bias thresholds
    label_rate_disparity_warning: float = 0.1
    label_rate_disparity_critical: float = 0.2
    
    def to_dict(self) -> dict[str, float]:
        """Convert thresholds to dictionary."""
        return {
            "disparate_impact_warning": self.disparate_impact_warning,
            "disparate_impact_critical": self.disparate_impact_critical,
            "statistical_parity_warning": self.statistical_parity_warning,
            "statistical_parity_critical": self.statistical_parity_critical,
            "imbalance_ratio_warning": self.imbalance_ratio_warning,
            "imbalance_ratio_critical": self.imbalance_ratio_critical,
            "min_group_proportion_warning": self.min_group_proportion_warning,
            "min_group_proportion_critical": self.min_group_proportion_critical,
            "missing_rate_disparity_warning": self.missing_rate_disparity_warning,
            "missing_rate_disparity_critical": self.missing_rate_disparity_critical,
            "proxy_correlation_warning": self.proxy_correlation_warning,
            "proxy_correlation_critical": self.proxy_correlation_critical,
            "proxy_mutual_info_warning": self.proxy_mutual_info_warning,
            "proxy_mutual_info_critical": self.proxy_mutual_info_critical,
            "label_rate_disparity_warning": self.label_rate_disparity_warning,
            "label_rate_disparity_critical": self.label_rate_disparity_critical,
        }


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    
    provider: LLMProvider = LLMProvider.NONE
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # Feature flags for LLM capabilities
    enable_text_analysis: bool = True
    enable_explanations: bool = True
    enable_code_generation: bool = True
    enable_remediation_suggestions: bool = True
    
    # Caching
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class AuditConfig:
    """
    Main configuration for the bias auditor.
    
    Parameters
    ----------
    protected_attributes : list[str]
        Column names containing protected/sensitive attributes (e.g., gender, race, age).
    target_column : str, optional
        The target/label column for supervised learning tasks.
    positive_label : Any, optional
        The value considered as the positive outcome (e.g., 1, 'approved', True).
    reference_distributions : dict, optional
        Expected population distributions for comparison.
    thresholds : BiasThresholds, optional
        Custom threshold values for bias detection.
    llm_config : LLMConfig, optional
        Configuration for LLM features.
    """
    
    protected_attributes: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    positive_label: Any = 1
    
    # Reference population distributions for comparison
    # Format: {"attribute_name": {"group1": proportion1, "group2": proportion2}}
    reference_distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Columns to exclude from analysis
    exclude_columns: list[str] = field(default_factory=list)
    
    # Columns containing free text (for LLM analysis)
    text_columns: list[str] = field(default_factory=list)
    
    # Threshold configuration
    thresholds: BiasThresholds = field(default_factory=BiasThresholds)
    
    # LLM configuration
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    
    # Analysis settings
    compute_intersectional: bool = True  # Analyze intersections of protected attributes
    max_intersectional_depth: int = 2  # Max number of attributes to combine
    min_group_size: int = 30  # Minimum samples to consider a group
    
    # Output settings
    verbose: bool = True
    generate_visualizations: bool = True
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        
        if not self.protected_attributes:
            warnings.append("No protected attributes specified. "
                          "Consider adding attributes like 'gender', 'race', 'age_group'.")
        
        if self.target_column is None:
            warnings.append("No target column specified. "
                          "Label bias detection will be skipped.")
        
        if self.llm_config.provider != LLMProvider.NONE and not self.llm_config.api_key:
            warnings.append(f"LLM provider '{self.llm_config.provider}' selected but no API key provided. "
                          "Set via api_key or environment variable.")
        
        return warnings
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "AuditConfig":
        """Create config from dictionary."""
        # Handle nested configs
        if "thresholds" in config_dict and isinstance(config_dict["thresholds"], dict):
            config_dict["thresholds"] = BiasThresholds(**config_dict["thresholds"])
        
        if "llm_config" in config_dict and isinstance(config_dict["llm_config"], dict):
            llm_dict = config_dict["llm_config"]
            if "provider" in llm_dict and isinstance(llm_dict["provider"], str):
                llm_dict["provider"] = LLMProvider(llm_dict["provider"])
            config_dict["llm_config"] = LLMConfig(**llm_dict)
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "protected_attributes": self.protected_attributes,
            "target_column": self.target_column,
            "positive_label": self.positive_label,
            "reference_distributions": self.reference_distributions,
            "exclude_columns": self.exclude_columns,
            "text_columns": self.text_columns,
            "thresholds": self.thresholds.to_dict(),
            "llm_config": {
                "provider": self.llm_config.provider.value,
                "model": self.llm_config.model,
                "enable_text_analysis": self.llm_config.enable_text_analysis,
                "enable_explanations": self.llm_config.enable_explanations,
                "enable_code_generation": self.llm_config.enable_code_generation,
            },
            "compute_intersectional": self.compute_intersectional,
            "max_intersectional_depth": self.max_intersectional_depth,
            "min_group_size": self.min_group_size,
        }
