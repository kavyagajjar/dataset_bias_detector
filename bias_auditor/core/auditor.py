"""Main BiasAuditor class - the primary interface for bias detection."""

import warnings
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from bias_auditor.core.config import AuditConfig, BiasThresholds, LLMConfig, LLMProvider
from bias_auditor.core.report import (
    AuditReport,
    BiasCategory,
    BiasSeverity,
    DatasetProfile,
)


class BiasAuditor:
    """
    Main class for detecting biases in datasets before model training.

    The BiasAuditor analyzes datasets for various types of bias including:
    - Representation bias (underrepresented groups)
    - Label bias (systematic labeling differences)
    - Feature proxy bias (features that encode protected attributes)
    - Missing data bias (non-random missingness patterns)
    - Text bias (stereotypes in text columns, requires LLM)

    Parameters
    ----------
    protected_attributes : list[str]
        Column names containing protected/sensitive attributes.
    target_column : str, optional
        The target/label column for supervised learning tasks.
    positive_label : Any, optional
        The value considered as the positive outcome. Default is 1.
    thresholds : BiasThresholds, optional
        Custom threshold values for bias detection.
    llm_provider : str, optional
        LLM provider for enhanced analysis: 'openai', 'anthropic', 'local', or None.
    llm_model : str, optional
        Specific model to use (e.g., 'gpt-4o', 'claude-opus-4-8').
    llm_api_key : str, optional
        API key for the LLM provider.
    config : AuditConfig, optional
        Full configuration object (overrides individual parameters).

    Examples
    --------
    Basic usage:

    >>> auditor = BiasAuditor(
    ...     protected_attributes=['gender', 'race'],
    ...     target_column='approved'
    ... )
    >>> report = auditor.audit(df)
    >>> print(report.summary())

    With LLM integration:

    >>> auditor = BiasAuditor(
    ...     protected_attributes=['gender', 'race'],
    ...     target_column='approved',
    ...     llm_provider='openai',
    ...     llm_api_key='sk-...'
    ... )
    >>> report = auditor.audit(df)
    >>> report.explain("Why is zip_code flagged?")
    """

    def __init__(
        self,
        protected_attributes: Optional[list[str]] = None,
        target_column: Optional[str] = None,
        positive_label: Any = 1,
        thresholds: Optional[BiasThresholds] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        config: Optional[AuditConfig] = None,
        auto_detect: bool = False,
        verbose: bool = True,
    ):
        # Use provided config or build from parameters
        if config is not None:
            self.config = config
        else:
            # Build LLM config
            llm_config = LLMConfig(
                provider=LLMProvider(llm_provider) if llm_provider else LLMProvider.NONE,
                model=llm_model or "gpt-4o",
                api_key=llm_api_key,
            )

            self.config = AuditConfig(
                protected_attributes=protected_attributes or [],
                target_column=target_column,
                positive_label=positive_label,
                thresholds=thresholds or BiasThresholds(),
                llm_config=llm_config,
                auto_detect=auto_detect,
                verbose=verbose,
            )

        # Validate configuration
        config_warnings = self.config.validate()
        if config_warnings and self.config.verbose:
            for warning in config_warnings:
                warnings.warn(warning, UserWarning, stacklevel=2)

        # Initialize detector modules (lazy loading)
        self._representation_detector = None
        self._label_bias_detector = None
        self._feature_proxy_detector = None
        self._missing_data_detector = None
        self._text_analyzer = None
        self._llm_explainer = None

    @property
    def representation_detector(self):
        """Lazy load representation bias detector."""
        if self._representation_detector is None:
            from bias_auditor.detectors.representation import RepresentationDetector
            self._representation_detector = RepresentationDetector(self.config)
        return self._representation_detector

    @property
    def label_bias_detector(self):
        """Lazy load label bias detector."""
        if self._label_bias_detector is None:
            from bias_auditor.detectors.label_bias import LabelBiasDetector
            self._label_bias_detector = LabelBiasDetector(self.config)
        return self._label_bias_detector

    @property
    def feature_proxy_detector(self):
        """Lazy load feature proxy detector."""
        if self._feature_proxy_detector is None:
            from bias_auditor.detectors.feature_proxy import FeatureProxyDetector
            self._feature_proxy_detector = FeatureProxyDetector(self.config)
        return self._feature_proxy_detector

    @property
    def missing_data_detector(self):
        """Lazy load missing data bias detector."""
        if self._missing_data_detector is None:
            from bias_auditor.detectors.missing_data import MissingDataDetector
            self._missing_data_detector = MissingDataDetector(self.config)
        return self._missing_data_detector

    @property
    def text_analyzer(self):
        """Lazy load text analyzer (requires LLM)."""
        if self._text_analyzer is None and self.config.llm_config.provider != LLMProvider.NONE:
            from bias_auditor.llm.text_analyzer import TextBiasAnalyzer
            self._text_analyzer = TextBiasAnalyzer(self.config.llm_config)
        return self._text_analyzer

    @property
    def llm_explainer(self):
        """Lazy load LLM explainer."""
        if self._llm_explainer is None and self.config.llm_config.provider != LLMProvider.NONE:
            from bias_auditor.llm.explainer import BiasExplainer
            self._llm_explainer = BiasExplainer(self.config.llm_config)
        return self._llm_explainer

    def audit(
        self,
        data: pd.DataFrame,
        dataset_name: Optional[str] = None,
        skip_detectors: Optional[list[str]] = None,
    ) -> AuditReport:
        """
        Perform a complete bias audit on the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to audit.
        dataset_name : str, optional
            Name for the dataset in the report.
        skip_detectors : list[str], optional
            Detectors to skip: 'representation', 'label', 'proxy', 'missing', 'text'

        Returns
        -------
        AuditReport
            Complete audit report with findings and recommendations.
        """
        skip_detectors = skip_detectors or []

        # Auto-detect protected attributes / target if requested
        detection_summary = None
        if self.config.auto_detect and not self.config.protected_attributes:
            from bias_auditor.core.auto_detect import auto_detect

            data, detection = auto_detect(data, target_column=self.config.target_column)
            self.config.protected_attributes = detection.protected_attributes
            self.config.target_column = detection.target_column
            if detection.positive_label is not None:
                self.config.positive_label = detection.positive_label
            detection_summary = detection.to_dict()

            if self.config.verbose:
                print("🔎 Auto-detection results:")
                for note in detection.notes:
                    print(f"   • {note}")
            if detection.target_column is None:
                warnings.warn(
                    "Auto-detection found no binary target column, so label-bias "
                    "checks (disparate impact, 80% rule) will be SKIPPED. If the "
                    "dataset has an outcome column, pass target_column explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
            if not detection.protected_attributes:
                raise ValueError(
                    "Auto-detection found no protected attribute columns. "
                    "Specify protected_attributes explicitly."
                )

        # Create report
        report = AuditReport(
            audit_id=str(uuid4())[:8],
            audit_timestamp=datetime.now(),
            dataset_name=dataset_name,
            config_summary=self.config.to_dict(),
        )
        if detection_summary:
            report.config_summary["auto_detection"] = detection_summary

        # Validate data
        self._validate_data(data)

        # Create dataset profile and per-group statistics
        report.profile = self._create_profile(data)
        report.group_stats = self._compute_group_stats(data)

        if self.config.verbose:
            print(f"🔍 Starting bias audit on {len(data)} rows, {len(data.columns)} columns")
            print(f"   Protected attributes: {self.config.protected_attributes}")
            if self.config.target_column:
                print(f"   Target column: {self.config.target_column}")

        # Run detectors
        if "representation" not in skip_detectors:
            if self.config.verbose:
                print("   → Checking representation bias...")
            findings = self.representation_detector.detect(data)
            for finding in findings:
                report.add_finding(finding)

        if "label" not in skip_detectors and self.config.target_column:
            if self.config.verbose:
                print("   → Checking label bias...")
            findings = self.label_bias_detector.detect(data)
            for finding in findings:
                report.add_finding(finding)

        if "proxy" not in skip_detectors:
            if self.config.verbose:
                print("   → Checking feature proxy bias...")
            findings = self.feature_proxy_detector.detect(data)
            for finding in findings:
                report.add_finding(finding)

        if "missing" not in skip_detectors:
            if self.config.verbose:
                print("   → Checking missing data bias...")
            findings = self.missing_data_detector.detect(data)
            for finding in findings:
                report.add_finding(finding)

        # Text analysis (requires LLM)
        if ("text" not in skip_detectors and
            self.text_analyzer is not None and
            self.config.text_columns):
            if self.config.verbose:
                print("   → Analyzing text columns for bias...")
            findings = self.text_analyzer.analyze(data, self.config.text_columns)
            for finding in findings:
                report.add_finding(finding)

        # Generate embedded visualizations for the HTML report
        if self.config.generate_visualizations:
            if self.config.verbose:
                print("   → Generating visualizations...")
            try:
                from bias_auditor.visualizations import generate_all_visualizations

                report.visualizations = generate_all_visualizations(
                    data=data,
                    protected_attrs=self.config.protected_attributes,
                    target_column=self.config.target_column,
                    positive_label=self.config.positive_label,
                    category_scores=report.category_scores,
                )
            except ImportError:
                if self.config.verbose:
                    print("     (skipped: matplotlib/plotly not installed)")

        # Generate LLM explanations if available
        if self.llm_explainer is not None and self.config.llm_config.enable_explanations:
            if self.config.verbose:
                print("   → Generating explanations...")
            self._add_llm_explanations(report)

        # Generate executive summary
        if self.llm_explainer is not None:
            report.executive_summary = self.llm_explainer.generate_summary(report)

        if self.config.verbose:
            print(f"✅ Audit complete: {len(report.findings)} findings")
            print(f"   Bias score: {report.overall_bias_score:.2f}")

        return report

    def quick_check(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Perform a quick bias check and return summary metrics.

        Faster than full audit, returns key metrics only.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to check.

        Returns
        -------
        dict
            Dictionary with bias indicators and scores.
        """
        results = {
            "n_rows": len(data),
            "protected_attributes": {},
            "has_critical_bias": False,
            "key_metrics": {},
        }

        # Check each protected attribute
        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue

            # Distribution
            dist = data[attr].value_counts(normalize=True).to_dict()
            results["protected_attributes"][attr] = {
                "distribution": dist,
                "n_groups": len(dist),
                "min_proportion": min(dist.values()),
                "max_proportion": max(dist.values()),
            }

            # Check imbalance
            imbalance = max(dist.values()) / min(dist.values())
            if imbalance > self.config.thresholds.imbalance_ratio_critical:
                results["has_critical_bias"] = True

            results["key_metrics"][f"{attr}_imbalance_ratio"] = imbalance

        # Label rate by group if target exists
        if self.config.target_column and self.config.target_column in data.columns:
            for attr in self.config.protected_attributes:
                if attr not in data.columns:
                    continue

                label_rates = data.groupby(attr)[self.config.target_column].apply(
                    lambda x: (x == self.config.positive_label).mean()
                ).to_dict()

                if label_rates:
                    max_rate = max(label_rates.values())
                    min_rate = min(label_rates.values())
                    disparity = max_rate - min_rate

                    results["key_metrics"][f"{attr}_label_disparity"] = disparity

                    if disparity > self.config.thresholds.label_rate_disparity_critical:
                        results["has_critical_bias"] = True

        return results

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

        if len(data) == 0:
            raise ValueError("DataFrame is empty")

        # Check protected attributes exist
        missing_attrs = [
            attr for attr in self.config.protected_attributes
            if attr not in data.columns
        ]
        if missing_attrs:
            raise ValueError(f"Protected attributes not found in data: {missing_attrs}")

        # Check target column exists if specified
        if self.config.target_column and self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")

    def _create_profile(self, data: pd.DataFrame) -> DatasetProfile:
        """Create dataset profile with basic statistics."""
        # Get column types
        column_types = {col: str(dtype) for col, dtype in data.dtypes.items()}

        # Get protected attribute distributions
        protected_dists = {}
        for attr in self.config.protected_attributes:
            if attr in data.columns:
                protected_dists[attr] = data[attr].value_counts(normalize=True).to_dict()

        # Get target distribution
        target_dist = None
        if self.config.target_column and self.config.target_column in data.columns:
            target_dist = data[self.config.target_column].value_counts(normalize=True).to_dict()

        # Get missing rates
        missing_rates = (data.isnull().sum() / len(data)).to_dict()

        return DatasetProfile(
            n_rows=len(data),
            n_columns=len(data.columns),
            column_types=column_types,
            protected_attribute_distributions=protected_dists,
            target_distribution=target_dist,
            missing_rates=missing_rates,
        )

    def _compute_group_stats(self, data: pd.DataFrame) -> dict[str, dict]:
        """
        Compute per-group statistics for each protected attribute.

        For each attribute: group counts, dataset share, positive-outcome rate
        (when a target is configured), and a chi-square p-value testing
        independence between the attribute and the target.
        """
        stats: dict[str, dict] = {}
        has_target = (
            self.config.target_column is not None
            and self.config.target_column in data.columns
        )

        for attr in self.config.protected_attributes:
            if attr not in data.columns:
                continue

            counts = data[attr].value_counts(dropna=True)
            groups = []
            for group, count in counts.items():
                entry = {
                    "group": str(group),
                    "count": int(count),
                    "share": float(count / len(data)),
                }
                if has_target:
                    mask = data[attr] == group
                    entry["positive_rate"] = float(
                        (data.loc[mask, self.config.target_column]
                         == self.config.positive_label).mean()
                    )
                groups.append(entry)

            p_value = None
            if has_target and len(counts) >= 2:
                try:
                    from scipy.stats import chi2_contingency

                    contingency = pd.crosstab(
                        data[attr], data[self.config.target_column]
                    )
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        _, p_value, _, _ = chi2_contingency(contingency)
                        p_value = float(p_value)
                except (ImportError, ValueError):
                    p_value = None

            stats[attr] = {"groups": groups, "chi2_p_value": p_value}

        return stats

    def _add_llm_explanations(self, report: AuditReport) -> None:
        """Add LLM explanations to findings."""
        for finding in report.findings:
            if finding.severity in [BiasSeverity.WARNING, BiasSeverity.CRITICAL]:
                explanation = self.llm_explainer.explain_finding(finding)
                finding.llm_explanation = explanation

    def explain(self, report: AuditReport, question: str) -> str:
        """
        Ask the LLM to explain something about the audit.

        Parameters
        ----------
        report : AuditReport
            The audit report to explain.
        question : str
            The question to answer.

        Returns
        -------
        str
            LLM-generated explanation.
        """
        if self.llm_explainer is None:
            return "LLM not configured. Initialize BiasAuditor with llm_provider to enable explanations."

        return self.llm_explainer.answer_question(report, question)

    def generate_remediation_code(
        self,
        report: AuditReport,
        finding_index: Optional[int] = None,
    ) -> str:
        """
        Generate code to remediate detected biases.

        Parameters
        ----------
        report : AuditReport
            The audit report.
        finding_index : int, optional
            Generate code for specific finding. If None, generates for all critical findings.

        Returns
        -------
        str
            Python code to implement remediations.
        """
        if self.llm_explainer is None:
            # Return template-based code if no LLM
            return self._generate_template_code(report, finding_index)

        return self.llm_explainer.generate_code(report, finding_index)

    def _generate_template_code(
        self,
        report: AuditReport,
        finding_index: Optional[int] = None,
    ) -> str:
        """Generate template-based remediation code without LLM."""
        lines = [
            "# Auto-generated remediation code",
            "# Review and adapt to your specific needs",
            "",
            "import pandas as pd",
            "from sklearn.utils import resample",
            "",
        ]

        findings = [report.findings[finding_index]] if finding_index else report.critical_findings

        for finding in findings:
            lines.append(f"# Fix for: {finding.title}")

            if finding.category == BiasCategory.REPRESENTATION:
                lines.append(f"# Resample to balance {finding.affected_attribute}")
                lines.append("# Option 1: Oversample minority groups")
                lines.append("# Option 2: Undersample majority groups")
                lines.append("# Option 3: SMOTE for synthetic samples")
                lines.append("")
            elif finding.category == BiasCategory.LABEL:
                lines.append(f"# Review labeling process for {finding.affected_attribute}")
                lines.append("# Consider: blind labeling, multiple annotators, label audits")
                lines.append("")
            elif finding.category == BiasCategory.FEATURE_PROXY:
                lines.append("# Consider removing or transforming proxy feature")
                lines.append("# Alternatively: use fairness-aware preprocessing")
                lines.append("")

        return "\n".join(lines)

    def generate_visualizations(
        self,
        data: pd.DataFrame,
        report: Optional[AuditReport] = None,
    ) -> dict[str, str]:
        """
        Generate visualizations for the audit.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        report : AuditReport, optional
            The audit report (for category scores).

        Returns
        -------
        dict
            Dictionary mapping visualization names to HTML/image content.
        """
        try:
            from bias_auditor.visualizations import generate_all_visualizations
        except ImportError:
            return {"error": "Visualization dependencies not installed. pip install matplotlib plotly"}

        category_scores = report.category_scores if report else None

        return generate_all_visualizations(
            data=data,
            protected_attrs=self.config.protected_attributes,
            target_column=self.config.target_column,
            positive_label=self.config.positive_label,
            category_scores=category_scores,
        )

    def __repr__(self) -> str:
        return (f"BiasAuditor(protected_attributes={self.config.protected_attributes}, "
                f"target_column={self.config.target_column}, "
                f"llm_provider={self.config.llm_config.provider.value})")
