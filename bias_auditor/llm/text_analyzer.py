"""Text bias analyzer using LLM."""

from typing import Optional

import pandas as pd

from bias_auditor.core.config import LLMConfig
from bias_auditor.core.report import BiasCategory, BiasFindings, BiasSeverity
from bias_auditor.llm.base import get_llm_provider
from bias_auditor.llm.prompts import TEXT_ANALYSIS_TEMPLATE, TEXT_ANALYST_SYSTEM


class TextBiasAnalyzer:
    """
    Analyzer for bias in text columns using LLM.

    Detects:
    - Stereotyping language
    - Sentiment disparities across groups
    - Exclusionary terminology
    - Gendered or culturally biased language
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = get_llm_provider(config)

        if self.provider is None:
            raise ValueError("LLM provider is required for text analysis")

    def analyze(
        self,
        data: pd.DataFrame,
        text_columns: list[str],
        protected_attributes: Optional[list[str]] = None,
        sample_size: int = 50,
    ) -> list[BiasFindings]:
        """
        Analyze text columns for bias.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        text_columns : list[str]
            Columns containing text to analyze.
        protected_attributes : list[str], optional
            Protected attributes to analyze against.
        sample_size : int
            Number of samples per group to analyze.

        Returns
        -------
        list[BiasFindings]
            List of detected bias findings.
        """
        findings = []

        if not self.provider.is_available():
            return findings

        # If no protected attributes specified, analyze text in isolation
        if not protected_attributes:
            for col in text_columns:
                if col in data.columns:
                    findings.extend(self._analyze_column_standalone(data, col, sample_size))
        else:
            # Analyze text by protected attribute groups
            for col in text_columns:
                if col not in data.columns:
                    continue
                for attr in protected_attributes:
                    if attr not in data.columns:
                        continue
                    findings.extend(self._analyze_column_by_group(data, col, attr, sample_size))

        return findings

    def _analyze_column_standalone(
        self,
        data: pd.DataFrame,
        column: str,
        sample_size: int,
    ) -> list[BiasFindings]:
        """Analyze a text column without grouping."""
        findings = []

        # Sample text
        text_data = data[column].dropna()
        if len(text_data) == 0:
            return findings

        samples = text_data.sample(min(sample_size, len(text_data))).tolist()

        prompt = f"""Analyze these text samples for potential bias:

**Column:** {column}

**Samples:**
{chr(10).join(f'- "{s[:500]}"' for s in samples[:20])}

Look for:
1. Stereotyping or discriminatory language
2. Gendered language that could exclude groups
3. Culturally insensitive terminology
4. Professional tone inconsistencies
5. Exclusionary patterns

If you find concerning patterns, describe them specifically with examples."""

        try:
            response = self.provider.complete(prompt, TEXT_ANALYST_SYSTEM)

            # Parse response for findings
            if any(
                word in response.content.lower()
                for word in [
                    "stereotype",
                    "bias",
                    "discriminat",
                    "exclusion",
                    "problematic",
                    "concern",
                ]
            ):
                findings.append(
                    BiasFindings(
                        category=BiasCategory.TEXT,
                        severity=BiasSeverity.WARNING,
                        title=f"Potential text bias in '{column}'",
                        description="LLM analysis identified potential bias patterns in text content.",
                        affected_attribute=column,
                        llm_explanation=response.content,
                        remediation_suggestions=[
                            "Review flagged text patterns manually",
                            "Consider bias-aware text preprocessing",
                            "Implement content guidelines for text fields",
                        ],
                    )
                )
        except Exception:
            # Silently skip on LLM errors
            pass

        return findings

    def _analyze_column_by_group(
        self,
        data: pd.DataFrame,
        column: str,
        protected_attr: str,
        sample_size: int,
    ) -> list[BiasFindings]:
        """Analyze text column grouped by protected attribute."""
        findings = []

        # Get samples by group
        samples_by_group = {}
        for group in data[protected_attr].dropna().unique():
            group_data = data[data[protected_attr] == group][column].dropna()
            if len(group_data) > 0:
                n_samples = min(sample_size // len(data[protected_attr].unique()), len(group_data))
                samples_by_group[str(group)] = group_data.sample(max(1, n_samples)).tolist()

        if len(samples_by_group) < 2:
            return findings

        # Format samples for prompt
        samples_text = ""
        for group, samples in samples_by_group.items():
            samples_text += f"\n**Group: {group}**\n"
            for s in samples[:10]:
                samples_text += f'- "{s[:300]}"\n'

        prompt = TEXT_ANALYSIS_TEMPLATE.format(
            protected_attribute=protected_attr,
            column_name=column,
            use_case="machine learning training",
            samples_by_group=samples_text,
        )

        try:
            response = self.provider.complete(prompt, TEXT_ANALYST_SYSTEM)

            # Check for significant findings
            content_lower = response.content.lower()
            severity = BiasSeverity.INFO

            if any(
                word in content_lower
                for word in ["significant", "clear pattern", "systematic", "discriminat"]
            ):
                severity = BiasSeverity.CRITICAL
            elif any(
                word in content_lower
                for word in ["pattern", "tendency", "difference", "stereotype", "bias"]
            ):
                severity = BiasSeverity.WARNING

            if severity != BiasSeverity.INFO or "no significant" not in content_lower:
                findings.append(
                    BiasFindings(
                        category=BiasCategory.TEXT,
                        severity=severity,
                        title=f"Text bias analysis: '{column}' by '{protected_attr}'",
                        description=(
                            f"LLM analysis of text in '{column}' across "
                            f"'{protected_attr}' groups identified potential patterns."
                        ),
                        affected_attribute=protected_attr,
                        affected_groups=list(samples_by_group.keys()),
                        llm_explanation=response.content,
                        remediation_suggestions=[
                            "Review identified text patterns with domain experts",
                            "Consider text standardization or augmentation",
                            "Implement fairness-aware text preprocessing",
                            "Add text bias monitoring to model evaluation",
                        ],
                        evidence={
                            "n_samples_per_group": {g: len(s) for g, s in samples_by_group.items()},
                            "column": column,
                            "protected_attribute": protected_attr,
                        },
                    )
                )
        except Exception:
            pass

        return findings

    def analyze_specific_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Analyze a specific text sample for bias.

        Parameters
        ----------
        text : str
            Text to analyze.
        context : str, optional
            Additional context about the text.

        Returns
        -------
        str
            Analysis of the text.
        """
        prompt = f"""Analyze this text for potential bias:

{f'Context: {context}' if context else ''}

Text: "{text}"

Identify any:
1. Stereotyping or discriminatory language
2. Gendered assumptions
3. Cultural insensitivity
4. Exclusionary patterns
5. Implicit biases

Be specific and provide suggestions for improvement if issues are found."""

        response = self.provider.complete(prompt, TEXT_ANALYST_SYSTEM)
        return response.content
