"""Prompt templates for LLM-based bias analysis."""

# System prompts
BIAS_ANALYST_SYSTEM = """You are an expert AI fairness analyst specializing in detecting and explaining biases in datasets. Your role is to:

1. Analyze data patterns that indicate potential discrimination or unfairness
2. Explain findings in clear, accessible language for both technical and non-technical audiences
3. Provide actionable, evidence-based remediation suggestions
4. Consider intersectionality and systemic factors when analyzing bias
5. Be precise with statistical terminology while remaining understandable

Always ground your analysis in the specific metrics and evidence provided. Avoid speculation without data support."""

TEXT_ANALYST_SYSTEM = """You are an expert at detecting bias, stereotypes, and unfair language in text data. Your task is to:

1. Identify language that may perpetuate stereotypes or discrimination
2. Detect sentiment disparities across demographic groups
3. Flag exclusionary or non-inclusive language patterns
4. Consider context and domain when making assessments
5. Distinguish between intentional bias and unintentional language patterns

Be specific about what you find and why it may be problematic. Provide concrete examples from the text."""

CODE_GENERATOR_SYSTEM = """You are an expert Python programmer specializing in data preprocessing and fairness in machine learning. Generate clean, well-documented code that:

1. Uses pandas, scikit-learn, and standard fairness libraries
2. Includes comments explaining each step
3. Handles edge cases and errors gracefully
4. Follows best practices for reproducibility
5. Is ready to integrate into existing ML pipelines

Only generate code that directly addresses the bias issue described."""

# Analysis prompts
EXPLAIN_FINDING_TEMPLATE = """Analyze this bias finding and provide a clear explanation:

**Finding:** {title}
**Category:** {category}
**Severity:** {severity}
**Description:** {description}

**Metrics:**
{metrics}

**Affected Groups:**
{affected_groups}

Please provide:
1. A plain-language explanation of what this bias means
2. Why this is problematic for ML models and affected groups
3. Potential root causes of this bias
4. The real-world impact if not addressed

Keep your explanation concise but thorough (3-4 paragraphs)."""

EXECUTIVE_SUMMARY_TEMPLATE = """Generate an executive summary of this dataset bias audit for stakeholders:

**Dataset:** {dataset_name}
**Total Findings:** {total_findings}
**Critical Issues:** {critical_count}
**Warning Issues:** {warning_count}

**Category Scores (0-1, higher = more bias):**
{category_scores}

**Top Critical Findings:**
{critical_findings}

Write a 2-3 paragraph executive summary that:
1. Summarizes the overall bias risk level
2. Highlights the most important findings
3. Recommends immediate actions
4. Uses non-technical language suitable for business stakeholders"""

REMEDIATION_CODE_TEMPLATE = """Generate Python code to remediate this bias issue:

**Finding:** {title}
**Category:** {category}
**Description:** {description}
**Affected Attribute:** {affected_attribute}
**Metrics:** {metrics}

Generate production-ready Python code that:
1. Addresses this specific bias issue
2. Uses pandas and scikit-learn
3. Includes clear comments
4. Can be integrated into a data preprocessing pipeline
5. Includes before/after validation

Only output the Python code, no explanations."""

TEXT_ANALYSIS_TEMPLATE = """Analyze these text samples for bias related to {protected_attribute}:

**Context:** These are samples from the '{column_name}' column in a dataset used for {use_case}.

**Samples by group:**
{samples_by_group}

Analyze for:
1. Stereotyping language associated with specific groups
2. Sentiment differences across groups
3. Exclusionary or non-inclusive terminology
4. Gendered or culturally biased language
5. Professional vs. casual tone differences by group

For each issue found, provide:
- Specific example(s) from the samples
- Why it may be problematic
- Suggested alternative language

Be thorough but focus on patterns, not isolated cases."""

PROXY_EXPLANATION_TEMPLATE = """Explain why this feature may be a proxy for a protected attribute:

**Feature:** {feature_name}
**Protected Attribute:** {protected_attribute}
**Correlation/Association:** {correlation}
**Mutual Information:** {mutual_information}

Consider:
1. Historical and systemic reasons this correlation might exist
2. Whether this is a direct or indirect proxy
3. Legal and ethical implications of using this feature
4. Whether the feature has legitimate predictive value independent of the proxy effect

Provide a balanced analysis in 2-3 paragraphs."""

QA_TEMPLATE = """Based on this bias audit report, answer the user's question:

**Audit Summary:**
- Dataset: {dataset_name}
- Overall Bias Score: {bias_score}
- Critical Findings: {critical_count}
- Warning Findings: {warning_count}

**Findings:**
{findings_summary}

**User Question:** {question}

Provide a direct, helpful answer based on the audit findings. If the information isn't available in the report, say so."""
