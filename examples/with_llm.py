"""
Example: Bias audit with LLM integration.

This example demonstrates how to use LLM capabilities for
enhanced analysis, explanations, and code generation.

Requires: OpenAI or Anthropic API key
"""

import os
import pandas as pd
import numpy as np

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Note: Set OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM features")
    print("Running in demo mode with mock responses...")
    api_key = None

# Create sample dataset with text columns
np.random.seed(42)
n_samples = 200

# Job descriptions with potential gender bias
male_descriptions = [
    "Looking for a rockstar developer to crush it in our fast-paced environment",
    "Need an aggressive salesperson who can dominate the market",
    "Seeking a competitive individual to lead our engineering team",
    "We want a ninja coder who can work independently",
]

female_descriptions = [
    "Seeking a supportive team member to help with administrative tasks",
    "Looking for someone nurturing to manage our customer relationships",
    "Need a detail-oriented person for organizational duties",
    "Seeking a collaborative individual to assist the team",
]

neutral_descriptions = [
    "Looking for a skilled professional to join our team",
    "Seeking an experienced candidate for this role",
    "Need a qualified individual to contribute to our projects",
]

data = {
    'applicant_id': range(n_samples),
    'gender': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
    'years_experience': np.random.randint(0, 15, n_samples),
    'education': np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], n_samples),
}

# Assign job descriptions with bias
job_descriptions = []
for gender in data['gender']:
    if gender == 'male':
        if np.random.random() < 0.6:
            job_descriptions.append(np.random.choice(male_descriptions))
        else:
            job_descriptions.append(np.random.choice(neutral_descriptions))
    else:
        if np.random.random() < 0.6:
            job_descriptions.append(np.random.choice(female_descriptions))
        else:
            job_descriptions.append(np.random.choice(neutral_descriptions))

data['job_description'] = job_descriptions

# Biased hiring decisions
def generate_hired(row):
    base_prob = 0.3 + row['years_experience'] * 0.03
    if row['gender'] == 'male':
        base_prob += 0.2
    if row['education'] in ['masters', 'phd']:
        base_prob += 0.1
    return 1 if np.random.random() < min(base_prob, 0.9) else 0

df = pd.DataFrame(data)
df['hired'] = df.apply(generate_hired, axis=1)

print("Sample hiring dataset created:")
print(f"  Shape: {df.shape}")
print(f"  Gender distribution: {df['gender'].value_counts(normalize=True).to_dict()}")
print(f"  Hiring rate by gender: {df.groupby('gender')['hired'].mean().to_dict()}")
print()

# --- Run Bias Audit with LLM ---

from bias_auditor import BiasAuditor
from bias_auditor.core.config import AuditConfig, LLMConfig, LLMProvider

# Configure LLM
if api_key and "sk-" in api_key:
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        api_key=api_key,
        enable_text_analysis=True,
        enable_explanations=True,
        enable_code_generation=True,
    )
elif api_key:
    llm_config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        enable_text_analysis=True,
        enable_explanations=True,
    )
else:
    llm_config = LLMConfig(provider=LLMProvider.NONE)

# Create config with text columns
config = AuditConfig(
    protected_attributes=['gender'],
    target_column='hired',
    positive_label=1,
    text_columns=['job_description'],  # Enable text analysis
    llm_config=llm_config,
    verbose=True,
)

# Create auditor
auditor = BiasAuditor(config=config)

# Run audit
print("Running bias audit with LLM integration...")
print("=" * 60)
report = auditor.audit(df, dataset_name="hiring_data")

# Display summary
print()
print(report.summary())

# LLM Features (if available)
if llm_config.provider != LLMProvider.NONE:
    print("\n" + "=" * 60)
    print("LLM-ENHANCED FEATURES")
    print("=" * 60)
    
    # Executive summary
    if report.executive_summary:
        print("\n[Executive Summary]")
        print(report.executive_summary)
    
    # Explain a finding
    if report.critical_findings:
        print("\n[Detailed Explanation]")
        explanation = auditor.explain(
            report, 
            "Why is there a hiring disparity between genders?"
        )
        print(explanation)
    
    # Generate remediation code
    print("\n[Generated Remediation Code]")
    code = auditor.generate_remediation_code(report)
    print(code[:1000] + "..." if len(code) > 1000 else code)

# Show findings with LLM explanations
print("\n" + "=" * 60)
print("FINDINGS WITH EXPLANATIONS")
print("=" * 60)
for finding in report.findings[:3]:
    print(f"\n[{finding.severity.value.upper()}] {finding.title}")
    print(f"  {finding.description}")
    if finding.llm_explanation:
        print(f"\n  LLM Explanation:")
        print(f"  {finding.llm_explanation[:300]}...")

# Save report
report.to_html("llm_audit_report.html")
print("\nHTML report saved to: llm_audit_report.html")
