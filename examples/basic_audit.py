"""
Example: Basic bias audit without LLM.

This example demonstrates how to run a statistical bias audit
on a dataset using only the built-in detectors.
"""

import pandas as pd
import numpy as np

# Create a sample biased dataset
np.random.seed(42)

n_samples = 1000

# Generate data with intentional biases
data = {
    # Protected attributes
    'gender': np.random.choice(['male', 'female'], n_samples, p=[0.7, 0.3]),  # Imbalanced
    'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_samples, 
                              p=[0.6, 0.15, 0.15, 0.1]),  # Imbalanced
    'age': np.random.randint(18, 65, n_samples),
    
    # Features
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'years_employed': np.random.randint(0, 30, n_samples),
    'zip_code': np.random.choice(['10001', '10002', '10003', '10004', '10005'], n_samples),
}

df = pd.DataFrame(data)

# Add biased target (approval rate differs by gender)
def generate_approval(row):
    base_prob = 0.5
    
    # Gender bias
    if row['gender'] == 'male':
        base_prob += 0.15
    
    # Race bias
    if row['race'] == 'white':
        base_prob += 0.1
    
    # Legitimate factors
    if row['credit_score'] > 700:
        base_prob += 0.1
    if row['years_employed'] > 5:
        base_prob += 0.05
    
    return 1 if np.random.random() < min(base_prob, 0.95) else 0

df['approved'] = df.apply(generate_approval, axis=1)

# Make zip_code correlate with race (proxy variable)
df.loc[df['race'] == 'white', 'zip_code'] = np.random.choice(
    ['10001', '10002'], (df['race'] == 'white').sum(), p=[0.7, 0.3]
)
df.loc[df['race'] == 'black', 'zip_code'] = np.random.choice(
    ['10003', '10004', '10005'], (df['race'] == 'black').sum()
)

# Add some missing data (more missing for minority groups)
missing_mask = (df['race'].isin(['black', 'hispanic'])) & (np.random.random(n_samples) < 0.2)
df.loc[missing_mask, 'income'] = np.nan

print("Sample dataset created with intentional biases:")
print(f"  Shape: {df.shape}")
print(f"  Gender distribution: {df['gender'].value_counts(normalize=True).to_dict()}")
print(f"  Approval by gender: {df.groupby('gender')['approved'].mean().to_dict()}")
print()

# --- Run Bias Audit ---

from bias_auditor import BiasAuditor

# Create auditor
auditor = BiasAuditor(
    protected_attributes=['gender', 'race'],
    target_column='approved',
    positive_label=1,
    verbose=True,
)

# Run full audit
print("Running bias audit...")
print("=" * 60)
report = auditor.audit(df, dataset_name="sample_biased_data")

# Display results
print()
print(report.summary())
print()
print("=" * 60)
print("REMEDIATION PLAN")
print("=" * 60)
print(report.remediation_plan())

# Save HTML report
report.to_html("basic_audit_report.html")
print("\nHTML report saved to: basic_audit_report.html")

# Quick check alternative
print("\n" + "=" * 60)
print("QUICK CHECK (for CI/CD)")
print("=" * 60)
quick_results = auditor.quick_check(df)
print(f"Has critical bias: {quick_results['has_critical_bias']}")
for metric, value in quick_results['key_metrics'].items():
    print(f"  {metric}: {value:.4f}")
