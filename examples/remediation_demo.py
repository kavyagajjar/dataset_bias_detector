"""
Example: Using remediation tools.

This example demonstrates how to use the resampling and
reweighting tools to address detected biases.
"""

import pandas as pd
import numpy as np

# Create a biased dataset
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'gender': np.random.choice(['male', 'female'], n_samples, p=[0.75, 0.25]),
    'race': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'approved': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
})

# Add label bias
df.loc[(df['gender'] == 'female') & (df['approved'] == 1), 'approved'] = np.where(
    np.random.random((df['gender'] == 'female').sum()) < 0.3, 0, 1
)[:((df['gender'] == 'female') & (df['approved'] == 1)).sum()]

print("Original Dataset:")
print(f"  Shape: {df.shape}")
print(f"  Gender distribution: {df['gender'].value_counts(normalize=True).to_dict()}")
print(f"  Race distribution: {df['race'].value_counts(normalize=True).to_dict()}")
print(f"  Approval by gender: {df.groupby('gender')['approved'].mean().to_dict()}")
print()

# --- Resampling Remediation ---

from bias_auditor.remediation import ResamplingRemediation

resampler = ResamplingRemediation(random_state=42)

# 1. Random Oversampling
print("=" * 60)
print("RESAMPLING REMEDIATION")
print("=" * 60)

print("\n[1] Random Oversampling:")
df_oversampled = resampler.oversample(df, 'gender', target_ratio=1.0)
print(f"  Original size: {len(df)}")
print(f"  New size: {len(df_oversampled)}")
print(f"  New gender distribution: {df_oversampled['gender'].value_counts(normalize=True).to_dict()}")

# 2. Random Undersampling
print("\n[2] Random Undersampling:")
df_undersampled = resampler.undersample(df, 'gender')
print(f"  Original size: {len(df)}")
print(f"  New size: {len(df_undersampled)}")
print(f"  New gender distribution: {df_undersampled['gender'].value_counts(normalize=True).to_dict()}")

# 3. Hybrid Approach
print("\n[3] Hybrid Resampling (over + under):")
df_hybrid = resampler.hybrid_resample(df, 'race', target_size=200)
print(f"  Original size: {len(df)}")
print(f"  New size: {len(df_hybrid)}")
print(f"  New race distribution: {df_hybrid['race'].value_counts(normalize=True).to_dict()}")

# 4. Stratified Split
print("\n[4] Stratified Train/Test Split:")
train_df, test_df = resampler.stratified_split(df, 'gender', test_size=0.2, target_column='approved')
print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
print(f"  Train gender dist: {train_df['gender'].value_counts(normalize=True).to_dict()}")
print(f"  Test gender dist: {test_df['gender'].value_counts(normalize=True).to_dict()}")

# --- Reweighting Remediation ---

from bias_auditor.remediation import ReweightingRemediation

reweighter = ReweightingRemediation()

print("\n" + "=" * 60)
print("REWEIGHTING REMEDIATION")
print("=" * 60)

# 1. Inverse Frequency Weights
print("\n[1] Inverse Frequency Weights:")
weights_if = reweighter.inverse_frequency_weights(df, 'gender')
print(f"  Male weight (mean): {weights_if[df['gender'] == 'male'].mean():.4f}")
print(f"  Female weight (mean): {weights_if[df['gender'] == 'female'].mean():.4f}")

# 2. Balanced Weights
print("\n[2] Balanced Weights:")
weights_bal = reweighter.balanced_weights(df, 'gender')
print(f"  Male total weight: {weights_bal[df['gender'] == 'male'].sum():.2f}")
print(f"  Female total weight: {weights_bal[df['gender'] == 'female'].sum():.2f}")

# 3. Label Balancing Weights
print("\n[3] Label Balancing Weights:")
weights_label = reweighter.label_balancing_weights(df, 'gender', 'approved', positive_label=1)
print(f"  Original approval rate by gender: {df.groupby('gender')['approved'].mean().to_dict()}")

# Calculate weighted approval rate
df_temp = df.copy()
df_temp['weight'] = weights_label
weighted_rates = df_temp.groupby('gender').apply(
    lambda x: (x['approved'] * x['weight']).sum() / x['weight'].sum()
)
print(f"  Weighted approval rate: {weighted_rates.to_dict()}")

# 4. Intersectional Weights
print("\n[4] Intersectional Weights (gender × race):")
weights_int = reweighter.intersectional_weights(df, ['gender', 'race'])
print(f"  Weight range: [{weights_int.min():.4f}, {weights_int.max():.4f}]")
print(f"  Unique weights: {len(weights_int.unique())}")

# --- Using Weights in Model Training ---

print("\n" + "=" * 60)
print("USING WEIGHTS IN MODEL TRAINING")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare data
X = df[['feature_1', 'feature_2']].values
y = df['approved'].values
sample_weights = weights_if.values

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# Train without weights
model_unweighted = LogisticRegression()
model_unweighted.fit(X_train, y_train)
pred_unweighted = model_unweighted.predict(X_test)

# Train with weights
model_weighted = LogisticRegression()
model_weighted.fit(X_train, y_train, sample_weight=w_train)
pred_weighted = model_weighted.predict(X_test)

print(f"\nUnweighted model accuracy: {accuracy_score(y_test, pred_unweighted):.4f}")
print(f"Weighted model accuracy: {accuracy_score(y_test, pred_weighted):.4f}")

# Check fairness of predictions
test_df = pd.DataFrame({'gender': df.iloc[y_test.tolist()]['gender'].values})
test_df['actual'] = y_test
test_df['pred_unweighted'] = pred_unweighted
test_df['pred_weighted'] = pred_weighted

print(f"\nUnweighted prediction rate by gender:")
print(f"  {test_df.groupby('gender')['pred_unweighted'].mean().to_dict()}")
print(f"Weighted prediction rate by gender:")
print(f"  {test_df.groupby('gender')['pred_weighted'].mean().to_dict()}")
