"""
Example: Pipeline integration for CI/CD.

This example demonstrates how to integrate bias auditing
into a data pipeline or CI/CD workflow.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Simulated data loading from various sources
def load_data(source: str) -> pd.DataFrame:
    """Load data from various sources."""
    # In real usage, this would load from:
    # - S3, GCS, Azure Blob
    # - Database queries
    # - API endpoints
    # - Local files
    
    # For demo, create sample data
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'user_id': range(n),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n),
        'gender': np.random.choice(['M', 'F', 'NB'], n, p=[0.45, 0.45, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], n),
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'target': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })


def run_bias_audit(
    df: pd.DataFrame,
    protected_attrs: list[str],
    target_col: str,
    fail_on_critical: bool = True,
    output_dir: str = ".",
) -> dict:
    """
    Run bias audit as part of a pipeline.
    
    Returns:
        dict with audit results and exit status
    """
    from bias_auditor import BiasAuditor, BiasThresholds
    
    # Configure thresholds (can be loaded from config file)
    thresholds = BiasThresholds(
        disparate_impact_critical=0.8,
        statistical_parity_critical=0.15,
        imbalance_ratio_critical=4.0,
    )
    
    # Create auditor
    auditor = BiasAuditor(
        protected_attributes=protected_attrs,
        target_column=target_col,
        thresholds=thresholds,
        verbose=False,  # Quiet for CI/CD
    )
    
    # Run audit
    report = auditor.audit(df, dataset_name="pipeline_data")
    
    # Prepare results
    results = {
        "status": "pass" if not report.has_critical_bias else "fail",
        "overall_score": report.overall_bias_score,
        "critical_count": len(report.critical_findings),
        "warning_count": len(report.warning_findings),
        "findings": [
            {
                "title": f.title,
                "severity": f.severity.value,
                "category": f.category.value,
                "affected_attribute": f.affected_attribute,
            }
            for f in report.findings
        ],
    }
    
    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON results (machine-readable)
    with open(output_path / "bias_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # HTML report (human-readable)
    report.to_html(str(output_path / "bias_audit_report.html"))
    
    # Summary for logs
    print(f"Bias Audit Complete")
    print(f"  Status: {results['status'].upper()}")
    print(f"  Score: {results['overall_score']:.2%}")
    print(f"  Critical: {results['critical_count']}, Warnings: {results['warning_count']}")
    
    return results


def main():
    """Main pipeline entry point."""
    
    # Configuration (could come from environment variables, config files, etc.)
    config = {
        "data_source": "s3://bucket/training_data.parquet",
        "protected_attributes": ["gender", "age_group"],
        "target_column": "target",
        "fail_on_critical": True,
        "output_dir": "./audit_artifacts",
    }
    
    print("=" * 60)
    print("DATA PIPELINE - BIAS AUDIT STAGE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/3] Loading data...")
    df = load_data(config["data_source"])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Run bias audit
    print("\n[2/3] Running bias audit...")
    results = run_bias_audit(
        df=df,
        protected_attrs=config["protected_attributes"],
        target_col=config["target_column"],
        fail_on_critical=config["fail_on_critical"],
        output_dir=config["output_dir"],
    )
    
    # Step 3: Decide pipeline action
    print("\n[3/3] Pipeline decision...")
    if results["status"] == "fail" and config["fail_on_critical"]:
        print("  BLOCKING: Critical bias detected. Fix before training.")
        print("  See: audit_artifacts/bias_audit_report.html")
        sys.exit(1)
    elif results["warning_count"] > 0:
        print("  WARNING: Bias warnings detected. Review recommended.")
        print("  Continuing pipeline with warnings...")
        sys.exit(0)
    else:
        print("  PASS: No significant bias detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
