# Dataset Bias Auditor

**Detect representation biases, label biases, and fairness issues in datasets *before* training — with actionable remediation suggestions.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Why This Tool?

Most fairness tools focus on **models**, not **raw data**. But biases baked into training data propagate to models. This tool catches issues early — before you train a single model.

### Key Features

- 🔍 **Statistical Bias Detection**: Representation, label, proxy, and missing data bias
- 🪄 **Auto-Detection**: Point it at any tabular dataset — protected attributes, target
  column, and positive label are detected from column names; continuous age is auto-binned
- 🤖 **LLM-Powered Analysis**: Text column bias detection, natural language explanations
- 📊 **Rich Reports**: HTML dashboards with embedded interactive charts, per-group
  breakdown tables with chi-square significance tests, JSON exports, CLI summaries
- 🛠️ **Actionable Remediation**: Code generation for resampling, reweighting, preprocessing
- ⚡ **Fast Integration**: Works with pandas DataFrames, CLI, and CI/CD pipelines

## Installation

```bash
# From GitHub
pip install git+https://github.com/kavyagajjar/dataset_bias_detector.git

# From a local checkout
pip install -e .

# With LLM support
pip install -e ".[llm]"

# With MLOps integrations
pip install -e ".[mlops]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from bias_auditor import BiasAuditor
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Create auditor
auditor = BiasAuditor(
    protected_attributes=['gender', 'race', 'age_group'],
    target_column='approved',
    positive_label=1
)

# Run audit
report = auditor.audit(df)

# View results
print(report.summary())
report.to_html("bias_report.html")
```

### With LLM Integration

```python
auditor = BiasAuditor(
    protected_attributes=['gender', 'race'],
    target_column='approved',
    llm_provider='openai',  # or 'anthropic', 'local'
    llm_api_key='your-api-key'
)

report = auditor.audit(df)

# Get intelligent explanations
explanation = auditor.explain(report, "Why is zip_code flagged as a proxy?")

# Generate remediation code
code = auditor.generate_remediation_code(report)
```

### Command Line

```bash
# Zero-config: auto-detect protected attributes, target, and positive label
bias-auditor audit data.csv --auto -o report.html

# Basic audit with explicit columns
bias-auditor audit data.csv -p gender -p race -t approved

# Full report with HTML output
bias-auditor audit data.csv -p gender -t approved -o report.html -f full

# Quick check (CI/CD friendly)
bias-auditor quick-check data.csv -p gender -p race
```

### Auto-Detection

Don't know (or don't want to type) the sensitive columns? Pass `--auto` on the
CLI or `auto_detect=True` in Python:

```python
auditor = BiasAuditor(auto_detect=True)
report = auditor.audit(df)  # finds gender/race/age/..., bins continuous age,
                            # guesses the binary target and positive label
```

Every detection decision is printed and recorded in the report's configuration
appendix so you can verify it. Supported inputs: CSV, Parquet, JSON (CLI), or
any pandas DataFrame (Python API).

## What It Detects

| Category | What It Finds |
|----------|---------------|
| **Representation** | Underrepresented groups, severe imbalance, distribution skew |
| **Label Bias** | Disparate impact, statistical parity violations, label rate disparities |
| **Feature Proxies** | Features that encode protected attributes (e.g., zip code → race) |
| **Missing Data** | Differential missingness, MNAR patterns |
| **Text Bias** | Stereotypes, sentiment disparities, exclusionary language (LLM) |
| **Intersectional** | Compound biases across multiple attributes |

## What's in the HTML Report

- **Score dashboard** — overall bias score, counts by severity, per-category bars
- **Dataset overview** — rows/columns, target distribution, missing-data table
- **Group breakdown** — per-group counts, shares, and positive-outcome rates for every
  protected attribute, each with a chi-square test of independence (p-value)
- **Findings** — critical/warning/info cards with metrics and remediation actions
- **Interactive charts** — group distributions, label rates with the 80%-rule threshold
  line, a category radar, and intersectional heatmaps (attr × attr positive rates)
- **Configuration appendix** — thresholds and columns used (plus auto-detection notes),
  so every report is reproducible

## Report Example

```
============================================================
DATASET BIAS AUDIT REPORT
============================================================
Audit ID: a1b2c3d4
Overall Bias Score: 0.72 (HIGH)

FINDINGS SUMMARY
----------------------------------------
  Critical: 3
  Warning:  5
  Info:     2

CRITICAL ISSUES
----------------------------------------
  [LABEL] Disparate impact detected for 'gender'
    The positive outcome rate for 'female' is only 62% of male rate.
    Fix: Review labeling criteria for potential bias

CATEGORY SCORES
----------------------------------------
  representation       [████████░░] 0.80
  label               [███████░░░] 0.72
  feature_proxy       [███░░░░░░░] 0.30
============================================================
```

## Remediation Support

The auditor doesn't just find problems — it helps fix them:

```python
# Get remediation strategies
from bias_auditor.remediation import get_remediation_strategies

strategies = get_remediation_strategies(report.critical_findings[0])
for s in strategies:
    print(f"{s.name}: {s.description}")

# Apply resampling
from bias_auditor.remediation import ResamplingRemediation

resampler = ResamplingRemediation()
balanced_df = resampler.oversample(df, 'gender', strategy='smote')

# Apply reweighting
from bias_auditor.remediation import ReweightingRemediation

reweighter = ReweightingRemediation()
sample_weights = reweighter.inverse_frequency_weights(df, 'race')
```

## CI/CD Integration

The CLI returns exit code 1 if critical bias is detected:

```yaml
# GitHub Actions example
- name: Bias Audit
  run: |
    pip install git+https://github.com/kavyagajjar/dataset_bias_detector.git
    bias-auditor audit data/training.csv -p gender -p race -t label -o bias_report.html

- name: Upload Report
  uses: actions/upload-artifact@v4
  with:
    name: bias-report
    path: bias_report.html
```

## Configuration

```python
from bias_auditor import BiasAuditor, BiasThresholds, AuditConfig

# Custom thresholds
thresholds = BiasThresholds(
    disparate_impact_critical=0.8,  # 80% rule
    statistical_parity_critical=0.2,
    imbalance_ratio_critical=5.0,
)

# Full configuration
config = AuditConfig(
    protected_attributes=['gender', 'race', 'age'],
    target_column='outcome',
    positive_label='approved',
    thresholds=thresholds,
    text_columns=['job_description'],  # For LLM analysis
    compute_intersectional=True,
    min_group_size=30,
)

auditor = BiasAuditor(config=config)
```

## Supported LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | `OPENAI_API_KEY` |
| Anthropic | claude-opus-4-8, claude-sonnet-5, claude-haiku-4-5 | `ANTHROPIC_API_KEY` |
| Azure OpenAI | Any deployed model | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Local (Ollama) | llama3, mistral, etc. | Running Ollama server |

## API Reference

### BiasAuditor

```python
class BiasAuditor:
    def __init__(
        self,
        protected_attributes: list[str],
        target_column: str = None,
        positive_label: Any = 1,
        thresholds: BiasThresholds = None,
        llm_provider: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        auto_detect: bool = False,
        verbose: bool = True,
    ): ...
    
    def audit(self, data: pd.DataFrame, dataset_name: str = None) -> AuditReport: ...
    def quick_check(self, data: pd.DataFrame) -> dict: ...
    def explain(self, report: AuditReport, question: str) -> str: ...
    def generate_remediation_code(self, report: AuditReport) -> str: ...
```

### AuditReport

```python
class AuditReport:
    audit_id: str
    overall_bias_score: float  # 0-1, higher = more bias
    findings: list[BiasFindings]
    critical_findings: list[BiasFindings]
    warning_findings: list[BiasFindings]
    
    def summary(self) -> str: ...
    def to_html(self, path: str) -> str: ...
    def to_json(self) -> str: ...
    def remediation_plan(self) -> str: ...
```

## Claude Skill

This repo ships a [Claude skill](skills/dataset-bias-audit/SKILL.md) that
teaches Claude (Claude Code, claude.ai, or the desktop app) to run bias audits
with this tool — point Claude at any tabular dataset and ask "check this for
bias". To install it:

- **Claude Code**: copy `skills/dataset-bias-audit/` into `~/.claude/skills/`
  (or your project's `.claude/skills/`).
- **claude.ai / desktop**: package it as a `.skill` file (zip the
  `dataset-bias-audit` folder) and upload it under Settings → Capabilities.

The skill installs this package automatically (from a bundled wheel or from
this repo) and follows a verified workflow: inspect the data, run the audit
with auto-detection, verify the detection decisions, and report findings with
statistical significance and honest caveats.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
