---
name: dataset-bias-audit
description: >
  Audit tabular datasets (CSV, Parquet, JSON, Excel, or pandas DataFrames) for
  bias and fairness issues before model training — representation imbalance,
  label/outcome disparities (disparate impact, 80% rule), proxy features,
  missing-data bias, and intersectional effects — producing an interactive HTML
  report plus a chat summary with statistical significance tests. Use this
  skill whenever the user wants to check a dataset for bias, fairness, or
  discrimination; asks whether their training data is balanced or fair; mentions
  protected attributes (gender, race, age, etc.) in the context of data quality;
  wants a bias report or fairness audit; or is preparing data for ML and asks
  about representation, disparate impact, or group imbalance — even if they
  don't use the word "bias" (e.g. "is my hiring data skewed against anyone?").
---

# Dataset Bias Audit

Audit a tabular dataset for bias using the `dataset-bias-auditor` Python
package (bundled with this skill as a wheel). The tool detects representation
bias, label/outcome bias, feature proxies, missing-data bias, and
intersectional effects, and generates an interactive HTML report with charts,
per-group tables, and chi-square significance tests.

## Step 1: Ensure the package is installed

Check for the package and install it if missing:

```bash
python -c "import bias_auditor" 2>/dev/null || pip install <source>
```

where `<source>` is, in order of preference:

1. The wheel bundled with this skill, if an `assets/*.whl` file exists next
   to this SKILL.md (packaged `.skill` distributions include it).
2. The GitHub repository:
   `git+https://github.com/kavyagajjar/dataset_bias_detector.git`
3. If this skill lives inside a checkout of that repo (`skills/` directory),
   an editable install of the repo root: `pip install -e <repo root>`.

Dependencies (pandas, scipy, scikit-learn, plotly, etc.) come from PyPI. On
Windows, prefer an existing project venv's python over the system alias.

## Step 2: Look at the data before auditing

Read the first few rows and column names (`pandas.read_csv(..., nrows=5)` or
similar). You need this to:

- decide whether auto-detection will work (are there recognizably named
  columns like `gender`, `race`, `age`, `sex`, `ethnicity`?),
- spot the target column and which value is the favorable outcome,
- sanity-check the tool's decisions afterwards.

## Step 3: Run the audit

**Files (CSV, Parquet, JSON)** — use the CLI. Start with auto-detection:

```bash
bias-auditor audit data.csv --auto -o bias_report.html
```

If the user named specific columns, or auto-detection misses/mislabels
something, be explicit instead:

```bash
bias-auditor audit data.csv -p gender -p ethnicity -t hired --positive-label 1 -o bias_report.html
```

**Exit codes matter**: `1` means critical bias was detected — that is a
successful audit, not a failure. `2` means a usage error (bad columns, no
protected attributes found). Don't retry on exit 1.

**Excel files or in-memory DataFrames** — use the Python API:

```python
import pandas as pd
from bias_auditor import BiasAuditor

df = pd.read_excel("data.xlsx")          # or any DataFrame
auditor = BiasAuditor(auto_detect=True)  # or protected_attributes=[...], target_column=...
report = auditor.audit(df)
report.to_html("bias_report.html")
print(report.summary())
```

## Step 4: Verify auto-detection decisions

Auto-detection prints every decision it makes (and records them in the
report's configuration appendix): which columns it treated as protected, how
it binned continuous age, which column it took as the target, and which value
it treated as the favorable outcome.

Check these against what you saw in Step 2. The **positive label** guess is
the one most worth verifying — if the tool picks the unfavorable value, every
disparity ratio inverts and the report blames the wrong group. If any decision
is wrong, rerun with explicit `-p` / `-t` / `--positive-label` flags.

If auto-detection finds no protected attributes (exit code 2 / ValueError),
the columns are probably named unrecognizably. Identify likely sensitive
columns from Step 2 yourself and pass them with `-p`. Ask the user if it's
genuinely ambiguous.

## Step 5: Summarize the results for the user

Read `references/interpreting-results.md` before writing the summary — it
explains each metric, the severity thresholds, and pitfalls to avoid
(over-claiming causation, ignoring sample size).

A good summary:

- Leads with the overall verdict: bias score, number of critical findings.
- Reports the concrete disparities with numbers: "female applicants' approval
  rate is 48.7% vs 71.7% for male applicants — 67.9% of the male rate, below
  the 80% rule threshold; chi-square p = 1.2e-06, so this is statistically
  significant, not noise."
- Distinguishes significant findings from underpowered ones: a disparity in a
  group of 16 rows is a data-collection problem, not a proven bias pattern.
- Points to the HTML report file for the interactive charts and full tables.
- States clearly what bias in the *data* does and does not imply (see the
  reference file) — the tool finds statistical patterns, not intent or cause.

If the audit found nothing critical, say so plainly — don't invent concerns.
Mention what was checked so the user knows the clean result is meaningful.

## Constraints and edge cases

- **Label-bias checks need a binary target.** Multi-class or continuous
  targets: representation, proxy, and missing-data checks still run; disparate
  impact is skipped. Offer to binarize (e.g. rating >= 4) if the user wants
  outcome analysis.
- **Very large files**: the audit loads the full dataset into pandas. For
  multi-GB data, sample first (e.g. 100k rows) and say you did.
- **The HTML report's interactive charts load plotly from a CDN** — offline
  viewers will see tables and findings but blank chart areas.
- **Generated reports may contain sensitive data** (group names, rates).
  Treat the report file with the same care as the dataset.
