# Interpreting Bias Audit Results

How to read each metric the auditor reports, and how to communicate findings
responsibly.

## Overall bias score

0–1, computed as the maximum category severity (critical finding = 1.0,
warning = 0.5, info = 0.1). It is a triage signal, not a measurement: a score
of 1.0 means "at least one critical issue exists", not "the dataset is 100%
biased". Never report it as a percentage of bias.

## Finding categories

### Representation bias
- **Underrepresented group**: a group's share of the dataset is below the
  threshold (default: critical under 5%, warning under 10%). Models see too
  few examples to learn reliable patterns for these groups.
- **Imbalance ratio**: largest group ÷ smallest group (default: critical
  above 10x). Drives models to optimize for the majority.
- These are data-collection findings. The fix is more/better data or
  resampling — not necessarily evidence of discriminatory outcomes.

### Label bias
- **Disparate impact ratio (DIR)**: positive-outcome rate of the
  disadvantaged group ÷ rate of the advantaged group. The "80% rule"
  (from US EEOC guidance) flags DIR below 0.8 as adverse impact.
- **Statistical parity difference**: max group rate minus min group rate
  (default: critical above 0.2).
- A label disparity means outcomes differ by group in the data. It does NOT
  by itself prove discrimination — legitimate factors may correlate with
  group membership. Say "disparity", not "discrimination".

### Feature proxy bias
Features that statistically encode a protected attribute (e.g. zip code
predicting race, via correlation/mutual information). Removing the protected
column while keeping a strong proxy does not de-bias a model. Report which
features are proxies and how strongly.

### Missing-data bias
Differential missingness: a column is missing at different rates across
protected groups (suggesting data collection differed by group), or
missingness correlates with the outcome (MNAR). Imputation strategies that
ignore this bake the pattern in.

### Intersectional
The same checks applied to attribute pairs (e.g. gender × race). Small
intersectional groups are common and often underpowered — check the group
size before treating a rate difference as meaningful.

## Statistical significance (chi-square p-values)

The Group Breakdown section reports a chi-square test of independence between
each protected attribute and the target.

- **p < 0.05**: the outcome disparity is unlikely to be random noise given the
  sample size. Report it as statistically significant.
- **p >= 0.05**: the observed disparity could plausibly be chance. Do not
  present it as an established pattern — say the data is inconclusive.
- Significance scales with sample size: in a 10M-row dataset a trivial 0.5%
  rate difference is "significant" but practically irrelevant; in a 200-row
  dataset a large disparity may not reach significance. Report both the
  effect size (the rate gap) and the p-value, and lead with effect size.

## Sample-size caveats

- Groups under ~30 rows: rate estimates are unstable. Flag the group as
  underrepresented; do not quote its positive rate as a reliable statistic.
- The tool's `min_group_size` (default 30) suppresses some checks for tiny
  groups — a silent group in the findings may simply have been too small
  to test.

## Communicating results

- Data bias findings describe the dataset, not the people or process that
  produced it, and not any model trained on it — though they predict model
  risk.
- Prefer concrete numbers over adjectives: "approval rate 48.7% vs 71.7%"
  beats "severely biased".
- When the audit is clean, say what was checked (which attributes, which
  detectors) so the user understands the scope of the clean result.
- Recommend the remediation actions the report lists (resampling,
  reweighting, collecting more data, reviewing labeling) rather than
  inventing new ones; the `bias_auditor.remediation` module implements
  several of them.
