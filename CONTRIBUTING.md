# Contributing to Dataset Bias Auditor

Thanks for your interest in contributing! This document covers how to set up a
development environment and submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/kavyagajjar/dataset_bias_detector.git
cd dataset_bias_detector

# Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

The test suite runs with coverage reporting enabled by default (configured in
`pyproject.toml`). All tests must pass before a PR is merged.

## Code Style

- **black** for formatting (line length 100): `black bias_auditor/ tests/`
- **ruff** for linting: `ruff check bias_auditor/`
- **mypy** for type checking: `mypy bias_auditor/ --ignore-missing-imports`

CI runs all three, so run them locally before pushing.

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, adding tests for any new behavior.
3. Ensure the test suite and linters pass.
4. Open a pull request with a clear description of the change and motivation.

## Guidelines

- New bias detectors go in `bias_auditor/detectors/` and should follow the
  existing detector interface (a `detect(data)` method returning a list of
  `BiasFindings`).
- New fairness metrics go in `bias_auditor/metrics/fairness.py` with
  accompanying tests in `tests/test_metrics.py`.
- LLM features must remain optional — the core audit path must work without
  any LLM provider installed or configured.
- Return plain Python types (`bool`, `float`, `int`) from public APIs, not
  numpy scalars — they break `is` comparisons and JSON serialization.

## Reporting Issues

Open a GitHub issue with a minimal reproducible example (a small DataFrame
and the auditor configuration that triggers the problem).
