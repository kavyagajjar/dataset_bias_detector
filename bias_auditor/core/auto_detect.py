"""Automatic detection of protected attributes and target columns.

Enables "just point it at a dataset" usage: column-name heuristics identify
likely protected attributes, continuous attributes like age are binned into
groups, and a binary target column is guessed from common naming conventions.
All detections are surfaced to the caller so they can be reviewed.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import re

import pandas as pd


# Column-name patterns for protected/sensitive attributes.
# Matched case-insensitively against the full column name.
PROTECTED_PATTERNS: dict[str, str] = {
    "gender": r"(gender|^sex$|_sex$|^sex_)",
    "race": r"(race|ethnic)",
    "age": r"(^age$|_age$|^age_|age_group|agegroup|age_band)",
    "religion": r"religio",
    "disability": r"(disab|handicap)",
    "marital_status": r"marital",
    "nationality": r"(nationalit|citizen|immigra|country_of_birth|birth_country)",
    "veteran_status": r"veteran",
    "pregnancy": r"pregnan",
    "sexual_orientation": r"(sexual_orientation|orientation)",
    "zip_code": r"(zip|postal|postcode)",
    "language": r"(^language$|native_language|primary_language)",
}

# Common names for a supervised-learning target column, in priority order.
TARGET_NAME_CANDIDATES: list[str] = [
    "target", "label", "outcome", "y",
    "approved", "approval", "hired", "admitted", "accepted", "granted",
    "default", "churn", "fraud", "readmitted", "recidivism",
    "loan_status", "credit_risk", "decision", "result", "class",
]

# Values that typically represent the favorable/positive outcome.
POSITIVE_VALUE_HINTS: list[str] = [
    "1", "true", "yes", "y", "approved", "hired", "admitted", "accepted",
    "granted", "positive", "good", "pass", "paid",
]

# Bins used when a continuous age column is detected.
AGE_BINS = [0, 25, 35, 50, 65, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50-64", "65+"]

# A column is treated as categorical if it has at most this many unique values.
MAX_CATEGORICAL_CARDINALITY = 20


@dataclass
class AutoDetectionResult:
    """Outcome of automatic dataset inspection."""

    protected_attributes: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    positive_label: Any = None
    # Columns that were derived (e.g. binned age), mapping new -> source column
    derived_columns: dict[str, str] = field(default_factory=dict)
    # Human-readable notes about every decision taken
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "protected_attributes": self.protected_attributes,
            "target_column": self.target_column,
            "positive_label": self.positive_label,
            "derived_columns": self.derived_columns,
            "notes": self.notes,
        }


def detect_protected_attributes(data: pd.DataFrame) -> dict[str, str]:
    """
    Find columns that look like protected attributes by name.

    Returns
    -------
    dict
        Mapping of column name -> matched attribute kind (e.g. 'gender').
    """
    matches: dict[str, str] = {}
    for column in data.columns:
        for kind, pattern in PROTECTED_PATTERNS.items():
            if re.search(pattern, column, flags=re.IGNORECASE):
                matches[column] = kind
                break
    return matches


def detect_target_column(
    data: pd.DataFrame,
    exclude: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Guess the target/label column from common naming conventions.

    Only columns with exactly two distinct non-null values qualify —
    the label-bias detectors assume a binary outcome.
    """
    exclude = exclude or []
    columns_lower = {c.lower(): c for c in data.columns}

    for candidate in TARGET_NAME_CANDIDATES:
        column = columns_lower.get(candidate)
        if column is None or column in exclude:
            continue
        if data[column].dropna().nunique() == 2:
            return column
    return None


def infer_positive_label(series: pd.Series) -> Any:
    """
    Guess which of a binary column's two values is the favorable outcome.
    """
    values = list(series.dropna().unique())

    for value in values:
        if str(value).strip().lower() in POSITIVE_VALUE_HINTS:
            return value

    # Numeric binary: the larger value (1 in {0,1}) is conventionally positive
    try:
        return max(values)
    except TypeError:
        # Mixed/unorderable types: fall back to the less frequent value,
        # since favorable outcomes (approval, hire) are typically the minority
        return series.value_counts().idxmin()


def bin_continuous_attribute(
    data: pd.DataFrame,
    column: str,
    bins: Optional[list] = None,
    labels: Optional[list[str]] = None,
) -> pd.Series:
    """Bin a continuous column (e.g. raw age) into categorical groups."""
    return pd.cut(
        data[column],
        bins=bins or AGE_BINS,
        labels=labels or AGE_LABELS,
        right=False,
    ).astype(str)


def auto_detect(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
) -> tuple[pd.DataFrame, AutoDetectionResult]:
    """
    Inspect a dataset and detect protected attributes and target column.

    Continuous protected attributes (raw age) are binned into a derived
    ``<column>_group`` column on a copy of the data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to inspect.
    target_column : str, optional
        Known target column; skips target detection when provided.

    Returns
    -------
    (pd.DataFrame, AutoDetectionResult)
        A copy of the data (with any derived columns added) and the
        detection result describing what was found.
    """
    result = AutoDetectionResult()
    data = data.copy()

    name_matches = detect_protected_attributes(data)
    for column, kind in name_matches.items():
        series = data[column]
        n_unique = series.dropna().nunique()

        if pd.api.types.is_numeric_dtype(series) and n_unique > MAX_CATEGORICAL_CARDINALITY:
            if kind == "age":
                derived = f"{column}_group"
                data[derived] = bin_continuous_attribute(data, column)
                result.protected_attributes.append(derived)
                result.derived_columns[derived] = column
                result.notes.append(
                    f"Detected continuous '{column}' ({kind}); binned into '{derived}' "
                    f"with groups {AGE_LABELS}."
                )
            else:
                result.notes.append(
                    f"Column '{column}' looks like {kind} but has {n_unique} numeric values; "
                    "skipped (too granular to treat as groups). Bin it manually to include it."
                )
        elif n_unique < 2:
            result.notes.append(
                f"Column '{column}' looks like {kind} but has fewer than 2 groups; skipped."
            )
        else:
            result.protected_attributes.append(column)
            result.notes.append(f"Detected protected attribute '{column}' ({kind}).")

    if target_column:
        result.target_column = target_column
    else:
        result.target_column = detect_target_column(
            data, exclude=result.protected_attributes
        )
        if result.target_column:
            result.notes.append(f"Detected target column '{result.target_column}'.")
        else:
            result.notes.append(
                "No binary target column detected; label bias checks will be skipped."
            )

    if result.target_column is not None:
        result.positive_label = infer_positive_label(data[result.target_column])
        result.notes.append(
            f"Treating '{result.positive_label}' as the positive/favorable outcome "
            f"in '{result.target_column}'. Verify this is correct."
        )

    return data, result
