"""JSON export utilities."""

from typing import Any

from bias_auditor.core.report import AuditReport


def export_dict(report: AuditReport) -> dict[str, Any]:
    """
    Export audit report as a dictionary.

    Parameters
    ----------
    report : AuditReport
        The audit report.

    Returns
    -------
    dict
        Dictionary representation.
    """
    return report.to_dict()


def export_json(
    report: AuditReport,
    output_path: str = None,
    indent: int = 2,
) -> str:
    """
    Export audit report as JSON.

    Parameters
    ----------
    report : AuditReport
        The audit report.
    output_path : str, optional
        If provided, write JSON to this file.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        JSON string.
    """
    json_str = report.to_json(indent=indent)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)

    return json_str
