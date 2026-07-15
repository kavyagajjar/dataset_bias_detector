"""HTML report generation."""

from typing import Optional
from datetime import datetime

from bias_auditor.core.report import AuditReport, BiasSeverity, BiasCategory


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Bias Audit Report</title>
    <style>
        :root {{
            --critical-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --success-color: #28a745;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #212529;
            --border-color: #dee2e6;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        
        .score-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .score-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}
        
        .score-item {{
            text-align: center;
            padding: 1rem;
            background: var(--bg-color);
            border-radius: 8px;
        }}
        
        .score-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .score-label {{
            font-size: 0.85rem;
            color: #666;
            text-transform: uppercase;
        }}
        
        .severity-critical {{ color: var(--critical-color); }}
        .severity-warning {{ color: var(--warning-color); }}
        .severity-info {{ color: var(--info-color); }}
        .severity-success {{ color: var(--success-color); }}
        
        .findings-section {{
            margin-bottom: 2rem;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .finding-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--border-color);
        }}
        
        .finding-card.critical {{
            border-left-color: var(--critical-color);
        }}
        
        .finding-card.warning {{
            border-left-color: var(--warning-color);
        }}
        
        .finding-card.info {{
            border-left-color: var(--info-color);
        }}
        
        .finding-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }}
        
        .finding-title {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .finding-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-critical {{
            background-color: var(--critical-color);
            color: white;
        }}
        
        .badge-warning {{
            background-color: var(--warning-color);
            color: #212529;
        }}
        
        .badge-info {{
            background-color: var(--info-color);
            color: white;
        }}
        
        .finding-description {{
            color: #555;
            margin-bottom: 1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }}
        
        .metric {{
            background: var(--bg-color);
            padding: 0.75rem;
            border-radius: 6px;
        }}
        
        .metric-label {{
            font-size: 0.75rem;
            color: #666;
        }}
        
        .metric-value {{
            font-weight: 600;
        }}
        
        .remediation {{
            background: #e8f5e9;
            padding: 1rem;
            border-radius: 6px;
            margin-top: 1rem;
        }}
        
        .remediation-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #2e7d32;
        }}
        
        .remediation ul {{
            margin-left: 1.5rem;
        }}
        
        .category-scores {{
            margin-top: 1.5rem;
        }}
        
        .category-bar {{
            margin-bottom: 0.75rem;
        }}
        
        .category-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }}
        
        .bar-container {{
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .executive-summary {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }}

        .data-table th {{
            background: #f1f3f5;
            text-align: left;
            padding: 0.6rem 1rem;
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #555;
        }}

        .data-table td {{
            padding: 0.6rem 1rem;
            border-top: 1px solid var(--border-color);
        }}

        .data-table .num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .table-caption {{
            font-size: 0.85rem;
            color: #666;
            margin: -0.5rem 0 1.5rem 0;
        }}

        .rate-bar {{
            display: inline-block;
            height: 8px;
            background: #667eea;
            border-radius: 4px;
            vertical-align: middle;
            margin-right: 0.5rem;
        }}

        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1rem;
        }}

        .chart-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            overflow-x: auto;
        }}

        .chart-card img {{
            max-width: 100%;
            height: auto;
        }}

        details.config-appendix {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}

        details.config-appendix summary {{
            cursor: pointer;
            font-weight: 600;
            padding: 0.5rem 0;
        }}

        details.config-appendix pre {{
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            margin-top: 0.75rem;
        }}

        .detection-notes {{
            background: #fff8e1;
            border-left: 4px solid var(--warning-color);
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        
        .executive-summary h2 {{
            margin-bottom: 1rem;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            header {{
                padding: 1.5rem;
            }}
            
            header h1 {{
                font-size: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dataset Bias Audit Report</h1>
            <div class="meta">
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Audit ID:</strong> {audit_id}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
        </header>
        
        <div class="score-card">
            <div class="score-grid">
                <div class="score-item">
                    <div class="score-value {overall_score_class}">{overall_score:.0%}</div>
                    <div class="score-label">Bias Score</div>
                </div>
                <div class="score-item">
                    <div class="score-value severity-critical">{critical_count}</div>
                    <div class="score-label">Critical Issues</div>
                </div>
                <div class="score-item">
                    <div class="score-value severity-warning">{warning_count}</div>
                    <div class="score-label">Warnings</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{total_findings}</div>
                    <div class="score-label">Total Findings</div>
                </div>
            </div>
            
            <div class="category-scores">
                <h3>Bias by Category</h3>
                {category_bars}
            </div>
        </div>
        
        {executive_summary_section}

        {profile_section}

        {group_stats_section}

        <div class="findings-section">
            <h2 class="section-title">Critical Findings</h2>
            {critical_findings}
        </div>

        <div class="findings-section">
            <h2 class="section-title">Warnings</h2>
            {warning_findings}
        </div>

        <div class="findings-section">
            <h2 class="section-title">Informational</h2>
            {info_findings}
        </div>

        {visualizations_section}

        {config_appendix}

        <footer>
            <p>Generated by Dataset Bias Auditor</p>
        </footer>
    </div>
</body>
</html>"""


FINDING_CARD_TEMPLATE = """
<div class="finding-card {severity_class}">
    <div class="finding-header">
        <div class="finding-title">{title}</div>
        <span class="finding-badge badge-{severity_class}">{severity}</span>
    </div>
    <p class="finding-description">{description}</p>
    <div class="metrics-grid">
        {metrics_html}
    </div>
    {remediation_html}
</div>
"""


def generate_html_report(report: AuditReport) -> str:
    """
    Generate an HTML report from an audit report.
    
    Parameters
    ----------
    report : AuditReport
        The audit report to render.
    
    Returns
    -------
    str
        HTML content.
    """
    # Overall score class
    if report.overall_bias_score < 0.3:
        overall_score_class = "severity-success"
    elif report.overall_bias_score < 0.6:
        overall_score_class = "severity-warning"
    else:
        overall_score_class = "severity-critical"
    
    # Category bars
    category_bars = ""
    for category, score in report.category_scores.items():
        color = "#28a745" if score < 0.3 else "#ffc107" if score < 0.6 else "#dc3545"
        category_bars += f"""
        <div class="category-bar">
            <div class="category-label">
                <span>{category.replace('_', ' ').title()}</span>
                <span>{score:.0%}</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: {score * 100}%; background-color: {color};"></div>
            </div>
        </div>
        """
    
    # Executive summary
    executive_summary_section = ""
    if report.executive_summary:
        executive_summary_section = f"""
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>{report.executive_summary}</p>
        </div>
        """
    
    # Findings by severity
    critical_findings = _render_findings(report.critical_findings)
    warning_findings = _render_findings(report.warning_findings)
    info_findings = _render_findings(
        [f for f in report.findings if f.severity == BiasSeverity.INFO]
    )
    
    # Render template
    html = HTML_TEMPLATE.format(
        dataset_name=report.dataset_name or "Unknown Dataset",
        audit_id=report.audit_id,
        timestamp=report.audit_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        overall_score=report.overall_bias_score,
        overall_score_class=overall_score_class,
        critical_count=len(report.critical_findings),
        warning_count=len(report.warning_findings),
        total_findings=len(report.findings),
        category_bars=category_bars,
        executive_summary_section=executive_summary_section,
        profile_section=_render_profile_section(report),
        group_stats_section=_render_group_stats_section(report),
        critical_findings=critical_findings or "<p>No critical findings.</p>",
        warning_findings=warning_findings or "<p>No warnings.</p>",
        info_findings=info_findings or "<p>No informational findings.</p>",
        visualizations_section=_render_visualizations_section(report),
        config_appendix=_render_config_appendix(report),
    )

    return html


def _render_profile_section(report: AuditReport) -> str:
    """Render the dataset overview section from the report profile."""
    profile = report.profile
    if profile is None:
        return ""

    # Missing-rate table (only columns with missing values)
    missing = {
        col: rate for col, rate in sorted(
            profile.missing_rates.items(), key=lambda kv: -kv[1]
        ) if rate > 0
    }
    if missing:
        missing_rows = "\n".join(
            f"<tr><td>{col}</td>"
            f"<td class='num'>{rate:.1%}</td></tr>"
            for col, rate in list(missing.items())[:15]
        )
        missing_html = f"""
        <div>
            <h3>Missing Data</h3>
            <table class="data-table">
                <tr><th>Column</th><th>Missing</th></tr>
                {missing_rows}
            </table>
        </div>
        """
    else:
        missing_html = "<div><h3>Missing Data</h3><p>No missing values detected.</p></div>"

    # Target distribution
    target_html = ""
    if profile.target_distribution:
        target_rows = "\n".join(
            f"<tr><td>{value}</td><td class='num'>{share:.1%}</td></tr>"
            for value, share in profile.target_distribution.items()
        )
        target_html = f"""
        <div>
            <h3>Target Distribution</h3>
            <table class="data-table">
                <tr><th>Value</th><th>Share</th></tr>
                {target_rows}
            </table>
        </div>
        """

    return f"""
    <div class="findings-section">
        <h2 class="section-title">Dataset Overview</h2>
        <div class="score-card">
            <div class="score-grid">
                <div class="score-item">
                    <div class="score-value">{profile.n_rows:,}</div>
                    <div class="score-label">Rows</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{profile.n_columns}</div>
                    <div class="score-label">Columns</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{len(profile.protected_attribute_distributions)}</div>
                    <div class="score-label">Protected Attributes</div>
                </div>
            </div>
        </div>
        <div class="overview-grid">
            {target_html}
            {missing_html}
        </div>
    </div>
    """


def _render_group_stats_section(report: AuditReport) -> str:
    """Render per-group breakdown tables for each protected attribute."""
    if not report.group_stats:
        return ""

    tables = ""
    for attr, stats in report.group_stats.items():
        groups = stats.get("groups", [])
        if not groups:
            continue

        has_rate = any("positive_rate" in g for g in groups)
        rate_header = "<th>Positive Rate</th>" if has_rate else ""

        rows = ""
        for g in groups:
            rate_cell = ""
            if has_rate:
                rate = g.get("positive_rate")
                if rate is not None:
                    bar_width = int(rate * 100)
                    rate_cell = (
                        f"<td class='num'>"
                        f"<span class='rate-bar' style='width:{bar_width}px'></span>"
                        f"{rate:.1%}</td>"
                    )
                else:
                    rate_cell = "<td class='num'>—</td>"
            rows += (
                f"<tr><td>{g['group']}</td>"
                f"<td class='num'>{g['count']:,}</td>"
                f"<td class='num'>{g['share']:.1%}</td>"
                f"{rate_cell}</tr>\n"
            )

        p_value = stats.get("chi2_p_value")
        caption = ""
        if p_value is not None:
            significance = (
                "statistically significant (p &lt; 0.05)" if p_value < 0.05
                else "not statistically significant (p &ge; 0.05)"
            )
            caption = (
                f"<p class='table-caption'>Chi-square test of independence between "
                f"<strong>{attr}</strong> and the target: p = {p_value:.2g} — "
                f"the outcome disparity is {significance}.</p>"
            )

        tables += f"""
        <h3>{attr.replace('_', ' ').title()}</h3>
        <table class="data-table">
            <tr><th>Group</th><th>Count</th><th>Share</th>{rate_header}</tr>
            {rows}
        </table>
        {caption}
        """

    if not tables:
        return ""

    return f"""
    <div class="findings-section">
        <h2 class="section-title">Group Breakdown</h2>
        {tables}
    </div>
    """


def _render_visualizations_section(report: AuditReport) -> str:
    """Render embedded charts (plotly HTML fragments or base64 images)."""
    if not report.visualizations:
        return ""

    charts = ""
    for name, content in report.visualizations.items():
        if not content:
            continue
        label = name.replace("_", " ").title()
        charts += f"""
        <div class="chart-card">
            <h3>{label}</h3>
            {content}
        </div>
        """

    if not charts:
        return ""

    return f"""
    <div class="findings-section">
        <h2 class="section-title">Visualizations</h2>
        {charts}
    </div>
    """


def _render_config_appendix(report: AuditReport) -> str:
    """Render the audit configuration as a collapsible appendix."""
    if not report.config_summary:
        return ""

    import json as _json

    detection_html = ""
    auto = report.config_summary.get("auto_detection")
    if auto and auto.get("notes"):
        notes = "".join(f"<li>{n}</li>" for n in auto["notes"])
        detection_html = f"""
        <div class="detection-notes">
            <strong>Auto-detection was used.</strong> Review these decisions:
            <ul style="margin-left: 1.5rem;">{notes}</ul>
        </div>
        """

    config_json = _json.dumps(report.config_summary, indent=2, default=str)
    return f"""
    {detection_html}
    <details class="config-appendix">
        <summary>Audit Configuration</summary>
        <pre>{config_json}</pre>
    </details>
    """


def _render_findings(findings: list) -> str:
    """Render a list of findings as HTML cards."""
    if not findings:
        return ""
    
    html = ""
    for finding in findings:
        severity_class = finding.severity.value
        
        # Render metrics
        metrics_html = ""
        for key, value in list(finding.metrics.items())[:6]:
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            metrics_html += f"""
            <div class="metric">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value">{value_str}</div>
            </div>
            """
        
        # Render remediation
        remediation_html = ""
        if finding.remediation_suggestions:
            suggestions = "\n".join(f"<li>{s}</li>" for s in finding.remediation_suggestions)
            remediation_html = f"""
            <div class="remediation">
                <div class="remediation-title">Recommended Actions</div>
                <ul>{suggestions}</ul>
            </div>
            """
        
        html += FINDING_CARD_TEMPLATE.format(
            severity_class=severity_class,
            title=finding.title,
            severity=finding.severity.value.upper(),
            description=finding.description,
            metrics_html=metrics_html,
            remediation_html=remediation_html,
        )
    
    return html
