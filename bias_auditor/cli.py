"""Command-line interface for the bias auditor."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="bias-auditor")
def main():
    """Dataset Bias Auditor - Detect biases in datasets before training."""
    pass


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--protected", "-p",
    multiple=True,
    help="Protected attribute column names (can specify multiple). "
         "Omit and pass --auto to detect them automatically.",
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Auto-detect protected attributes, target column, and positive label "
         "from column names.",
)
@click.option(
    "--target", "-t",
    default=None,
    help="Target/label column name",
)
@click.option(
    "--positive-label",
    default="1",
    help="Value representing positive outcome",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path (.html or .json)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["summary", "full", "json"]),
    default="summary",
    help="Output format",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["openai", "anthropic", "azure", "local", "none"]),
    default="none",
    help="LLM provider for enhanced analysis",
)
@click.option(
    "--threshold-critical",
    default=0.8,
    type=float,
    help="Disparate impact critical threshold",
)
@click.option(
    "--verbose/--quiet", "-v/-q",
    default=True,
    help="Verbose output",
)
def audit(
    data_path: str,
    protected: tuple,
    auto: bool,
    target: Optional[str],
    positive_label: str,
    output: Optional[str],
    format: str,
    llm_provider: str,
    threshold_critical: float,
    verbose: bool,
):
    """
    Audit a dataset for biases.

    DATA_PATH: Path to the dataset file (CSV, Parquet, or JSON)
    """
    import pandas as pd

    from bias_auditor import BiasAuditor, BiasThresholds

    if not protected and not auto:
        console.print(
            "[red]Error:[/red] Specify protected attributes with -p, "
            "or pass --auto to detect them automatically."
        )
        sys.exit(2)

    # Load data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading dataset...", total=None)

        path = Path(data_path)
        if path.suffix == ".csv":
            data = pd.read_csv(path)
        elif path.suffix == ".parquet":
            data = pd.read_parquet(path)
        elif path.suffix == ".json":
            data = pd.read_json(path)
        else:
            console.print(f"[red]Unsupported file format: {path.suffix}[/red]")
            sys.exit(1)

    console.print(f"[green]✓[/green] Loaded {len(data)} rows, {len(data.columns)} columns")

    # Parse positive label
    try:
        positive_label_parsed = int(positive_label)
    except ValueError:
        try:
            positive_label_parsed = float(positive_label)
        except ValueError:
            positive_label_parsed = positive_label

    # Create auditor
    thresholds = BiasThresholds(disparate_impact_critical=threshold_critical)

    llm_provider_value = None if llm_provider == "none" else llm_provider

    auditor = BiasAuditor(
        protected_attributes=list(protected),
        target_column=target,
        positive_label=positive_label_parsed,
        thresholds=thresholds,
        llm_provider=llm_provider_value,
        auto_detect=auto,
        verbose=verbose,
    )

    # Run audit
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running bias audit...", total=None)
        report = auditor.audit(data, dataset_name=path.name)

    # Output results
    if format == "json":
        console.print(report.to_json())
    elif format == "full":
        console.print(report.summary())
        console.print()
        console.print(report.remediation_plan())
    else:
        _print_summary(report)

    # Save output file
    if output:
        output_path = Path(output)
        if output_path.suffix == ".html":
            report.to_html(str(output_path))
            console.print(f"[green]✓[/green] HTML report saved to {output_path}")
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                f.write(report.to_json())
            console.print(f"[green]✓[/green] JSON report saved to {output_path}")
        else:
            console.print("[yellow]Warning: Unknown output format, saving as text[/yellow]")
            with open(output_path, "w") as f:
                f.write(report.summary())

    # Exit code based on findings
    if report.has_critical_bias:
        sys.exit(1)
    else:
        sys.exit(0)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--protected", "-p",
    multiple=True,
    required=True,
    help="Protected attribute column names",
)
@click.option(
    "--target", "-t",
    default=None,
    help="Target/label column name",
)
def quick_check(data_path: str, protected: tuple, target: Optional[str]):
    """
    Quick bias check - returns key metrics only.

    DATA_PATH: Path to the dataset file
    """
    import pandas as pd

    from bias_auditor import BiasAuditor

    # Load data
    path = Path(data_path)
    if path.suffix == ".csv":
        data = pd.read_csv(path)
    elif path.suffix == ".parquet":
        data = pd.read_parquet(path)
    else:
        console.print(f"[red]Unsupported file format: {path.suffix}[/red]")
        sys.exit(1)

    auditor = BiasAuditor(
        protected_attributes=list(protected),
        target_column=target,
        verbose=False,
    )

    results = auditor.quick_check(data)

    # Display results
    console.print(Panel.fit("Quick Bias Check Results", style="bold blue"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in results["key_metrics"].items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    if results["has_critical_bias"]:
        console.print("[red]⚠ Critical bias detected![/red]")
        sys.exit(1)
    else:
        console.print("[green]✓ No critical bias detected[/green]")
        sys.exit(0)


def _print_summary(report):
    """Print a formatted summary of the audit report."""

    # Header panel
    score_color = "green" if report.overall_bias_score < 0.3 else \
                  "yellow" if report.overall_bias_score < 0.6 else "red"

    console.print(Panel(
        f"[bold]Bias Score:[/bold] [{score_color}]{report.overall_bias_score:.0%}[/{score_color}]\n"
        f"[bold]Critical:[/bold] [red]{len(report.critical_findings)}[/red]  "
        f"[bold]Warnings:[/bold] [yellow]{len(report.warning_findings)}[/yellow]  "
        f"[bold]Total:[/bold] {len(report.findings)}",
        title="Audit Summary",
        border_style="blue",
    ))

    # Category scores
    if report.category_scores:
        table = Table(title="Bias by Category", show_header=True)
        table.add_column("Category")
        table.add_column("Score")
        table.add_column("Level", width=20)

        for cat, score in sorted(report.category_scores.items(), key=lambda x: -x[1]):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            color = "green" if score < 0.3 else "yellow" if score < 0.6 else "red"
            table.add_row(
                cat.replace("_", " ").title(),
                f"{score:.0%}",
                f"[{color}]{bar}[/{color}]",
            )

        console.print(table)

    # Critical findings
    if report.critical_findings:
        console.print("\n[bold red]Critical Findings:[/bold red]")
        for finding in report.critical_findings[:5]:
            console.print(f"  [red]●[/red] {finding.title}")
            console.print(f"    {finding.description[:100]}...")

    # Top remediation
    if report.critical_findings:
        console.print("\n[bold green]Recommended Actions:[/bold green]")
        for finding in report.critical_findings[:3]:
            if finding.remediation_suggestions:
                console.print(f"  • {finding.remediation_suggestions[0]}")


if __name__ == "__main__":
    main()
