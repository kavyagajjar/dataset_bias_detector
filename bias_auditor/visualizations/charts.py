"""Chart generation for bias audit reports."""

import base64
from io import BytesIO
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.patches as mpatches  # noqa: F401
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px  # noqa: F401
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class BiasVisualizer:
    """
    Generate visualizations for bias audit reports.

    Supports both static (matplotlib) and interactive (plotly) visualizations.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize visualizer.

        Parameters
        ----------
        backend : str
            Visualization backend: 'matplotlib', 'plotly', or 'auto'
        """
        if backend == "auto":
            self.backend = "plotly" if HAS_PLOTLY else "matplotlib" if HAS_MATPLOTLIB else None
        else:
            self.backend = backend

        if self.backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib not installed. Install with: pip install matplotlib")
        if self.backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("plotly not installed. Install with: pip install plotly")

    def group_distribution(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        title: Optional[str] = None,
        reference_dist: Optional[dict] = None,
    ) -> str:
        """
        Plot distribution of protected attribute groups.

        Returns base64-encoded image or HTML for plotly.
        """
        if self.backend == "plotly":
            return self._group_distribution_plotly(data, protected_attr, title, reference_dist)
        else:
            return self._group_distribution_matplotlib(data, protected_attr, title, reference_dist)

    def _group_distribution_plotly(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        title: Optional[str],
        reference_dist: Optional[dict],
    ) -> str:
        """Generate group distribution chart with plotly."""
        counts = data[protected_attr].value_counts()

        fig = go.Figure()

        # Observed distribution
        fig.add_trace(
            go.Bar(
                name="Observed",
                x=counts.index.astype(str),
                y=counts.values / counts.sum(),
                marker_color="#667eea",
                text=[f"{v/counts.sum():.1%}" for v in counts.values],
                textposition="outside",
            )
        )

        # Reference distribution if provided
        if reference_dist:
            ref_values = [reference_dist.get(str(k), 0) for k in counts.index]
            fig.add_trace(
                go.Bar(
                    name="Reference",
                    x=counts.index.astype(str),
                    y=ref_values,
                    marker_color="#764ba2",
                    opacity=0.6,
                )
            )

        fig.update_layout(
            title=title or f"Distribution of {protected_attr}",
            xaxis_title=protected_attr,
            yaxis_title="Proportion",
            barmode="group",
            template="plotly_white",
            height=400,
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _group_distribution_matplotlib(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        title: Optional[str],
        reference_dist: Optional[dict],
    ) -> str:
        """Generate group distribution chart with matplotlib."""
        counts = data[protected_attr].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(counts))
        width = 0.35

        # Observed distribution
        bars1 = ax.bar(
            x - width / 2 if reference_dist else x,
            counts.values / counts.sum(),
            width if reference_dist else 0.6,
            label="Observed",
            color="#667eea",
        )

        # Reference distribution if provided
        if reference_dist:
            ref_values = [reference_dist.get(str(k), 0) for k in counts.index]
            ax.bar(x + width / 2, ref_values, width, label="Reference", color="#764ba2", alpha=0.6)

        ax.set_xlabel(protected_attr)
        ax.set_ylabel("Proportion")
        ax.set_title(title or f"Distribution of {protected_attr}")
        ax.set_xticks(x)
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

        if reference_dist:
            ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def label_rates_by_group(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
        positive_label: Any = 1,
        title: Optional[str] = None,
    ) -> str:
        """Plot positive label rates by group."""
        if self.backend == "plotly":
            return self._label_rates_plotly(
                data, protected_attr, target_column, positive_label, title
            )
        else:
            return self._label_rates_matplotlib(
                data, protected_attr, target_column, positive_label, title
            )

    def _label_rates_plotly(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
        positive_label: Any,
        title: Optional[str],
    ) -> str:
        """Generate label rates chart with plotly."""
        rates = (
            data.groupby(protected_attr)[target_column]
            .apply(lambda x: (x == positive_label).mean())
            .sort_values(ascending=False)
        )

        # Color by rate (red for low, green for high)
        colors = [
            "#dc3545" if r < 0.3 else "#ffc107" if r < 0.6 else "#28a745" for r in rates.values
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=rates.index.astype(str),
                y=rates.values,
                marker_color=colors,
                text=[f"{v:.1%}" for v in rates.values],
                textposition="outside",
            )
        )

        # Add 80% threshold line
        overall_rate = (data[target_column] == positive_label).mean()
        fig.add_hline(
            y=overall_rate * 0.8,
            line_dash="dash",
            line_color="red",
            annotation_text=f"80% Rule Threshold ({overall_rate * 0.8:.1%})",
        )

        fig.update_layout(
            title=title or f"Positive Outcome Rate by {protected_attr}",
            xaxis_title=protected_attr,
            yaxis_title="Positive Rate",
            template="plotly_white",
            height=400,
            yaxis=dict(range=[0, 1.1]),
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _label_rates_matplotlib(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
        positive_label: Any,
        title: Optional[str],
    ) -> str:
        """Generate label rates chart with matplotlib."""
        rates = (
            data.groupby(protected_attr)[target_column]
            .apply(lambda x: (x == positive_label).mean())
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [
            "#dc3545" if r < 0.3 else "#ffc107" if r < 0.6 else "#28a745" for r in rates.values
        ]

        bars = ax.bar(range(len(rates)), rates.values, color=colors)

        # 80% rule line
        overall_rate = (data[target_column] == positive_label).mean()
        ax.axhline(
            y=overall_rate * 0.8,
            color="red",
            linestyle="--",
            label=f"80% Rule ({overall_rate * 0.8:.1%})",
        )

        ax.set_xlabel(protected_attr)
        ax.set_ylabel("Positive Rate")
        ax.set_title(title or f"Positive Outcome Rate by {protected_attr}")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index.astype(str), rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()

        for bar, rate in zip(bars, rates.values):
            ax.annotate(
                f"{rate:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, rate),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def category_scores_radar(
        self,
        category_scores: dict[str, float],
        title: Optional[str] = None,
    ) -> str:
        """Generate radar chart of bias category scores."""
        if self.backend == "plotly":
            return self._radar_plotly(category_scores, title)
        else:
            return self._radar_matplotlib(category_scores, title)

    def _radar_plotly(self, category_scores: dict[str, float], title: Optional[str]) -> str:
        """Generate radar chart with plotly."""
        categories = list(category_scores.keys())
        values = list(category_scores.values())

        # Close the radar chart
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=[c.replace("_", " ").title() for c in categories_closed],
                fill="toself",
                fillcolor="rgba(102, 126, 234, 0.3)",
                line=dict(color="#667eea", width=2),
                name="Bias Score",
            )
        )

        fig.update_layout(
            title=title or "Bias by Category",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ),
            ),
            template="plotly_white",
            height=400,
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _radar_matplotlib(self, category_scores: dict[str, float], title: Optional[str]) -> str:
        """Generate radar chart with matplotlib."""
        categories = list(category_scores.keys())
        values = list(category_scores.values())

        # Number of categories
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values, "o-", linewidth=2, color="#667eea")
        ax.fill(angles, values, alpha=0.3, color="#667eea")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
        ax.set_ylim(0, 1)
        ax.set_title(title or "Bias by Category", y=1.1)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def bias_heatmap(
        self,
        data: pd.DataFrame,
        protected_attrs: list[str],
        target_column: str,
        positive_label: Any = 1,
        title: Optional[str] = None,
    ) -> str:
        """Generate heatmap of disparate impact across attribute pairs."""
        if len(protected_attrs) < 2:
            return ""

        if self.backend == "plotly":
            return self._heatmap_plotly(data, protected_attrs, target_column, positive_label, title)
        else:
            return self._heatmap_matplotlib(
                data, protected_attrs, target_column, positive_label, title
            )

    def _heatmap_plotly(
        self,
        data: pd.DataFrame,
        protected_attrs: list[str],
        target_column: str,
        positive_label: Any,
        title: Optional[str],
    ) -> str:
        """Generate heatmap with plotly."""
        # Calculate disparate impact ratios for all pairs
        n = len(protected_attrs)
        matrix = np.zeros((n, n))

        for i, attr in enumerate(protected_attrs):
            rates = data.groupby(attr)[target_column].apply(lambda x: (x == positive_label).mean())
            if len(rates) >= 2:
                dir_val = rates.min() / rates.max() if rates.max() > 0 else 1
                matrix[i, i] = dir_val

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[a.replace("_", " ").title() for a in protected_attrs],
                y=[a.replace("_", " ").title() for a in protected_attrs],
                colorscale=[[0, "#dc3545"], [0.5, "#ffc107"], [1, "#28a745"]],
                zmin=0,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(title="DIR"),
            )
        )

        fig.update_layout(
            title=title or "Disparate Impact Ratio by Attribute",
            template="plotly_white",
            height=400,
        )

        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _heatmap_matplotlib(
        self,
        data: pd.DataFrame,
        protected_attrs: list[str],
        target_column: str,
        positive_label: Any,
        title: Optional[str],
    ) -> str:
        """Generate heatmap with matplotlib."""
        n = len(protected_attrs)
        matrix = np.zeros((n, n))

        for i, attr in enumerate(protected_attrs):
            rates = data.groupby(attr)[target_column].apply(lambda x: (x == positive_label).mean())
            if len(rates) >= 2:
                dir_val = rates.min() / rates.max() if rates.max() > 0 else 1
                matrix[i, i] = dir_val

        fig, ax = plt.subplots(figsize=(8, 8))

        from matplotlib.colors import LinearSegmentedColormap

        colors = ["#dc3545", "#ffc107", "#28a745"]
        cmap = LinearSegmentedColormap.from_list("bias", colors)

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            [a.replace("_", " ").title() for a in protected_attrs], rotation=45, ha="right"
        )
        ax.set_yticklabels([a.replace("_", " ").title() for a in protected_attrs])

        # Add text annotations
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=12)

        plt.colorbar(im, ax=ax, label="Disparate Impact Ratio")
        ax.set_title(title or "Disparate Impact Ratio by Attribute")

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def intersectional_heatmap(
        self,
        data: pd.DataFrame,
        attr1: str,
        attr2: str,
        target_column: str,
        positive_label: Any = 1,
        title: Optional[str] = None,
    ) -> str:
        """
        Heatmap of positive-outcome rates for the intersection of two
        protected attributes (attr1 groups x attr2 groups).
        """
        rates = data.pivot_table(
            index=attr1,
            columns=attr2,
            values=target_column,
            aggfunc=lambda x: (x == positive_label).mean(),
        )
        counts = data.pivot_table(index=attr1, columns=attr2, values=target_column, aggfunc="count")

        title = title or f"Positive Rate: {attr1} × {attr2}"

        if self.backend == "plotly":
            text = [
                [
                    (
                        f"{rates.iloc[i, j]:.1%}<br>n={int(counts.iloc[i, j])}"
                        if pd.notna(rates.iloc[i, j])
                        else ""
                    )
                    for j in range(rates.shape[1])
                ]
                for i in range(rates.shape[0])
            ]
            fig = go.Figure(
                data=go.Heatmap(
                    z=rates.values,
                    x=rates.columns.astype(str),
                    y=rates.index.astype(str),
                    colorscale=[[0, "#dc3545"], [0.5, "#ffc107"], [1, "#28a745"]],
                    zmin=0,
                    zmax=1,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    colorbar=dict(title="Positive Rate"),
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=attr2,
                yaxis_title=attr1,
                template="plotly_white",
                height=400,
            )
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            from matplotlib.colors import LinearSegmentedColormap

            cmap = LinearSegmentedColormap.from_list("rate", ["#dc3545", "#ffc107", "#28a745"])
            im = ax.imshow(rates.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(rates.shape[1]))
            ax.set_yticks(range(rates.shape[0]))
            ax.set_xticklabels(rates.columns.astype(str), rotation=45, ha="right")
            ax.set_yticklabels(rates.index.astype(str))
            ax.set_xlabel(attr2)
            ax.set_ylabel(attr1)
            for i in range(rates.shape[0]):
                for j in range(rates.shape[1]):
                    if pd.notna(rates.iloc[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{rates.iloc[i, j]:.0%}\nn={int(counts.iloc[i, j])}",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            plt.colorbar(im, ax=ax, label="Positive Rate")
            ax.set_title(title)
            plt.tight_layout()
            return self._fig_to_base64(fig)

    def findings_timeline(
        self,
        findings: list,
        title: Optional[str] = None,
    ) -> str:
        """Generate timeline/bar chart of findings by severity."""
        if not findings:
            return ""

        severity_counts = {}
        category_counts = {}

        for f in findings:
            sev = f.severity.value
            cat = f.category.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if self.backend == "plotly":
            fig = make_subplots(rows=1, cols=2, subplot_titles=("By Severity", "By Category"))

            # Severity chart
            colors_sev = {
                "critical": "#dc3545",
                "warning": "#ffc107",
                "info": "#17a2b8",
                "none": "#28a745",
            }
            fig.add_trace(
                go.Bar(
                    x=list(severity_counts.keys()),
                    y=list(severity_counts.values()),
                    marker_color=[colors_sev.get(k, "#666") for k in severity_counts.keys()],
                ),
                row=1,
                col=1,
            )

            # Category chart
            fig.add_trace(
                go.Bar(
                    x=[c.replace("_", " ").title() for c in category_counts.keys()],
                    y=list(category_counts.values()),
                    marker_color="#667eea",
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                title=title or "Findings Overview",
                showlegend=False,
                template="plotly_white",
                height=400,
            )

            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            colors_sev = {
                "critical": "#dc3545",
                "warning": "#ffc107",
                "info": "#17a2b8",
                "none": "#28a745",
            }
            ax1.bar(
                severity_counts.keys(),
                severity_counts.values(),
                color=[colors_sev.get(k, "#666") for k in severity_counts.keys()],
            )
            ax1.set_title("By Severity")
            ax1.set_ylabel("Count")

            ax2.bar(
                [c.replace("_", " ").title() for c in category_counts.keys()],
                category_counts.values(),
                color="#667eea",
            )
            ax2.set_title("By Category")
            ax2.tick_params(axis="x", rotation=45)

            plt.suptitle(title or "Findings Overview")
            plt.tight_layout()
            return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_base64}" alt="Chart">'


# Convenience functions
def plot_group_distribution(data: pd.DataFrame, protected_attr: str, **kwargs) -> str:
    """Plot distribution of protected attribute groups."""
    viz = BiasVisualizer()
    return viz.group_distribution(data, protected_attr, **kwargs)


def plot_label_rates(
    data: pd.DataFrame, protected_attr: str, target_column: str, positive_label: Any = 1, **kwargs
) -> str:
    """Plot positive label rates by group."""
    viz = BiasVisualizer()
    return viz.label_rates_by_group(data, protected_attr, target_column, positive_label, **kwargs)


def plot_category_scores(category_scores: dict[str, float], **kwargs) -> str:
    """Generate radar chart of category scores."""
    viz = BiasVisualizer()
    return viz.category_scores_radar(category_scores, **kwargs)


def plot_fairness_radar(category_scores: dict[str, float], **kwargs) -> str:
    """Alias for plot_category_scores."""
    return plot_category_scores(category_scores, **kwargs)


def generate_all_visualizations(
    data: pd.DataFrame,
    protected_attrs: list[str],
    target_column: Optional[str] = None,
    positive_label: Any = 1,
    category_scores: Optional[dict[str, float]] = None,
) -> dict[str, str]:
    """
    Generate all available visualizations for a dataset.

    Returns dictionary mapping visualization name to HTML/base64 content.
    """
    viz = BiasVisualizer()
    visualizations = {}

    # Group distributions
    for attr in protected_attrs:
        if attr in data.columns:
            visualizations[f"distribution_{attr}"] = viz.group_distribution(data, attr)

    # Label rates
    if target_column and target_column in data.columns:
        for attr in protected_attrs:
            if attr in data.columns:
                visualizations[f"label_rates_{attr}"] = viz.label_rates_by_group(
                    data, attr, target_column, positive_label
                )

    # Category scores radar
    if category_scores:
        visualizations["category_radar"] = viz.category_scores_radar(category_scores)

    # Intersectional heatmaps for each attribute pair
    if len(protected_attrs) >= 2 and target_column and target_column in data.columns:
        available_attrs = [a for a in protected_attrs if a in data.columns]
        for i in range(len(available_attrs)):
            for j in range(i + 1, len(available_attrs)):
                attr1, attr2 = available_attrs[i], available_attrs[j]
                visualizations[f"intersectional_{attr1}_x_{attr2}"] = viz.intersectional_heatmap(
                    data, attr1, attr2, target_column, positive_label
                )

    return visualizations
