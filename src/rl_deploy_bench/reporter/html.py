"""HTML report generator with interactive Plotly charts.

Generates self-contained HTML reports with interactive latency
distribution histograms, comparison bar charts, and metric tables.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..benchmark.accuracy import AccuracyComparisonResult
from ..benchmark.latency import BenchmarkResult
from ..utils.platform import PlatformInfo


def generate_html_report(
    output_path: str,
    benchmark_results: List[BenchmarkResult],
    model_names: Optional[List[str]] = None,
    accuracy_results: Optional[List[AccuracyComparisonResult]] = None,
    platform_info: Optional[PlatformInfo] = None,
    title: str = "RL Model Deployment Benchmark Report",
) -> str:
    """Generate a self-contained HTML report with interactive charts.

    Args:
        output_path: Path to save the HTML report.
        benchmark_results: List of benchmark results.
        model_names: Optional names for each result.
        accuracy_results: Optional accuracy comparison results.
        platform_info: Platform information.
        title: Report title.

    Returns:
        Absolute path to the generated report.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(benchmark_results))]

    # Filter out None accuracy results and get valid indices
    valid_accuracy = []
    valid_accuracy_names = []
    if accuracy_results is not None:
        for name, acc in zip(model_names, accuracy_results):
            if acc is not None:
                valid_accuracy.append(acc)
                valid_accuracy_names.append(name)

    has_accuracy = len(valid_accuracy) > 0

    # Create figure with subplots
    num_rows = 2 + (1 if has_accuracy else 0)
    subplot_titles = [
        "Latency Distribution",
        "Latency Percentiles Comparison",
        "Throughput Comparison",
        "System Metrics",
    ]
    if has_accuracy:
        subplot_titles.extend(["Action MSE Comparison", "Action Cosine Similarity"])

    fig = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Latency distribution histogram
    for i, (name, result) in enumerate(zip(model_names, benchmark_results)):
        fig.add_trace(
            go.Histogram(
                x=result.latency.latencies_ms,
                name=name,
                opacity=0.6,
                nbinsx=50,
                marker_color=colors[i % len(colors)],
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # 2. Latency percentiles bar chart
    percentile_labels = ["P50", "P90", "P95", "P99"]
    for i, (name, result) in enumerate(zip(model_names, benchmark_results)):
        lat = result.latency
        values = [lat.p50_ms, lat.p90_ms, lat.p95_ms, lat.p99_ms]
        fig.add_trace(
            go.Bar(
                x=percentile_labels,
                y=values,
                name=name,
                marker_color=colors[i % len(colors)],
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Percentile", row=1, col=2)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)

    # 3. Throughput comparison
    throughputs = [r.latency.throughput_fps for r in benchmark_results]
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=throughputs,
            marker_color=[colors[i % len(colors)] for i in range(len(model_names))],
            text=[f"{t:.1f}" for t in throughputs],
            textposition="auto",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Throughput (FPS)", row=2, col=1)

    # 4. System metrics
    gpu_utils = [r.avg_gpu_utilization or 0 for r in benchmark_results]
    gpu_powers = [r.avg_gpu_power_w or 0 for r in benchmark_results]
    fig.add_trace(
        go.Bar(x=model_names, y=gpu_utils, name="GPU Util (%)", marker_color="#1f77b4"),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(x=model_names, y=gpu_powers, name="GPU Power (W)", marker_color="#ff7f0e", yaxis="y2"),
        row=2, col=2,
    )
    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="GPU Util (%)", row=2, col=2)

    # 5. Accuracy comparison
    if has_accuracy:
        mses = [a.action_mse for a in valid_accuracy]
        fig.add_trace(
            go.Bar(
                x=valid_accuracy_names,
                y=mses,
                marker_color=[colors[i % len(colors)] for i in range(len(valid_accuracy_names))],
                text=[f"{m:.6f}" for m in mses],
                textposition="auto",
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text="Model", row=3, col=1)
        fig.update_yaxes(title_text="Action MSE", row=3, col=1)

        cos_sims = [a.action_cosine_similarity for a in valid_accuracy]
        fig.add_trace(
            go.Bar(
                x=valid_accuracy_names,
                y=cos_sims,
                marker_color=[colors[i % len(colors)] for i in range(len(valid_accuracy_names))],
                text=[f"{c:.6f}" for c in cos_sims],
                textposition="auto",
            ),
            row=3,
            col=2,
        )
        fig.update_xaxes(title_text="Model", row=3, col=2)
        fig.update_yaxes(title_text="Cosine Similarity", row=3, col=2)

    fig.update_layout(
        title_text=title,
        height=400 * num_rows,
        showlegend=True,
        barmode="group",
    )

    # Generate HTML
    plotly_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Summary table
    summary_rows = []
    for name, result in zip(model_names, benchmark_results):
        lat = result.latency
        row = f"<tr><td>{name}</td><td>{lat.mean_ms:.3f}</td><td>{lat.p50_ms:.3f}</td>"
        row += f"<td>{lat.p95_ms:.3f}</td><td>{lat.p99_ms:.3f}</td>"
        row += f"<td>{lat.throughput_fps:.1f}</td></tr>"
        summary_rows.append(row)

    platform_html = ""
    if platform_info is not None:
        platform_html = f"""
        <div class="platform-info">
            <h2>Platform Information</h2>
            <table>
                <tr><th>OS</th><td>{platform_info.os}</td></tr>
                <tr><th>Architecture</th><td>{platform_info.arch}</td></tr>
                <tr><th>Python</th><td>{platform_info.python_version}</td></tr>
                <tr><th>CPU Cores</th><td>{platform_info.cpu_count}</td></tr>
                <tr><th>Memory</th><td>{platform_info.total_memory_gb} GB</td></tr>
                {"<tr><th>GPU</th><td>" + str(platform_info.gpu_name) + "</td></tr>" if platform_info.has_nvidia_gpu else ""}
            </table>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #1f77b4; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .platform-info table {{ max-width: 600px; }}
        .footer {{ margin-top: 40px; text-align: center; color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        {platform_html}

        <h2>Summary</h2>
        <table>
            <tr><th>Model</th><th>Mean (ms)</th><th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th><th>Throughput (FPS)</th></tr>
            {''.join(summary_rows)}
        </table>

        <h2>Interactive Charts</h2>
        {plotly_html}

        <div class="footer">
            <p>Report generated by rl-deploy-bench</p>
        </div>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)
