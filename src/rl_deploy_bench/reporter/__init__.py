"""Report generation utilities."""

from .html import generate_html_report
from .markdown import generate_latency_distribution_data, generate_markdown_report

__all__ = [
    "generate_html_report",
    "generate_markdown_report",
    "generate_latency_distribution_data",
]
