"""
Evaluation and comparison utilities for model assessment.

Provides utilities for:
- WER and other metric calculation
- Model comparison (baseline vs fine-tuned)
- Performance analysis and reporting
"""

from .metrics import (
    calculate_wer,
    calculate_wer_from_files,
    calculate_der,
    PerformanceTimer,
    SystemMetrics,
    generate_comparison_report,
)

__all__ = [
    'calculate_wer',
    'calculate_wer_from_files',
    'calculate_der',
    'PerformanceTimer',
    'SystemMetrics',
    'generate_comparison_report',
]
