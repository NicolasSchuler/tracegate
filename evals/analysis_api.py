"""Public analysis API for the reports, the re-analysis audit and the baseline comparison.

These helpers are internals of the ``evals.analysis_*`` modules behind the
``analyze_results`` CLI. Other modules import them from here under public names,
so the analysis modules can change without touching their consumers (X-02).
"""

from __future__ import annotations

from evals.analysis_numeric import _expand_paths as expand_paths
from evals.analysis_numeric import _fmt_num_with_ci as fmt_num_with_ci
from evals.analysis_ope import _adaptive_ope_samples as adaptive_ope_samples
from evals.analysis_ope_support import (
    ADAPTIVE_BUDGET_ACTIONS,
    ADAPTIVE_ONLINE_ACTIONS,
    ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
)
from evals.analysis_output import _ope_target_order as ope_target_order
from evals.analysis_output import _print_input_errors as print_input_errors
from evals.analysis_paired_stats import _run_statistical_analysis as run_statistical_analysis
from evals.analysis_records import _load_and_group_records as load_and_group_records
from evals.analysis_rows import _case_success_rates as case_success_rates
from evals.analysis_rows import _derive_case_condition_rows as derive_case_condition_rows
from evals.analysis_rows import _matched_uplift as matched_uplift
from evals.analysis_rows import _records_by_case_condition as records_by_case_condition
from evals.analysis_summary import _summarize_bucket as summarize_bucket
from evals.analysis_usage import _is_external_baseline_condition as is_external_baseline_condition

__all__ = [
    "ADAPTIVE_BUDGET_ACTIONS",
    "ADAPTIVE_ONLINE_ACTIONS",
    "ADAPTIVE_OPE_MIN_ACTION_SUPPORT",
    "ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT",
    "adaptive_ope_samples",
    "case_success_rates",
    "derive_case_condition_rows",
    "expand_paths",
    "fmt_num_with_ci",
    "is_external_baseline_condition",
    "load_and_group_records",
    "matched_uplift",
    "ope_target_order",
    "print_input_errors",
    "records_by_case_condition",
    "run_statistical_analysis",
    "summarize_bucket",
]
