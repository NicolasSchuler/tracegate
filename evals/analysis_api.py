"""Public analysis API for the reports, the re-analysis audit and the baseline comparison.

These helpers live in ``evals.analyze_results`` as internals of its CLI. Other
modules import them from here under public names, so the analysis module can be
split without touching its consumers (X-02).
"""

from __future__ import annotations

from evals.analyze_results import (
    ADAPTIVE_BUDGET_ACTIONS,
    ADAPTIVE_ONLINE_ACTIONS,
    ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
)
from evals.analyze_results import _adaptive_ope_samples as adaptive_ope_samples
from evals.analyze_results import _case_success_rates as case_success_rates
from evals.analyze_results import _derive_case_condition_rows as derive_case_condition_rows
from evals.analyze_results import _expand_paths as expand_paths
from evals.analyze_results import _fmt_num_with_ci as fmt_num_with_ci
from evals.analyze_results import _is_external_baseline_condition as is_external_baseline_condition
from evals.analyze_results import _load_and_group_records as load_and_group_records
from evals.analyze_results import _matched_uplift as matched_uplift
from evals.analyze_results import _ope_target_order as ope_target_order
from evals.analyze_results import _print_input_errors as print_input_errors
from evals.analyze_results import _records_by_case_condition as records_by_case_condition
from evals.analyze_results import _run_statistical_analysis as run_statistical_analysis
from evals.analyze_results import _summarize_bucket as summarize_bucket

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
