"""Generated adaptive-policy defaults tables for the policy docs (POL-25, X-08).

The policy docs listed 8 actions, reward hints of 0.25/0.5/0.25/0.5, helper event schema 3,
an `alpha` of 0.6 or 0.8 and "linucb only" while the code had 10 actions, 1.0/1.0/0.5/1.0,
schema 4, 0.6 (CLI) and 0.8 (dataclass), and Thompson sampling. The block below is rendered
from the constants the CLI and the controllers use, and ``evals.doc_sync --check`` fails when
a doc's copy is stale.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from . import eval_types as et
from .adaptive_budget_online_policy import (
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ALGOS,
    BUDGET_POLICY_ARCHITECTURES,
    BUDGET_POLICY_FAMILIES,
    BUDGET_POLICY_MODES,
    BudgetPolicyConfig,
)
from .adaptive_budget_online_policy import EVENT_SCHEMA_VERSION as BUDGET_EVENT_SCHEMA_VERSION
from .adaptive_budget_online_policy import STATE_SCHEMA_VERSION as BUDGET_STATE_SCHEMA_VERSION
from .adaptive_online_policy import (
    EVENT_SCHEMA_VERSION,
    ONLINE_ACTION_SCORE_BIAS_LIMIT,
    ONLINE_POLICY_ACTIONS,
    ONLINE_POLICY_ALGOS,
    ONLINE_POLICY_MODES,
    STATE_SCHEMA_VERSION,
    OnlinePolicyConfig,
)
from .policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    POLICY_FEATURE_SCALING_PROFILES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DEFAULTS_DOCS = (
    REPO_ROOT / "docs" / "ADAPTIVE_POLICY_QUICK_REFERENCE.md",
    REPO_ROOT / "docs" / "adaptive_online_policy_design.md",
)
BEGIN_POLICY_DEFAULTS_MARKER = "<!-- BEGIN AUTO-GENERATED: POLICY_DEFAULTS -->"
END_POLICY_DEFAULTS_MARKER = "<!-- END AUTO-GENERATED: POLICY_DEFAULTS -->"


def _names(values: Iterable[object]) -> str:
    return ", ".join(f"`{name}`" for name in sorted(str(item) for item in values))


def _ordered(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{name}`" for name in values)


def render_policy_defaults_block() -> str:
    """The defaults block, delimited by its markers."""
    online = OnlinePolicyConfig()
    budget = BudgetPolicyConfig()
    rows = [
        ("Algorithms", _names(ONLINE_POLICY_ALGOS), _names(BUDGET_POLICY_ALGOS)),
        ("Modes", _names(ONLINE_POLICY_MODES), _names(BUDGET_POLICY_MODES)),
        (
            "Actions",
            f"{len(ONLINE_POLICY_ACTIONS)}: {_ordered(ONLINE_POLICY_ACTIONS)}",
            f"{_ordered(BUDGET_POLICY_ACTIONS)}; families {_ordered(BUDGET_POLICY_FAMILIES)}",
        ),
        ("Architectures", "-", _names(BUDGET_POLICY_ARCHITECTURES)),
        (
            "`alpha` (CLI / Python API)",
            f"{et.DEFAULT_ADAPTIVE_ONLINE_ALPHA} / {online.alpha}",
            f"{et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA} / {budget.alpha}",
        ),
        (
            "`epsilon` (CLI / Python API)",
            f"{et.DEFAULT_ADAPTIVE_ONLINE_EPSILON} / {online.epsilon}",
            f"{et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON} / {budget.epsilon}",
        ),
        (
            "Warm-up decisions (CLI / Python API)",
            f"{et.DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS} / {online.warmup_decisions}",
            f"{et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS} / {budget.warmup_decisions}",
        ),
        (
            "Recency decay",
            f"{et.DEFAULT_ADAPTIVE_ONLINE_DECAY} (per arm when < 1)",
            f"{et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY} (per arm when < 1)",
        ),
        ("`l2`", str(et.DEFAULT_ADAPTIVE_POLICY_L2), str(et.DEFAULT_ADAPTIVE_POLICY_L2)),
        (
            "Save every (updates)",
            str(et.DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY),
            str(et.DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY),
        ),
        (
            "Feature scaling",
            f"`{DEFAULT_POLICY_FEATURE_SCALING_PROFILE}` "
            f"(of {_names(POLICY_FEATURE_SCALING_PROFILES)})",
            f"`{DEFAULT_POLICY_FEATURE_SCALING_PROFILE}`",
        ),
        (
            "Score biases",
            f"`{et.DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES}`, each within "
            f"±{ONLINE_ACTION_SCORE_BIAS_LIMIT}",
            "-",
        ),
        (
            "Budget ladder",
            "-",
            f"`{et.DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS}`, max "
            f"{et.DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE} up(s) per case",
        ),
        (
            "Event / state schema",
            f"{EVENT_SCHEMA_VERSION} / {STATE_SCHEMA_VERSION}",
            f"{BUDGET_EVENT_SCHEMA_VERSION} / {BUDGET_STATE_SCHEMA_VERSION}",
        ),
    ]
    reward_rows = [
        ("`lambda_tokens` (token penalty weight)", et.ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT),
        ("`token_scale`", et.ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT),
        ("`mu_latency` (latency penalty weight)", et.ADAPTIVE_REWARD_MU_LATENCY_DEFAULT),
        ("`latency_scale` (seconds)", et.ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT),
        ("`eta_progress`", et.ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT),
        ("`kappa_stagnation`", et.ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT),
        ("Success hint scale", et.ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT),
        ("Progress hint scale", et.ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT),
        ("Stagnation penalty scale", et.ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT),
        ("Helper materialization scale", et.ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT),
        ("Budget `rho_budget_exceeded`", et.ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT),
    ]
    lines = [
        BEGIN_POLICY_DEFAULTS_MARKER,
        "<!-- Rendered from the code by `python -m evals.doc_sync --write`; do not edit. -->",
        "",
        "| Setting | Helper policy | Budget policy |",
        "|---------|---------------|---------------|",
        *(f"| {name} | {helper} | {budget_value} |" for name, helper, budget_value in rows),
        "",
        "| Reward coefficient (CLI default) | Value |",
        "|----------------------------------|-------|",
        *(f"| {name} | {value} |" for name, value in reward_rows),
        "",
        "Runs take the CLI defaults; the Python API defaults apply only to configs built in "
        "code. Percentile caps freeze after "
        f"{DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES} observations per feature. Run metadata "
        f"records `policy_defaults_version: {et.POLICY_DEFAULTS_VERSION}`, and so do "
        "`patcher_config` and the run settings fingerprint derived from it, so resuming a "
        "run recorded under an earlier generation is refused.",
        END_POLICY_DEFAULTS_MARKER,
    ]
    return "\n".join(lines)


def _block_pattern() -> re.Pattern[str]:
    return re.compile(
        f"{re.escape(BEGIN_POLICY_DEFAULTS_MARKER)}.*?{re.escape(END_POLICY_DEFAULTS_MARKER)}",
        re.DOTALL,
    )


def stale_policy_defaults_docs(docs: tuple[Path, ...] = POLICY_DEFAULTS_DOCS) -> list[str]:
    """Docs whose block is missing or out of date (relative paths)."""
    expected = render_policy_defaults_block()
    stale: list[str] = []
    for path in docs:
        match = _block_pattern().search(path.read_text(encoding="utf-8"))
        if match is None or match.group(0) != expected:
            stale.append(str(path.relative_to(REPO_ROOT)))
    return stale


def write_policy_defaults_blocks(docs: tuple[Path, ...] = POLICY_DEFAULTS_DOCS) -> list[str]:
    """Rewrite each doc's block; returns the docs that changed. Docs need the markers."""
    expected = render_policy_defaults_block()
    changed: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        if _block_pattern().search(text) is None:
            raise ValueError(
                f"{path}: missing {BEGIN_POLICY_DEFAULTS_MARKER} ... {END_POLICY_DEFAULTS_MARKER}"
            )
        updated = _block_pattern().sub(lambda _match: expected, text, count=1)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(str(path.relative_to(REPO_ROOT)))
    return changed
