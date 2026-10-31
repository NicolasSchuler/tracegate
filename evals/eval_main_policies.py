"""Policy-controller lifecycle of an eval run: the online helper policy and the budget policies.

Moved out of ``run_main_flow`` (X-01).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evals.adaptive_budget_online_policy import AdaptiveBudgetPolicyController, BudgetPolicyConfig
from evals.adaptive_online_policy import AdaptiveOnlinePolicyController, OnlinePolicyConfig
from evals.decision_model.config import decision_model_settings_from_args
from evals.decision_model.runtime import build_decision_model_binding
from evals.eval_config import _parse_adaptive_online_action_score_biases
from evals.eval_types import Condition
from evals.seed_derivation import derive_seed

_BUDGET_POLICY_CONDITIONS = (Condition.LAYERED_ADAPTIVE_GATED, Condition.LAYERED_ADAPTIVE_LLM)


def _per_condition_budget_configs(
    config: BudgetPolicyConfig,
    *,
    conditions: list[Condition],
    seed: int | None,
    horizon_decisions: int = 0,
) -> dict[str, BudgetPolicyConfig]:
    """Budget-policy configs keyed by condition value (POL-05, POL-23).

    With more than one adaptive condition each gets its own state and event files
    (``<stem>.<condition><suffix>``); every controller gets a seed derived from the run
    seed and its condition. Runs without an adaptive condition keep one shared config.
    *horizon_decisions* (cases x attempts per condition) sets the ``bwk_v1`` pacing
    horizon (POL-04).
    """
    keys = [condition.value for condition in conditions if condition in _BUDGET_POLICY_CONDITIONS]
    separate_files = len(keys) > 1
    out: dict[str, BudgetPolicyConfig] = {}
    for key in keys or [""]:
        updates: dict[str, Any] = {
            "random_seed": derive_seed(int(seed), "budget", key) if seed is not None else None,
            "bwk_horizon_decisions": max(0, int(horizon_decisions)),
        }
        if separate_files:
            for field_name in ("state_path", "events_path"):
                path = getattr(config, field_name)
                if path is not None:
                    updates[field_name] = path.with_name(f"{path.stem}.{key}{path.suffix}")
        out[key] = replace(config, **updates)
    return out


@dataclass(frozen=True)
class PolicyControllerInputs:
    """Read-only inputs of :func:`build_policy_controllers`."""

    _close_run_results_io: Any
    args: Any
    artifacts_root: Any
    cases: Any
    conditions: Any
    display: Any
    patcher_config: Any


@dataclass
class PolicyControllers:
    """What :func:`build_policy_controllers` hands back to its caller."""

    budget_policy_controller: Any
    budget_policy_controllers: Any
    online_policy_controller: Any


def build_policy_controllers(inputs: PolicyControllerInputs) -> PolicyControllers:
    """Create the run's online helper policy and per-condition budget policies.

    Also records the warm-start source and result in *patcher_config* and reports
    load errors on *display*. Moved verbatim from ``run_main_flow`` (X-01).
    """
    _close_run_results_io = inputs._close_run_results_io
    args = inputs.args
    artifacts_root = inputs.artifacts_root
    cases = inputs.cases
    conditions = inputs.conditions
    display = inputs.display
    patcher_config = inputs.patcher_config
    try:
        online_init_state_path = str(getattr(args, "adaptive_online_init_state_path", "") or "")
        online_action_score_biases = _parse_adaptive_online_action_score_biases(
            getattr(args, "adaptive_online_action_score_biases", "")
        )
        online_policy_config = OnlinePolicyConfig.normalize(
            mode=str(args.adaptive_online_policy_mode),
            algorithm=str(args.adaptive_online_policy_algo),
            alpha=float(args.adaptive_online_alpha),
            epsilon=float(args.adaptive_online_epsilon),
            l2=float(args.adaptive_online_l2),
            warmup_decisions=int(args.adaptive_online_warmup_decisions),
            save_every=int(args.adaptive_online_save_every),
            decay=float(args.adaptive_online_decay),
            feature_scaling_profile=str(args.adaptive_online_feature_scaling_profile),
            init_state_path=(
                Path(online_init_state_path).expanduser().resolve()
                if online_init_state_path.strip()
                else None
            ),
            state_path=Path(str(args.adaptive_online_state_path)).expanduser().resolve(),
            events_path=Path(str(args.adaptive_online_events_path)).expanduser().resolve(),
            # POL-23: each controller gets its own stream, derived from the run seed.
            random_seed=(derive_seed(int(args.seed), "helper") if args.seed is not None else None),
            action_score_biases=online_action_score_biases,
        )
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    if online_policy_config.mode == "off":
        online_policy_controller: AdaptiveOnlinePolicyController | None = None
    else:
        try:
            decision_model_binding = build_decision_model_binding(
                decision_model_settings_from_args(args),
                artifacts_root=artifacts_root,
            )
            # Pass the binding only when a role is active (keeps the one-argument form intact).
            online_policy_controller = (
                AdaptiveOnlinePolicyController(
                    online_policy_config,
                    decision_model=decision_model_binding,
                )
                if decision_model_binding is not None
                else AdaptiveOnlinePolicyController(online_policy_config)
            )
        except BaseException:
            _close_run_results_io()
            display.close()
            raise
        if online_policy_controller.load_error:
            display.add_event(
                message=(
                    "Online adaptive policy state load warning: "
                    f"{online_policy_controller.load_error}; starting with a fresh state."
                ),
                style="warning",
            )
    if isinstance(patcher_config, dict):
        patcher_config["adaptive_online_warm_start_source"] = online_init_state_path
        patcher_config["adaptive_online_warm_start_result"] = (
            "load_error"
            if online_policy_controller is not None and online_policy_controller.load_error
            else ("loaded" if online_init_state_path else "fresh")
        )

    try:
        budget_init_state_path = str(getattr(args, "adaptive_budget_init_state_path", "") or "")
        budget_policy_config = BudgetPolicyConfig.normalize(
            mode=str(args.adaptive_budget_policy_mode),
            algorithm=str(args.adaptive_budget_policy_algo),
            architecture=str(args.adaptive_budget_policy_architecture),
            alpha=float(args.adaptive_budget_online_alpha),
            epsilon=float(args.adaptive_budget_online_epsilon),
            l2=float(args.adaptive_budget_online_l2),
            warmup_decisions=int(args.adaptive_budget_online_warmup_decisions),
            save_every=int(args.adaptive_budget_online_save_every),
            decay=float(args.adaptive_budget_online_decay),
            feature_scaling_profile=str(args.adaptive_budget_feature_scaling_profile),
            init_state_path=(
                Path(budget_init_state_path).expanduser().resolve()
                if budget_init_state_path.strip()
                else None
            ),
            state_path=Path(str(args.adaptive_budget_online_state_path)).expanduser().resolve(),
            events_path=Path(str(args.adaptive_budget_online_events_path)).expanduser().resolve(),
            bwk_run_token_budget=int(getattr(args, "adaptive_budget_bwk_run_token_budget", 0)),
            random_seed=None,
        )
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    # POL-05: one budget controller per adaptive condition, so the conditions being
    # compared don't train (and draw exploration from) one shared model.
    budget_policy_controllers: dict[str, AdaptiveBudgetPolicyController] = {}
    if budget_policy_config.mode != "off":
        for condition_key, condition_config in _per_condition_budget_configs(
            budget_policy_config,
            conditions=conditions,
            seed=args.seed,
            # POL-04: bwk_v1 paces its token budget over at most one decision per attempt.
            horizon_decisions=len(cases) * max(1, int(getattr(args, "k", 1) or 1)),
        ).items():
            try:
                controller = AdaptiveBudgetPolicyController(condition_config)
            except BaseException:
                for created in budget_policy_controllers.values():
                    created.close()
                _close_run_results_io()
                display.close()
                raise
            if controller.load_error:
                display.add_event(
                    message=(
                        "Online adaptive budget policy state load warning: "
                        f"{controller.load_error}; starting with a fresh state."
                    ),
                    style="warning",
                )
            budget_policy_controllers[condition_key] = controller
    budget_policy_controller: AdaptiveBudgetPolicyController | None = next(
        iter(budget_policy_controllers.values()), None
    )
    if isinstance(patcher_config, dict) and budget_policy_controllers:
        patcher_config["adaptive_budget_controllers"] = {
            (key or "shared"): {
                "state_path": str(controller.config.state_path),
                "events_path": str(controller.config.events_path),
            }
            for key, controller in budget_policy_controllers.items()
        }
    if isinstance(patcher_config, dict):
        patcher_config["adaptive_budget_warm_start_source"] = budget_init_state_path
        patcher_config["adaptive_budget_warm_start_result"] = (
            "load_error"
            if budget_policy_controller is not None and budget_policy_controller.load_error
            else ("loaded" if budget_init_state_path else "fresh")
        )
    return PolicyControllers(
        budget_policy_controller=budget_policy_controller,
        budget_policy_controllers=budget_policy_controllers,
        online_policy_controller=online_policy_controller,
    )
