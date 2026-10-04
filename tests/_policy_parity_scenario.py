"""Seeded online-controller scenario for the Phase 3.1 LinUCB parity test.

``tests/golden/policy_online_parity.golden`` holds this scenario's output captured from the
pre-extraction code. Regenerate it only for an intended behaviour change, e.g.
``python -c "from tests._policy_parity_scenario import write_golden; write_golden()"``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from evals import _json as json
from evals.adaptive_online_policy import AdaptiveOnlinePolicyController, OnlinePolicyConfig

GOLDEN = Path(__file__).parent / "golden" / "policy_online_parity.golden"


def run_online_scenario(tmp: Path) -> dict[str, Any]:
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.8,
        epsilon=0.15,
        l2=1.0,
        warmup_decisions=3,
        save_every=1000,
        decay=0.95,
        feature_scaling_profile="v1",
        state_path=tmp / "state.json",
        events_path=tmp / "events.jsonl",
        random_seed=7,
    )
    controller = AdaptiveOnlinePolicyController(config)
    rng = np.random.default_rng(3)
    trace = []
    for step in range(40):
        features = {"bias": 1.0, "attempt_frac": float(rng.random()), "x1": float(rng.normal())}
        if step >= 15:
            features["late"] = float(rng.random())
        decision = controller.decide(feature_values=features, rng_key=f"k{step}")
        reward = float(rng.normal()) + (0.5 if decision.applied_action == "invoke_helper" else 0.0)
        controller.record_outcome(
            decision=decision,
            run_id="r",
            case_id=f"c{step}",
            condition="layered_adaptive_llm",
            attempt=1,
            note="n",
            reward_scope="helper_credit_v2",
            reward=reward,
            reward_success_hint=0.0,
            reward_progress_hint=0.0,
            reward_progress_score=0.0,
            reward_token_penalty=0.0,
            reward_latency_penalty=0.0,
            reward_stagnation_penalty=0.0,
            reward_helper_materialization_bonus=0.0,
            success=False,
            local_llm_call_count=0,
            local_tokens_prompt=0,
            local_tokens_completion=0,
            local_tokens_total=0,
            local_wall_sec=0.0,
            override_reason="",
        )
        trace.append(
            {
                "selected": decision.selected_action,
                "applied": decision.applied_action,
                "scores": {k: round(float(v), 10) for k, v in (decision.scores or {}).items()},
            }
        )
    state = controller.state
    controller.close()
    from evals.bootstrap_policy_state import main as bootstrap_main

    replay_path = tmp / "replay.json"
    assert (
        bootstrap_main(["--events", str(tmp / "events.jsonl"), "--output-state", str(replay_path)])
        == 0
    )
    replay = json.loads(replay_path.read_text())
    replay.pop("saved_at_utc", None)
    replay.pop("bootstrapped_at_utc", None)
    return {
        "replay": {k: replay[k] for k in ("feature_names", "updates", "A", "b") if k in replay},
        "trace": trace,
        "feature_names": list(state.feature_names),
        "updates": state.updates,
        "A": {a: np.round(m, 10).tolist() for a, m in state.A.items()},
        "b": {a: np.round(v, 10).tolist() for a, v in state.b.items()},
    }


def write_golden() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        GOLDEN.write_text(
            json.dumps(run_online_scenario(Path(tmp)), sort_keys=True), encoding="utf-8"
        )
