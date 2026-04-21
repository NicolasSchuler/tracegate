"""Convert llmdebug eval cases into standalone RLM baseline inputs."""

from __future__ import annotations

from pathlib import Path

from baselines.adapter_base import BaselineCaseContext
from baselines.case_schemas import LambdaRLMCaseInput
from baselines.lambda_rlm.case_converter import convert_case as _convert_lambda_case

# RLM uses the same case schema as lambda-RLM.
RLMCaseInput = LambdaRLMCaseInput


def convert_case(
    case_dir: Path,
    *,
    case_context: BaselineCaseContext | None = None,
) -> LambdaRLMCaseInput:
    """Reuse the shared llmdebug case packaging used by the lambda baseline."""
    return _convert_lambda_case(case_dir, case_context=case_context)
