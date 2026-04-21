"""Minimal vllm stub for smoke-only LDB env setup."""


class LLM:  # pragma: no cover - import-time compatibility shim
    def __init__(self, *args, **kwargs):
        raise RuntimeError("vllm smoke stub does not provide runtime generation")


class SamplingParams:  # pragma: no cover - import-time compatibility shim
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
