"""Tests for the decision-model sidecar's request handling.

The sidecar lives in the Docker build context (``evals/docker/decision_model/server.py``), so
it is loaded by path rather than imported as a module. Only the pure request-handling helpers
are exercised here: the model loader is the part that needs torch and a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SERVER_PATH = (
    Path(__file__).resolve().parents[1] / "evals" / "docker" / "decision_model" / "server.py"
)
_MODULE_NAME = "decision_model_server"


def _load_server_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their annotations through sys.modules, so register before executing.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


server = _load_server_module()


def _answer_ok(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": request["model"],
        "answers": {"q": {"type": "noul", "noul": 0.75}},
        "usage": {"input_tokens": 123, "output_tokens": 0},
    }


def _body(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_server_module_imports_without_a_gpu_stack() -> None:
    # The heavy imports must stay inside DecisionModelRuntime.load().
    for name in ("torch", "transformers", "huggingface_hub", "joint_schema_model"):
        assert not hasattr(server, name), name
    assert server.SYSTEMONE_PATH == "/v1/systemone"
    assert server.HEALTH_PATH == "/health"


def test_config_from_env_uses_defaults_for_blank_and_invalid_values() -> None:
    config = server.config_from_env({"CLEF_MODEL_ID": "  ", "CLEF_MAX_LENGTH": "not-a-number"})

    assert config.model_id == "Cloudflare/clef-flash"
    assert config.revision == "main"
    assert config.device == "cuda"
    assert config.dtype == "bfloat16"
    assert config.max_length == 8192
    assert config.host == "0.0.0.0"
    assert config.port == 8090
    assert config.max_request_bytes == 512 * 1024


def test_config_from_env_reads_every_clef_variable() -> None:
    config = server.config_from_env(
        {
            "CLEF_MODEL_ID": "Cloudflare/clef",
            "CLEF_REVISION": "deadbeef",
            "CLEF_DEVICE": "cuda:1",
            "CLEF_DTYPE": "float16",
            "CLEF_MAX_LENGTH": "4096",
            "CLEF_HOST": "127.0.0.1",
            "CLEF_PORT": "9000",
            "CLEF_MAX_REQUEST_BYTES": "2048",
        }
    )

    assert config.model_id == "Cloudflare/clef"
    assert config.revision == "deadbeef"
    assert config.device == "cuda:1"
    assert config.dtype == "float16"
    assert config.max_length == 4096
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.max_request_bytes == 2048


@pytest.mark.parametrize("value", ["0", "-5"])
def test_config_from_env_rejects_non_positive_sizes(value: str) -> None:
    config = server.config_from_env({"CLEF_MAX_LENGTH": value, "CLEF_PORT": value})

    assert config.max_length == 8192
    assert config.port == 8090


@pytest.mark.parametrize(
    ("revision", "pinned"),
    [("main", False), ("master", False), ("", False), ("  MAIN ", False), ("deadbeef", True)],
)
def test_revision_is_pinned_flags_moving_branches(revision: str, pinned: bool) -> None:
    assert server.revision_is_pinned(revision) is pinned


def test_health_request_reports_configuration_and_ready_state() -> None:
    config = server.ServerConfig(revision="abc123", device="cuda:0", max_length=4096)

    status, payload = server.handle_health_request(config, ready=True)
    assert status == 200
    assert payload == {
        "status": "ok",
        "model": "Cloudflare/clef-flash",
        "revision": "abc123",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "quantization": "none",
        "max_length": 4096,
    }

    loading_status, loading_payload = server.handle_health_request(config, ready=False)
    assert loading_status == 503
    assert loading_payload["status"] == "loading"


def test_systemone_request_returns_answers_and_server_side_latency() -> None:
    request = {"model": "clef-flash", "state": {"0_note": "x"}, "questions": {"q": {}}}

    status, payload = server.handle_systemone_request(
        _body(request),
        _answer_ok,
        max_bytes=1024,
    )

    assert status == 200
    assert payload["model"] == "clef-flash"
    assert payload["answers"] == {"q": {"type": "noul", "noul": 0.75}}
    assert payload["usage"]["input_tokens"] == 123
    assert payload["usage"]["latency_ms"] >= 0.0


def test_systemone_request_adds_usage_when_the_model_omits_it() -> None:
    def answer(_request: dict[str, Any]) -> dict[str, Any]:
        return {"model": "clef-flash", "answers": {}}

    status, payload = server.handle_systemone_request(
        _body({"model": "clef-flash"}),
        answer,
        max_bytes=1024,
    )

    assert status == 200
    assert set(payload["usage"]) == {"latency_ms"}


def test_systemone_request_rejects_oversized_body_with_413() -> None:
    calls: list[dict[str, Any]] = []

    def answer(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        return {}

    status, payload = server.handle_systemone_request(b"x" * 64, answer, max_bytes=8)

    assert status == 413
    assert "exceeds 8 bytes" in payload["error"]
    assert not calls


@pytest.mark.parametrize("body", [b"", b"{not json", b"[]", b'"text"'])
def test_systemone_request_rejects_malformed_bodies_with_400(body: bytes) -> None:
    status, payload = server.handle_systemone_request(body, _answer_ok, max_bytes=1024)

    assert status == 400
    assert payload["error"]


def test_systemone_request_rejects_undecodable_bytes_with_400() -> None:
    status, payload = server.handle_systemone_request(b"\xff\xfe", _answer_ok, max_bytes=1024)

    assert status == 400
    assert "invalid JSON body" in payload["error"]


def test_systemone_request_maps_model_validation_errors_to_400() -> None:
    def answer(_request: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("model and state are required")

    status, payload = server.handle_systemone_request(
        _body({"model": "clef-flash"}),
        answer,
        max_bytes=1024,
    )

    assert status == 400
    assert payload == {"error": "model and state are required"}


def test_systemone_request_maps_not_ready_to_503() -> None:
    def answer(_request: dict[str, Any]) -> dict[str, Any]:
        raise server.ModelNotReadyError("decision model is still loading")

    status, payload = server.handle_systemone_request(
        _body({"model": "clef-flash"}),
        answer,
        max_bytes=1024,
    )

    assert status == 503
    assert payload == {"error": "decision model is still loading"}


def test_systemone_request_maps_unexpected_failures_to_500() -> None:
    def answer(_request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("CUDA out of memory")

    status, payload = server.handle_systemone_request(
        _body({"model": "clef-flash"}),
        answer,
        max_bytes=1024,
    )

    assert status == 500
    assert payload["error"] == "RuntimeError: CUDA out of memory"


def test_systemone_request_maps_non_object_responses_to_500() -> None:
    def answer(_request: dict[str, Any]) -> Any:
        return ["not", "an", "object"]

    status, payload = server.handle_systemone_request(
        _body({"model": "clef-flash"}),
        answer,
        max_bytes=1024,
    )

    assert status == 500
    assert "expected a response object" in payload["error"]


def test_runtime_rejects_requests_before_the_model_is_loaded() -> None:
    runtime = server.DecisionModelRuntime(server.ServerConfig())

    assert runtime.ready is False
    with pytest.raises(server.ModelNotReadyError):
        runtime.answer({"model": "clef-flash"})


def test_runtime_serializes_forward_passes_through_one_lock() -> None:
    runtime = server.DecisionModelRuntime(server.ServerConfig(max_length=4096))
    seen: list[dict[str, Any]] = []

    def fake_systemone(
        model: Any,
        processor: Any,
        request: dict[str, Any],
        max_length: int,
    ) -> dict[str, Any]:
        seen.append({"model": model, "processor": processor, "max_length": max_length})
        assert runtime._lock.locked()
        return {"model": request["model"], "answers": {}}

    runtime._model = "model"
    runtime._processor = "processor"
    runtime._systemone = fake_systemone

    assert runtime.ready is True
    assert runtime.answer({"model": "clef-flash"}) == {"model": "clef-flash", "answers": {}}
    assert seen == [{"model": "model", "processor": "processor", "max_length": 4096}]
    assert not runtime._lock.locked()


def test_missing_linear_attention_kernels_reports_absent_packages() -> None:
    # Neither kernel package is installed in the repo's dev environment.
    assert server.missing_linear_attention_kernels() == ("fla", "causal_conv1d")


class _FakeRuntime:
    """Stand-in for :class:`DecisionModelRuntime` that needs no GPU."""

    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.requests: list[dict[str, Any]] = []

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return {"model": request.get("model", ""), "answers": {"q": {"type": "noul", "noul": 0.5}}}


@pytest.fixture
def live_sidecar() -> Iterator[tuple[str, _FakeRuntime]]:
    """Run the real handler on a loopback port against a fake runtime."""
    runtime = _FakeRuntime()
    config = server.ServerConfig(host="127.0.0.1", port=0, max_request_bytes=512)
    http_server = server.build_server(config, runtime)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    host, port = http_server.server_address[:2]
    try:
        yield f"http://{host}:{port}", runtime
    finally:
        http_server.shutdown()
        http_server.server_close()
        thread.join(timeout=5)


def _request(url: str, *, payload: bytes | None = None) -> tuple[int, dict[str, Any]]:
    method = "GET" if payload is None else "POST"
    # Fixed http:// loopback URL built by the fixture.
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.mark.integration
def test_live_handler_serves_health_and_systemone(
    live_sidecar: tuple[str, _FakeRuntime],
) -> None:
    base_url, runtime = live_sidecar

    status, health = _request(f"{base_url}/health")
    assert status == 200
    assert health["status"] == "ok"
    assert health["model"] == "Cloudflare/clef-flash"

    status, answered = _request(
        f"{base_url}/v1/systemone",
        payload=_body({"model": "clef-flash", "state": {"0_note": "x"}, "questions": {}}),
    )
    assert status == 200
    assert answered["answers"] == {"q": {"type": "noul", "noul": 0.5}}
    assert answered["usage"]["latency_ms"] >= 0.0
    assert [request["model"] for request in runtime.requests] == ["clef-flash"]


@pytest.mark.integration
def test_live_handler_rejects_unknown_paths_and_oversized_bodies(
    live_sidecar: tuple[str, _FakeRuntime],
) -> None:
    base_url, runtime = live_sidecar

    status, payload = _request(f"{base_url}/v1/unknown")
    assert status == 404
    assert "unknown path" in payload["error"]

    status, payload = _request(f"{base_url}/v1/unknown", payload=b"{}")
    assert status == 404

    status, payload = _request(
        f"{base_url}/v1/systemone", payload=b'{"pad": "' + b"x" * 600 + b'"}'
    )
    assert status == 413
    assert not runtime.requests

    status, payload = _request(f"{base_url}/v1/systemone", payload=b"{oops")
    assert status == 400


@pytest.mark.integration
def test_live_handler_reports_503_while_the_model_loads() -> None:
    runtime = _FakeRuntime(ready=False)
    config = server.ServerConfig(host="127.0.0.1", port=0)
    http_server = server.build_server(config, runtime)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    host, port = http_server.server_address[:2]
    try:
        status, payload = _request(f"http://{host}:{port}/health")
    finally:
        http_server.shutdown()
        http_server.server_close()
        thread.join(timeout=5)

    assert status == 503
    assert payload["status"] == "loading"


def test_loader_failure_shuts_the_server_down_so_the_container_restarts() -> None:
    class _FailingRuntime:
        def load(self) -> None:
            raise RuntimeError("snapshot_download failed")

    class _RecordingServer:
        def __init__(self) -> None:
            self.shutdowns = 0

        def shutdown(self) -> None:
            self.shutdowns += 1

    failed = threading.Event()
    recording_server = _RecordingServer()

    server._load_then_serve_or_stop(_FailingRuntime(), recording_server, failed)

    assert failed.is_set()
    assert recording_server.shutdowns == 1


def test_warmup_request_is_a_valid_systemone_body() -> None:
    warmup = server._WARMUP_REQUEST

    assert isinstance(warmup["model"], str)
    assert "state" in warmup
    assert warmup["questions"]
    for question in warmup["questions"].values():
        assert question["type"] in {"noul", "choice", "score"}


def test_quantization_is_configurable_and_validated() -> None:
    assert server.config_from_env({"CLEF_QUANTIZATION": "NF4"}).quantization == "nf4"
    assert server.config_from_env({}).quantization == "none"
    assert server.quantization_kwargs("none", compute_dtype=None) == {}
    with pytest.raises(ValueError, match="CLEF_QUANTIZATION"):
        server.quantization_kwargs("q2", compute_dtype=None)
