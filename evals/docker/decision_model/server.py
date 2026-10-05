"""SystemOne-compatible HTTP sidecar for Cloudflare's Clef decision model.

Serves exactly the two endpoints ``evals/decision_model/client.py`` speaks:

* ``GET /health`` -> ``{status, model, revision, device, dtype, quantization, max_length}``
* ``POST /v1/systemone`` -> the model card's ``systemone()`` response body plus a server-side
  ``usage.latency_ms``

The module imports without torch, transformers or ``huggingface_hub``: those imports happen
inside :meth:`DecisionModelRuntime.load`, so the request-handling helpers stay unit testable on
a laptop (``tests/test_evals_decision_model_server.py``). Decisions are answered by a single
model on a single device, so every forward pass is serialized behind one lock.

Configuration (all environment variables, all optional):

=========================== =============================== ===================================
Variable                    Default                         Meaning
=========================== =============================== ===================================
``CLEF_MODEL_ID``           ``Cloudflare/clef-flash``       Hugging Face repo id
``CLEF_REVISION``           ``main``                        Pin to a commit sha; ``main`` warns
``CLEF_DEVICE``             ``cuda``                        torch device for weights and head
``CLEF_DTYPE``              ``bfloat16``                    ``torch`` dtype attribute name
``CLEF_QUANTIZATION``       ``none``                        ``none``, ``nf4`` (4-bit) or ``int8``
``CLEF_MAX_LENGTH``         ``8192``                        Prompt token budget per request
``CLEF_HOST``/``CLEF_PORT`` ``0.0.0.0`` / ``8090``          Listen address
``CLEF_MAX_REQUEST_BYTES``  ``524288``                      Larger bodies are rejected with 413
=========================== =============================== ===================================

The endpoint is unauthenticated; keep it on a trusted network.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HEALTH_PATH = "/health"
SYSTEMONE_PATH = "/v1/systemone"

DEFAULT_MODEL_ID = "Cloudflare/clef-flash"
DEFAULT_REVISION = "main"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
# Backbone weight quantization (bitsandbytes). The joint schema head and the output embedding
# matrix it reads (lm_head) always stay in CLEF_DTYPE.
QUANTIZATION_NONE = "none"
QUANTIZATION_NF4 = "nf4"
QUANTIZATION_INT8 = "int8"
QUANTIZATION_MODES = (QUANTIZATION_NONE, QUANTIZATION_NF4, QUANTIZATION_INT8)
# Modules bitsandbytes must not convert: lm_head is indexed directly by the head, and the
# vision tower is unused for text-only states.
_UNQUANTIZED_MODULES = ["lm_head", "visual"]
DEFAULT_MAX_LENGTH = 8192
# Container-internal listener; the run wrapper decides which host port it is published on.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090
DEFAULT_MAX_REQUEST_BYTES = 512 * 1024

STATUS_OK = "ok"
STATUS_LOADING = "loading"

# Optional CUDA kernels for the backbone's linear-attention layers. Without them transformers
# uses its slower reference implementation, which is worth a startup warning.
_LINEAR_ATTENTION_MODULES = ("fla", "causal_conv1d")
_UNPINNED_REVISIONS = frozenset({"", "main", "master"})

# One tiny joint-schema request so the first real decision does not pay for CUDA graph
# compilation and lazy kernel loading.
_WARMUP_REQUEST: dict[str, Any] = {
    "model": "warmup",
    "state": {"0_note": "Warm-up request issued by the sidecar at startup.", "1_failure": "none"},
    "questions": {
        "warmup": {
            "type": "noul",
            "instructions": "Is this a warm-up request?",
            "criteria": {"true": "It is a warm-up request.", "false": "It is not."},
        }
    },
}

LOGGER = logging.getLogger("decision_model")


@dataclass(frozen=True)
class ServerConfig:
    """Resolved sidecar configuration."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_REVISION
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    quantization: str = QUANTIZATION_NONE
    max_length: int = DEFAULT_MAX_LENGTH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES


class ModelNotReadyError(RuntimeError):
    """Raised while the weights are still being downloaded or loaded."""


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = str(environ.get(name, "") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def config_from_env(environ: Mapping[str, str] | None = None) -> ServerConfig:
    """Build the configuration from ``CLEF_*`` variables, falling back to the defaults."""
    env = os.environ if environ is None else environ

    def text(name: str, default: str) -> str:
        return str(env.get(name, "") or "").strip() or default

    return ServerConfig(
        model_id=text("CLEF_MODEL_ID", DEFAULT_MODEL_ID),
        revision=text("CLEF_REVISION", DEFAULT_REVISION),
        device=text("CLEF_DEVICE", DEFAULT_DEVICE),
        dtype=text("CLEF_DTYPE", DEFAULT_DTYPE),
        quantization=text("CLEF_QUANTIZATION", QUANTIZATION_NONE).lower(),
        max_length=_positive_int(env, "CLEF_MAX_LENGTH", DEFAULT_MAX_LENGTH),
        host=text("CLEF_HOST", DEFAULT_HOST),
        port=_positive_int(env, "CLEF_PORT", DEFAULT_PORT),
        max_request_bytes=_positive_int(env, "CLEF_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES),
    )


def missing_linear_attention_kernels() -> tuple[str, ...]:
    """Names of the optional linear-attention kernel packages that are not importable."""
    missing: list[str] = []
    for name in _LINEAR_ATTENTION_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return tuple(missing)


def quantization_kwargs(quantization: str, *, compute_dtype: Any) -> dict[str, Any]:
    """``from_pretrained`` kwargs for the backbone's weight quantization (empty for ``none``)."""
    mode = str(quantization).strip().lower()
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"CLEF_QUANTIZATION must be one of {QUANTIZATION_MODES}: {quantization}")
    if mode == QUANTIZATION_NONE:
        return {}
    # Imported lazily like torch: this module must import without the GPU stack.
    from transformers import BitsAndBytesConfig

    if mode == QUANTIZATION_NF4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(_UNQUANTIZED_MODULES),
        )
    else:
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=list(_UNQUANTIZED_MODULES),
        )
    return {"quantization_config": config}


def revision_is_pinned(revision: str) -> bool:
    """Whether *revision* identifies an immutable commit rather than a moving branch."""
    return str(revision).strip().lower() not in _UNPINNED_REVISIONS


def handle_health_request(config: ServerConfig, *, ready: bool) -> tuple[int, dict[str, Any]]:
    """Return ``(http_status, body)`` for ``GET /health``."""
    status = 200 if ready else 503
    return status, {
        "status": STATUS_OK if ready else STATUS_LOADING,
        "model": config.model_id,
        "revision": config.revision,
        "device": config.device,
        "dtype": config.dtype,
        "quantization": config.quantization,
        "max_length": int(config.max_length),
    }


def _decode_request(body: bytes) -> dict[str, Any]:
    if not body:
        raise ValueError("request body is empty")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _with_latency(response: Any, latency_ms: float) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError(f"expected a response object, got {type(response).__name__}")
    payload = dict(response)
    usage = payload.get("usage")
    merged = dict(usage) if isinstance(usage, dict) else {}
    merged["latency_ms"] = round(float(latency_ms), 3)
    payload["usage"] = merged
    return payload


def handle_systemone_request(
    body: bytes,
    answer: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_bytes: int,
) -> tuple[int, dict[str, Any]]:
    """Validate *body*, run *answer*, and return ``(http_status, body)``.

    Oversized bodies become 413, malformed requests (and the model card's own validation
    errors) become 400, a model that is still loading becomes 503, and anything unexpected
    becomes 500 so one bad request never takes the sidecar down.
    """
    if len(body) > max_bytes:
        return 413, {"error": f"request body exceeds {max_bytes} bytes"}
    try:
        request = _decode_request(body)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    started = time.perf_counter()
    try:
        payload = _with_latency(answer(request), (time.perf_counter() - started) * 1000.0)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except ModelNotReadyError as exc:
        return 503, {"error": str(exc)}
    except Exception as exc:
        LOGGER.exception("decision model failed to answer a request")
        return 500, {"error": f"{type(exc).__name__}: {exc}"}
    return 200, payload


class DecisionModelRuntime:
    """Loads one Clef release and serializes forward passes behind a single lock."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._systemone: Any = None

    @property
    def ready(self) -> bool:
        """Whether the weights are loaded and requests can be answered."""
        return self._systemone is not None and self._model is not None

    def load(self) -> None:
        """Download the pinned snapshot, import the repo's model code, and warm up."""
        # Imported here (not at module scope) so this file stays importable without a GPU
        # stack; see the module docstring.
        import torch
        from huggingface_hub import snapshot_download

        config = self._config
        LOGGER.info("downloading %s (revision %s)", config.model_id, config.revision)
        path = str(snapshot_download(config.model_id, revision=config.revision))
        if path not in sys.path:
            # The joint schema head lives in the snapshot as joint_schema_model.py.
            sys.path.insert(0, path)
        import joint_schema_model

        dtype = getattr(torch, config.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"CLEF_DTYPE is not a torch dtype: {config.dtype}")
        missing = missing_linear_attention_kernels()
        if missing:
            LOGGER.warning(
                "linear-attention CUDA kernels missing (%s); transformers will fall back to its "
                "slower reference path. Rebuild with "
                "--build-arg INSTALL_LINEAR_ATTENTION_KERNELS=1 to install them.",
                ", ".join(missing),
            )
        LOGGER.info(
            "loading weights on %s with dtype %s, quantization %s",
            config.device,
            config.dtype,
            config.quantization,
        )
        model, processor = joint_schema_model.load_release_model(
            path,
            device=config.device,
            dtype=dtype,
            **quantization_kwargs(config.quantization, compute_dtype=dtype),
        )
        systemone = joint_schema_model.systemone
        started = time.perf_counter()
        # Warm up before publishing the runtime: while the first forward pass compiles CUDA
        # graphs and loads lazy kernels it holds the lock for far longer than a decision
        # timeout, so /health must keep answering 503 until it is done.
        self._forward(systemone, model, processor, dict(_WARMUP_REQUEST))
        self._model = model
        self._processor = processor
        self._systemone = systemone
        LOGGER.info(
            "ready: warm-up forward pass took %.0f ms (max_length=%d)",
            (time.perf_counter() - started) * 1000.0,
            config.max_length,
        )

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one SystemOne request body with one joint-schema forward pass."""
        systemone = self._systemone
        model = self._model
        if systemone is None or model is None:
            raise ModelNotReadyError("decision model is still loading")
        return self._forward(systemone, model, self._processor, request)

    def _forward(
        self,
        systemone: Any,
        model: Any,
        processor: Any,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """One serialized joint-schema forward pass."""
        with self._lock:
            return systemone(
                model,
                processor,
                request,
                max_length=self._config.max_length,
            )


class DecisionModelHandler(BaseHTTPRequestHandler):
    """Routes the two SystemOne endpoints; ``http.server`` builds one instance per request."""

    protocol_version = "HTTP/1.1"
    server_version = "tracegate-decision-model/1"

    def __init__(
        self,
        *args: Any,
        config: ServerConfig,
        runtime: DecisionModelRuntime,
        **kwargs: Any,
    ) -> None:
        # BaseHTTPRequestHandler.__init__ serves the request, so bind state before calling it.
        self._config = config
        self._runtime = runtime
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Answer ``GET /health``."""
        if self._route() != HEALTH_PATH:
            self._respond(404, {"error": f"unknown path: {self.path}"})
            return
        status, payload = handle_health_request(self._config, ready=self._runtime.ready)
        self._respond(status, payload)

    def do_POST(self) -> None:
        """Answer ``POST /v1/systemone``."""
        if self._route() != SYSTEMONE_PATH:
            self._respond(404, {"error": f"unknown path: {self.path}"})
            return
        max_bytes = int(self._config.max_request_bytes)
        status, payload = handle_systemone_request(
            self._read_body(max_bytes + 1),
            self._runtime.answer,
            max_bytes=max_bytes,
        )
        self._respond(status, payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Send access logs through :mod:`logging` instead of straight to stderr."""
        LOGGER.info("%s %s", self.address_string(), format % args)

    def _route(self) -> str:
        path = self.path.split("?", 1)[0]
        return path.rstrip("/") or "/"

    def _read_body(self, limit: int) -> bytes:
        try:
            declared = int(self.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared <= 0:
            return b""
        return self.rfile.read(min(declared, limit))

    def _respond(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        if status == 413:
            # The unread tail of an oversized body would desynchronize a kept-alive connection.
            self.close_connection = True
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if self.close_connection:
            self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)


def build_server(config: ServerConfig, runtime: DecisionModelRuntime) -> ThreadingHTTPServer:
    """Create the threading HTTP server bound to *config* and *runtime*."""
    handler = partial(DecisionModelHandler, config=config, runtime=runtime)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    server.daemon_threads = True
    return server


def _load_then_serve_or_stop(
    runtime: DecisionModelRuntime,
    server: ThreadingHTTPServer,
    failed: threading.Event,
) -> None:
    """Load the model; shut the server down when loading fails so the container restarts."""
    try:
        runtime.load()
    except BaseException:
        LOGGER.exception("decision model failed to load")
        failed.set()
        server.shutdown()


def main() -> int:
    """Serve ``/health`` immediately and load the model in the background."""
    logging.basicConfig(
        level=os.environ.get("CLEF_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = config_from_env()
    if not revision_is_pinned(config.revision):
        LOGGER.warning(
            "CLEF_REVISION=%r is a moving branch: the served weights can change between "
            "restarts and runs will not be reproducible. Pin a commit sha.",
            config.revision,
        )
    runtime = DecisionModelRuntime(config)
    server = build_server(config, runtime)
    failed = threading.Event()
    threading.Thread(
        target=_load_then_serve_or_stop,
        args=(runtime, server, failed),
        name="clef-loader",
        daemon=True,
    ).start()
    LOGGER.info("listening on http://%s:%d", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("interrupted; shutting down")
    finally:
        server.server_close()
    return 1 if failed.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
