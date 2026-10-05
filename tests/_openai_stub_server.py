"""Stub HTTP server that mimics the OpenAI chat-completions API for integration tests."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llmdebug import _json as json


@dataclass(frozen=True)
class StubbedResponse:
    status: int
    body: dict[str, Any] | list[Any] | str | None
    headers: dict[str, str]


# Called with (method, path, parsed JSON body) when no queued response matches.
StubResponder = Callable[[str, str, "dict[str, Any] | list[Any] | None"], "StubbedResponse | None"]


class OpenAIStubServer:
    def __init__(self, responder: StubResponder | None = None, *, port: int = 0) -> None:
        self._port = port
        self._responses: dict[tuple[str, str], deque[StubbedResponse]] = defaultdict(deque)
        self._requests: list[dict[str, Any]] = []
        self._responder = responder
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def enqueue_json(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | list[Any],
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        response_headers = dict(headers or {})
        response_headers.setdefault("Content-Type", "application/json")
        key = (method.upper(), path)
        self._responses[key].append(
            StubbedResponse(status=int(status), body=body, headers=response_headers)
        )

    def enqueue_text(
        self,
        *,
        method: str,
        path: str,
        body: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        response_headers = dict(headers or {})
        response_headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        key = (method.upper(), path)
        self._responses[key].append(
            StubbedResponse(status=int(status), body=body, headers=response_headers)
        )

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def base_url_v1(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    def requests_for(self, *, method: str, path: str) -> list[dict[str, Any]]:
        method_upper = method.upper()
        with self._lock:
            return [
                dict(item)
                for item in self._requests
                if str(item.get("method")) == method_upper and str(item.get("path")) == path
            ]

    def last_request(self, *, method: str, path: str) -> dict[str, Any] | None:
        matches = self.requests_for(method=method, path=path)
        if not matches:
            return None
        return matches[-1]

    def _handler_cls(self):
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                self._handle()

            def log_message(self, format: str, *args: Any) -> None:
                del format, args
                return

            def _handle(self) -> None:
                content_length_raw = self.headers.get("Content-Length")
                try:
                    content_length = int(content_length_raw or 0)
                except ValueError:
                    content_length = 0
                body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                body_text = body_bytes.decode("utf-8", errors="replace")
                body_json: dict[str, Any] | list[Any] | None = None
                if body_text:
                    try:
                        parsed = json.loads(body_text)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (dict, list)):
                        body_json = parsed

                request_record: dict[str, Any] = {
                    "method": self.command.upper(),
                    "path": self.path,
                    "headers": {str(k).lower(): str(v) for k, v in self.headers.items()},
                    "body": body_text,
                    "json": body_json,
                }
                with outer._lock:
                    outer._requests.append(request_record)
                    queue = outer._responses[(self.command.upper(), self.path)]
                    scripted = (
                        outer._responder(self.command.upper(), self.path, body_json)
                        if not queue and outer._responder is not None
                        else None
                    )
                    if queue:
                        response = queue.popleft()
                    elif scripted is not None:
                        response = scripted
                    else:
                        response = StubbedResponse(
                            status=404,
                            body={
                                "error": "no_stubbed_response",
                                "method": self.command.upper(),
                                "path": self.path,
                            },
                            headers={"Content-Type": "application/json"},
                        )

                response_body = response.body
                if isinstance(response_body, str):
                    payload = response_body.encode("utf-8")
                elif response_body is None:
                    payload = b""
                else:
                    payload = json.dumps(response_body).encode("utf-8")

                self.send_response(int(response.status))
                for key, value in response.headers.items():
                    self.send_header(str(key), str(value))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

        return _Handler

    def __enter__(self) -> OpenAIStubServer:
        if self._server is not None:
            raise RuntimeError("stub server already running")
        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), self._handler_cls())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
