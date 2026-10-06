"""Tests for serialization."""

import re
import sys
import types

import numpy as np
import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import (
    compile_redactors,
    compute_array_stats,
    locals_metadata,
    redact_jsonlike,
    redact_text,
    serialize_locals,
    serialize_locals_bundle,
    serialize_locals_with_stats,
    summarize_array,
    to_jsonlike,
)


def _has_module(name: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


class _FakeTorchScalar:
    __module__ = "torch"

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return float(self._value)


class _FakeTorchDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __str__(self) -> str:
        return self.type


class _FakeTorchTensor:
    __module__ = "torch"

    def __init__(
        self,
        data: object,
        *,
        requires_grad: bool = False,
        device: str = "cpu",
    ) -> None:
        self._array = np.asarray(data)
        self.requires_grad = requires_grad
        self.device = _FakeTorchDevice(device)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.size = self._array.size

    def detach(self) -> "_FakeTorchTensor":
        return self

    def cpu(self) -> "_FakeTorchTensor":
        return _FakeTorchTensor(self._array.copy(), requires_grad=self.requires_grad, device="cpu")

    def float(self) -> "_FakeTorchTensor":
        return _FakeTorchTensor(
            self._array.astype(np.float32),
            requires_grad=self.requires_grad,
            device=str(self.device),
        )

    def sum(self) -> _FakeTorchScalar:
        return _FakeTorchScalar(float(self._array.sum()))

    def reshape(self, *shape: int) -> np.ndarray:
        return self._array.reshape(*shape)

    def ravel(self) -> np.ndarray:
        return self._array.ravel()

    def flatten(self) -> np.ndarray:
        return self._array.flatten()


class _FakeJaxArray:
    __module__ = "jax.numpy"

    def __init__(self, data: object) -> None:
        self._array = np.asarray(data)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.size = self._array.size

    def astype(self, dtype: object) -> "_FakeJaxArray":
        return _FakeJaxArray(self._array.astype(dtype))

    def reshape(self, *shape: int) -> np.ndarray:
        return self._array.reshape(*shape)

    def ravel(self) -> np.ndarray:
        return self._array.ravel()

    def flatten(self) -> np.ndarray:
        return self._array.flatten()


_FakeTorchTensor.__name__ = "Tensor"
_FakeJaxArray.__name__ = "Array"


def _torch_value(value: object) -> np.ndarray:
    if isinstance(value, _FakeTorchTensor):
        return value._array
    return np.asarray(value)


def _jax_value(value: object) -> np.ndarray:
    if isinstance(value, _FakeJaxArray):
        return value._array
    return np.asarray(value)


@pytest.fixture
def torch_module(monkeypatch: pytest.MonkeyPatch):
    if _has_module("torch"):
        import torch  # type: ignore[import-not-found]

        return torch

    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda data, requires_grad=False, device="cpu": _FakeTorchTensor(
        data,
        requires_grad=requires_grad,
        device=device,
    )
    fake_torch.isnan = lambda arr: _FakeTorchTensor(np.isnan(_torch_value(arr)).astype(np.float32))
    fake_torch.isinf = lambda arr: _FakeTorchTensor(np.isinf(_torch_value(arr)).astype(np.float32))
    fake_torch.min = lambda arr: _FakeTorchScalar(float(np.min(_torch_value(arr))))
    fake_torch.max = lambda arr: _FakeTorchScalar(float(np.max(_torch_value(arr))))
    fake_torch.mean = lambda arr: _FakeTorchScalar(float(np.mean(_torch_value(arr))))
    fake_torch.std = lambda arr, unbiased=False: _FakeTorchScalar(
        float(np.std(_torch_value(arr), ddof=1 if unbiased else 0))
    )
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


@pytest.fixture
def jax_numpy_module(monkeypatch: pytest.MonkeyPatch):
    if _has_module("jax"):
        import jax.numpy as jnp  # type: ignore[import-not-found]

        return jnp

    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.float32 = np.float32
    fake_jnp.array = lambda data: _FakeJaxArray(data)
    fake_jnp.isnan = lambda arr: np.isnan(_jax_value(arr))
    fake_jnp.isinf = lambda arr: np.isinf(_jax_value(arr))
    fake_jnp.min = lambda arr: np.min(_jax_value(arr))
    fake_jnp.max = lambda arr: np.max(_jax_value(arr))
    fake_jnp.mean = lambda arr: np.mean(_jax_value(arr))
    fake_jnp.std = lambda arr: np.std(_jax_value(arr))
    fake_jax.numpy = fake_jnp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    return fake_jnp


@pytest.fixture
def cfg():
    return SnapshotConfig()


def test_primitives(cfg):
    """Test primitive type serialization."""
    assert to_jsonlike(None, cfg) is None
    assert to_jsonlike(True, cfg) is True
    assert to_jsonlike(42, cfg) == 42
    assert to_jsonlike(3.14, cfg) == 3.14
    assert to_jsonlike("hello", cfg) == "hello"


def test_string_truncation(cfg):
    """Test long strings are truncated."""
    long_str = "x" * 1000
    result = to_jsonlike(long_str, cfg)
    assert isinstance(result, str)
    assert len(result) <= cfg.max_str
    assert "TRUNC" in result


def test_list_serialization(cfg):
    """Test list serialization with truncation."""
    small_list = [1, 2, 3]
    assert to_jsonlike(small_list, cfg) == [1, 2, 3]

    large_list = list(range(100))
    result = to_jsonlike(large_list, cfg)
    assert isinstance(result, list)
    assert len(result) == cfg.max_items + 1  # items + TRUNC marker
    assert result[-1] == "...[TRUNC]"


def test_set_serialization_is_deterministic(cfg):
    """Set serialization should be deterministic for stable snapshot diffs."""
    values = {"zeta", "alpha", "mu", "beta", "eta", "omega"}
    result = to_jsonlike(values, cfg)
    assert result == sorted(values)


def test_dict_serialization(cfg):
    """Test dict serialization."""
    d = {"a": 1, "b": "hello"}
    result = to_jsonlike(d, cfg)
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": "hello"}


def test_dict_keys_plain_str_and_repr_otherwise(cfg):
    """CAP-20: str keys are emitted as-is; other key types keep their repr."""
    from enum import Enum

    class Color(str, Enum):
        RED = "red"

    result = to_jsonlike({"k": 1, 2: "two", (1, 2): "pair", None: 0, Color.RED: "c"}, cfg)

    # A str-based enum member is a str, so it renders as its plain string value.
    assert result == {"k": 1, "2": "two", "(1, 2)": "pair", "None": 0, "red": "c"}
    assert all(type(key) is str for key in result)


def test_dict_key_collisions_keep_every_entry(cfg):
    """CAP-20: {1: ..., "1": ...} would collide as "1"; the second gets a stable suffix."""
    assert to_jsonlike({1: "int", "1": "str"}, cfg) == {"1": "int", "1#2": "str"}


def test_long_str_keys_are_truncated(cfg):
    result = to_jsonlike({"k" * (cfg.max_str + 50): 1}, cfg)

    (key,) = result
    assert key.endswith("...[TRUNC]")
    assert len(key) <= cfg.max_str


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_numpy_array_summary():
    """Test numpy array summarization."""
    import numpy as np

    cfg = SnapshotConfig()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = summarize_array(arr, cfg)
    assert result["__array__"] == "numpy.ndarray"
    assert result["shape"] == [2, 3]
    assert result["dtype"] == "float32"
    assert "head" in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_numpy_nan_detection():
    """Test NaN/Inf detection in numpy arrays."""
    import numpy as np

    cfg = SnapshotConfig()

    # Array with NaN and Inf
    arr = np.array([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
    result = summarize_array(arr, cfg)

    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 2  # +inf and -inf

    # Clean array should have no anomalies key
    clean_arr = np.array([1.0, 2.0, 3.0])
    clean_result = summarize_array(clean_arr, cfg)
    assert "anomalies" not in clean_result


def test_pytorch_tensor_summary(torch_module):
    """Test PyTorch tensor with requires_grad and anomalies."""
    cfg = SnapshotConfig()
    torch = torch_module

    # Tensor with requires_grad and anomalies
    t = torch.tensor([1.0, float("nan"), float("inf")], requires_grad=True)
    result = summarize_array(t, cfg)

    assert result["__array__"] == "torch.Tensor"
    assert result["requires_grad"] is True
    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 1

    # CPU device should not appear (it's the default)
    assert "device" not in result or result.get("device") is None

    # Tensor without requires_grad
    t2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    result2 = summarize_array(t2, cfg)
    assert "requires_grad" not in result2
    assert "anomalies" not in result2


def test_pytorch_cuda_device(torch_module):
    """Test that CUDA device is captured when available."""
    cfg = SnapshotConfig()
    torch = torch_module

    if torch.cuda.is_available():
        t = torch.tensor([1.0, 2.0], device="cuda")
        result = summarize_array(t, cfg)
        assert "device" in result
        assert "cuda" in result["device"]
    else:
        # Just test that CPU device is omitted
        t = torch.tensor([1.0, 2.0], device="cpu")
        result = summarize_array(t, cfg)
        assert "device" not in result


def test_jax_array_summary(jax_numpy_module):
    """Test JAX array with anomalies."""
    cfg = SnapshotConfig()
    jnp = jax_numpy_module

    # JAX array with anomalies
    arr = jnp.array([1.0, float("nan"), float("inf")])
    result = summarize_array(arr, cfg)

    assert "jax" in result["__array__"].lower()
    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 1

    # JAX doesn't have requires_grad (uses functional transforms)
    assert "requires_grad" not in result


# ============================================================================
# Redaction Tests
# ============================================================================


def test_compile_redactors_treats_unresolved_patterns_as_empty() -> None:
    compile_redactors(("password",))

    assert compile_redactors(None) == []


def test_compile_redactors_results_are_independent_lists() -> None:
    """CAP-07: a caller mutating its list must not change what the next caller gets."""
    first = compile_redactors(("alpha",))
    first.clear()

    assert [rx.pattern for rx in compile_redactors(("alpha",))] == ["alpha"]


def test_compile_redactors_alternating_configs() -> None:
    a, b = ("a1", "a2"), ("b1",)
    for _ in range(3):
        assert [rx.pattern for rx in compile_redactors(a)] == ["a1", "a2"]
        assert [rx.pattern for rx in compile_redactors(b)] == ["b1"]


def test_compile_redactors_skips_invalid_patterns(capsys: pytest.CaptureFixture[str]) -> None:
    compiled = compile_redactors(("ok", "(unclosed"))

    assert [rx.pattern for rx in compiled] == ["ok"]
    assert "invalid redaction pattern '(unclosed'" in capsys.readouterr().err


def test_compile_redactors_is_consistent_across_threads() -> None:
    """CAP-07: concurrent captures with different configs each get their own patterns."""
    import threading

    configs = [tuple(f"t{i}_{j}" for j in range(3)) for i in range(6)]
    mismatches: list[int] = []

    def worker(index: int) -> None:
        mine = configs[index]
        for _ in range(500):
            if [rx.pattern for rx in compile_redactors(mine)] != list(mine):
                mismatches.append(index)
                return

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(configs))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)

    assert mismatches == []


def test_redact_text_basic():
    """Test direct redact_text function with simple pattern."""
    redactors = compile_redactors(("password",))
    result = redact_text("my password is secret", redactors)
    assert result == "my [REDACTED] is secret"


def test_redact_text_compiled_pattern():
    """Test redact_text with pre-compiled regex Pattern."""
    pattern = re.compile(r"api[_-]?key", re.IGNORECASE)
    redactors = compile_redactors((pattern,))

    result = redact_text("my API_KEY is abc123", redactors)
    assert "[REDACTED]" in result
    assert "API_KEY" not in result


def test_redact_text_multiple_patterns():
    """Test redact_text with multiple patterns."""
    redactors = compile_redactors(("password", "secret", "token"))
    result = redact_text("password=abc, secret=xyz, token=123", redactors)
    assert result == "[REDACTED]=abc, [REDACTED]=xyz, [REDACTED]=123"


def test_serialize_locals_redacts_password():
    """Test that sensitive patterns are redacted while preserving keys by default."""
    cfg = SnapshotConfig(redact=("password", "secret"))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "password": "password:hunter2",
        "username": "alice",
        "secret_key": "secret:abc123",
    }

    result = serialize_locals(local_vars, cfg, redactors)
    assert "password" in result
    assert "secret_key" in result
    # CAP-03: the whole value of a secret-named local is replaced, not just the match.
    assert result["password"] == "[REDACTED]"
    assert result["secret_key"] == "[REDACTED]"
    assert result["username"] == "alice"


def test_serialize_locals_redacts_nested_dict():
    """Test that redaction works for nested values while preserving nested keys."""
    cfg = SnapshotConfig(redact=("password", "secret"))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "config": {
            "database": {
                "password": "secret123",
                "host": "localhost",
            }
        }
    }

    result = serialize_locals(local_vars, cfg, redactors)
    database = result["config"]["database"]
    assert database["password"] == "[REDACTED]"  # CAP-03: secret-named key
    assert database["host"] == "localhost"


def test_serialize_locals_redacts_in_arrays():
    """Test that redaction works in array/list elements."""
    cfg = SnapshotConfig(redact=("password",))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "credentials": ["user1:password123", "user2:safe"],
    }

    result = serialize_locals(local_vars, cfg, redactors)

    result_str = str(result)
    # "password123" should be partially redacted (the "password" part)
    assert "[REDACTED]" in result_str


def test_serialize_locals_no_redaction_when_empty():
    """Test that empty redact list doesn't modify values."""
    cfg = SnapshotConfig(redact=())
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "password": "hunter2",
        "secret": "abc123",
    }

    result = serialize_locals(local_vars, cfg, redactors)

    # Without redaction patterns, values should be preserved
    assert result["password"] == "hunter2"
    assert result["secret"] == "abc123"


def test_serialize_locals_redact_keys_opt_in() -> None:
    cfg = SnapshotConfig(redact=("password",), redact_keys=True)
    redactors = compile_redactors(cfg.redact)

    local_vars = {"password_key": "ok", "message": "password inside value"}
    result = serialize_locals(local_vars, cfg, redactors)

    assert "password_key" not in result
    assert "[REDACTED]_key" in result
    assert result["message"] == "[REDACTED] inside value"


def test_serialize_locals_default_avoids_key_collision() -> None:
    cfg = SnapshotConfig(redact=("api_key|secret_token",))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "api_key": "api_key=value",
        "secret_token": "secret_token=value",
    }
    result = serialize_locals(local_vars, cfg, redactors)

    assert set(result.keys()) == {"api_key", "secret_token"}
    assert result["api_key"] == "[REDACTED]"
    assert result["secret_token"] == "[REDACTED]"


def test_serialize_locals_redact_keys_preserves_collisions() -> None:
    # Non-secret names, so the values stay visible and show which entry got the suffix.
    cfg = SnapshotConfig(redact=("user_id|order_id",), redact_keys=True)
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "user_id": "alpha",
        "order_id": "beta",
    }
    result = serialize_locals(local_vars, cfg, redactors)

    assert result["[REDACTED]"] == "alpha"
    assert result["[REDACTED]#2"] == "beta"


def test_serialize_locals_bundle_matches_compat_wrappers() -> None:
    cfg = SnapshotConfig(redact=("secret",))
    redactors = compile_redactors(cfg.redact)
    local_vars = {
        "a": 1,
        "msg": "secret-token",
        "items": [1, 2, 3],
    }

    bundle = serialize_locals_bundle(local_vars, cfg, redactors)
    values, truncated, truncated_keys = serialize_locals_with_stats(local_vars, cfg, redactors)
    meta = locals_metadata(local_vars, cfg)

    assert bundle.values == values
    assert bundle.truncated is truncated
    assert bundle.truncated_keys == truncated_keys
    assert bundle.metadata == meta


def test_serialize_locals_bundle_redacted_key_metadata_alignment() -> None:
    cfg = SnapshotConfig(redact=("api_key|secret_token",), redact_keys=True)
    redactors = compile_redactors(cfg.redact)
    local_vars = {"api_key": "alpha", "secret_token": "beta"}

    bundle = serialize_locals_bundle(local_vars, cfg, redactors)

    assert set(bundle.values.keys()) == {"[REDACTED]", "[REDACTED]#2"}
    assert set(bundle.metadata.keys()) == {"[REDACTED]", "[REDACTED]#2"}
    assert bundle.metadata["[REDACTED]"]["type"] == "builtins.str"
    assert bundle.metadata["[REDACTED]#2"]["type"] == "builtins.str"


def test_redact_text_regex_pattern():
    """Test redact_text with regex metacharacters."""
    # Pattern to match API keys like "sk-abc123..."
    pattern = re.compile(r"sk-[a-zA-Z0-9]+")
    redactors = compile_redactors((pattern,))

    result = redact_text("API key: sk-abc123XYZ", redactors)
    assert result == "API key: [REDACTED]"
    assert "sk-" not in result


# ============================================================================
# redact_jsonlike Tests
# ============================================================================


def test_redact_jsonlike_string_leaf():
    """Test that string leaves are redacted."""
    redactors = compile_redactors(("secret",))
    assert redact_jsonlike("my secret value", redactors) == "my [REDACTED] value"


def test_redact_jsonlike_non_string_leaves():
    """Test that non-string leaves pass through unchanged."""
    redactors = compile_redactors(("secret",))
    assert redact_jsonlike(42, redactors) == 42
    assert redact_jsonlike(3.14, redactors) == 3.14
    assert redact_jsonlike(True, redactors) is True
    assert redact_jsonlike(None, redactors) is None


def test_redact_jsonlike_nested_dict():
    """Test redaction in nested dicts preserves structure and redacts values."""
    redactors = compile_redactors(("password",))
    data = {"outer": {"inner": {"password": "password=hunter2", "safe": "ok"}}}
    result = redact_jsonlike(data, redactors)

    assert isinstance(result, dict)
    outer = result["outer"]
    assert isinstance(outer, dict)
    inner = outer["inner"]
    assert isinstance(inner, dict)

    assert inner["password"] == "[REDACTED]"  # CAP-03: secret-named key
    assert inner["safe"] == "ok"
    # Keys preserved by default
    assert "password" in inner


def test_redact_jsonlike_nested_list():
    """Test redaction in nested lists."""
    redactors = compile_redactors(("token",))
    data = [["token=abc", "safe"], "token=xyz"]
    result = redact_jsonlike(data, redactors)

    assert result == [["[REDACTED]=abc", "safe"], "[REDACTED]=xyz"]


def test_redact_jsonlike_tuple_set_and_frozenset() -> None:
    redactors = compile_redactors(("secret",))
    data = {
        # Three items: ("secret", value) would be a secret-named pair and redact "safe".
        "items_tuple": ("secret", "safe", "more"),
        "items_set": {"safe", "secret"},
        "items_frozenset": frozenset({"secret", "safe"}),
    }

    result = redact_jsonlike(data, redactors)

    assert result["items_tuple"] == ["[REDACTED]", "safe", "more"]
    assert sorted(result["items_set"]) == ["[REDACTED]", "safe"]
    assert sorted(result["items_frozenset"]) == ["[REDACTED]", "safe"]


def test_redact_jsonlike_empty_containers():
    """Test that empty containers pass through unchanged."""
    redactors = compile_redactors(("secret",))
    assert redact_jsonlike({}, redactors) == {}
    assert redact_jsonlike([], redactors) == []


def test_redact_jsonlike_keys_preserved_by_default():
    """Test that dict keys are NOT redacted when redact_keys=False (default)."""
    redactors = compile_redactors(("password",))
    data = {"password_field": "password_value"}
    result = redact_jsonlike(data, redactors)

    assert isinstance(result, dict)
    assert "password_field" in result
    assert result["password_field"] == "[REDACTED]"  # CAP-03: secret-named key


def test_redact_jsonlike_keys_redacted_when_opted_in():
    """Test that dict keys ARE redacted when redact_keys=True."""
    redactors = compile_redactors(("password",))
    data = {"password_field": "password_value"}
    result = redact_jsonlike(data, redactors, redact_keys=True)

    assert isinstance(result, dict)
    assert "password_field" not in result
    assert "[REDACTED]_field" in result
    assert result["[REDACTED]_field"] == "[REDACTED]"  # CAP-03: secret-named key


def test_redact_jsonlike_key_collisions_are_preserved() -> None:
    # Non-secret names, so the values stay visible and show which entry got the suffix.
    redactors = compile_redactors(("user_id|order_id",))
    data = {"user_id": "first", "order_id": "second"}
    result = redact_jsonlike(data, redactors, redact_keys=True)

    assert isinstance(result, dict)
    assert result["[REDACTED]"] == "first"
    assert result["[REDACTED]#2"] == "second"


def test_redact_jsonlike_nested_key_collisions_are_preserved() -> None:
    redactors = compile_redactors(("token_a|token_b",))
    data = {"outer": {"token_a": "x", "token_b": "y"}}
    result = redact_jsonlike(data, redactors, redact_keys=True)

    assert isinstance(result, dict)
    assert isinstance(result.get("outer"), dict)
    nested = result["outer"]
    assert nested["[REDACTED]"] == "x"
    assert nested["[REDACTED]#2"] == "y"


def test_redact_jsonlike_mixed_types_in_list():
    """Test that lists with mixed types are handled correctly."""
    redactors = compile_redactors(("secret",))
    data = ["secret_value", 42, None, True, {"key": "secret_data"}]
    result = redact_jsonlike(data, redactors)

    assert isinstance(result, list)
    assert result[0] == "[REDACTED]_value"
    assert result[1] == 42
    assert result[2] is None
    assert result[3] is True
    assert result[4] == {"key": "[REDACTED]_data"}


# ============================================================================
# Array Statistics Tests
# ============================================================================


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_numpy():
    """Test statistical computation for numpy arrays."""
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.0
    # std is population std
    assert abs(stats["std"] - 1.4142135) < 0.001


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_empty_array():
    """Test that empty arrays return None for stats."""
    import numpy as np

    arr = np.array([])
    stats = compute_array_stats(arr)

    assert stats is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_non_numeric():
    """Test that non-numeric arrays return None for stats."""
    import numpy as np

    arr = np.array(["a", "b", "c"])
    stats = compute_array_stats(arr)

    assert stats is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_integer():
    """Test stats for integer arrays."""
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_respects_max_elements():
    """Large arrays should be skipped when max_elements is exceeded."""
    import numpy as np

    arr = np.arange(10, dtype=np.float64)
    stats = compute_array_stats(arr, max_elements=3)
    assert stats is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_with_stats():
    """Test array summary includes stats when configured."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=True)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = summarize_array(arr, cfg)

    assert "stats" in result
    assert result["stats"]["min"] == 1.0
    assert result["stats"]["max"] == 5.0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_with_stats_avoids_duplicate_full_scans(monkeypatch):
    import numpy as np

    import llmdebug.serialize as serialize_module

    cfg = SnapshotConfig(include_array_stats=True)
    arr = np.array([1.0, 2.0, 3.0, float("nan"), float("inf")])

    def _unexpected_count(_arr):
        raise AssertionError("count_anomalies should not be called in combined stats path")

    def _unexpected_stats(_arr, *, max_elements=serialize_module.MAX_ARRAY_ELEMENTS_FOR_STATS):
        del max_elements
        raise AssertionError("compute_array_stats should not be called in combined stats path")

    monkeypatch.setattr(serialize_module, "count_anomalies", _unexpected_count)
    monkeypatch.setattr(serialize_module, "compute_array_stats", _unexpected_stats)

    result = summarize_array(arr, cfg)

    assert "stats" in result
    assert result["anomalies"] == {"nan": 1, "inf": 1}


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_without_stats():
    """Test array summary excludes stats by default."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=False)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = summarize_array(arr, cfg)

    assert "stats" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_omits_head_when_too_large(monkeypatch):
    import numpy as np

    import llmdebug.serialize as serialize_module

    monkeypatch.setattr(serialize_module, "MAX_ARRAY_ELEMENTS_FOR_HEAD", 3)
    arr = np.arange(10, dtype=np.float64)
    result = summarize_array(arr, SnapshotConfig())

    assert result["head_omitted_reason"] == "too_large"
    assert "head" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_omits_stats_when_too_large(monkeypatch):
    import numpy as np

    import llmdebug.serialize as serialize_module

    monkeypatch.setattr(serialize_module, "MAX_ARRAY_ELEMENTS_FOR_STATS", 3)
    arr = np.arange(10, dtype=np.float64)
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))

    assert result["stats_omitted_reason"] == "too_large"
    assert "stats" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_omits_anomalies_when_too_large(monkeypatch):
    import numpy as np

    import llmdebug.serialize as serialize_module

    monkeypatch.setattr(serialize_module, "MAX_ARRAY_ELEMENTS_FOR_ANOMALIES", 3)
    arr = np.array([1.0, float("nan"), float("inf"), 4.0, 5.0])
    result = summarize_array(arr, SnapshotConfig())

    assert result["anomalies_omitted_reason"] == "too_large"
    assert "anomalies" not in result


def test_summarize_array_omits_non_cpu_stats_and_head():
    class _FakeDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    class _FakeTensor:
        __module__ = "torch"

        def __init__(self) -> None:
            self.shape = (10,)
            self.dtype = "float32"
            self.device = _FakeDevice()
            self.size = 10

    result = summarize_array(_FakeTensor(), SnapshotConfig(include_array_stats=True))
    assert result["device"] == "cuda:0"
    assert result["head_omitted_reason"] == "non_cpu_device"
    assert result["stats_omitted_reason"] == "non_cpu_device"
    assert "head" not in result
    assert "stats" not in result


def test_compute_array_stats_pytorch(torch_module):
    """Test statistical computation for PyTorch tensors."""
    torch = torch_module
    arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.0


def test_compute_array_stats_jax(jax_numpy_module):
    """Test statistical computation for JAX arrays."""
    jnp = jax_numpy_module
    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


# ============================================================================
# Coverage-gap tests
# ============================================================================


def test_insert_redacted_key_triple_collision() -> None:
    """Three keys redacted to the same string should produce #2 and #3 suffixes."""
    redactors = compile_redactors(("key_a|key_b|key_c",))
    data = {"key_a": "x", "key_b": "y", "key_c": "z"}
    result = redact_jsonlike(data, redactors, redact_keys=True)
    assert isinstance(result, dict)
    assert result["[REDACTED]"] == "x"
    assert result["[REDACTED]#2"] == "y"
    assert result["[REDACTED]#3"] == "z"


def test_get_tensor_device_jax_style_non_cpu() -> None:
    """JAX-like device (no .type attr) on a non-CPU device returns device string."""
    from llmdebug.serialize import get_tensor_device

    class _JaxDevice:
        def __str__(self) -> str:
            return "gpu:0"

    class _FakeArr:
        __module__ = "jax"

        def __init__(self) -> None:
            self.device = _JaxDevice()

    assert get_tensor_device(_FakeArr()) == "gpu:0"


def test_get_tensor_device_jax_style_cpu() -> None:
    """JAX-like device (no .type attr) containing 'cpu' returns None."""
    from llmdebug.serialize import get_tensor_device

    class _JaxCpuDevice:
        def __str__(self) -> str:
            return "cpu:0"

    class _FakeArr:
        __module__ = "jax"

        def __init__(self) -> None:
            self.device = _JaxCpuDevice()

    assert get_tensor_device(_FakeArr()) is None


def test_get_tensor_device_empty_devices_set() -> None:
    """Object with .devices() returning empty set returns None."""
    from llmdebug.serialize import get_tensor_device

    class _FakeArr:
        __module__ = "jax"

        def devices(self) -> set[str]:
            return set()

    assert get_tensor_device(_FakeArr()) is None


def test_get_tensor_device_devices_with_cpu() -> None:
    """Object with .devices() returning set containing 'cpu' returns None."""
    from llmdebug.serialize import get_tensor_device

    class _FakeArr:
        __module__ = "jax"

        def devices(self) -> set[str]:
            return {"CPU:0"}

    assert get_tensor_device(_FakeArr()) is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_count_anomalies_only_nan() -> None:
    """Array with only NaN (no Inf) should have nan key but no inf key."""
    from llmdebug.serialize import count_anomalies

    arr = np.array([1.0, float("nan"), 2.0])
    result = count_anomalies(arr)
    assert result is not None
    assert result["nan"] == 1
    assert "inf" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_count_anomalies_only_inf() -> None:
    """Array with only Inf (no NaN) should have inf key but no nan key."""
    from llmdebug.serialize import count_anomalies

    arr = np.array([1.0, float("inf"), 2.0])
    result = count_anomalies(arr)
    assert result is not None
    assert result["inf"] == 1
    assert "nan" not in result


def test_compute_array_stats_unknown_module() -> None:
    """Array from unknown module (e.g. tensorflow) returns None from compute_array_stats."""

    class _FakeTfTensor:
        __module__ = "tensorflow.python.framework.ops"
        __name__ = "EagerTensor"

        def __init__(self) -> None:
            self.shape = (3,)
            self.dtype = "float32"
            self.size = 3

    result = compute_array_stats(_FakeTfTensor())
    assert result is None


def test_compute_array_stats_and_anomalies_jax_path(jax_numpy_module) -> None:
    """Combined stats+anomalies path dispatches correctly for JAX arrays."""
    jnp = jax_numpy_module
    arr = jnp.array([1.0, 2.0, 3.0])
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))
    assert "stats" in result
    assert result["stats"]["min"] == 1.0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_build_anomaly_counts_only_nan_via_combined_path() -> None:
    """Combined stats path with array having only NaN exercises _build_anomaly_counts nan-only."""
    arr = np.array([1.0, float("nan"), 2.0])
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))
    assert result.get("anomalies") == {"nan": 1}


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_build_anomaly_counts_only_inf_via_combined_path() -> None:
    """Combined stats path with array having only Inf exercises _build_anomaly_counts inf-only."""
    arr = np.array([1.0, float("inf"), 2.0])
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))
    assert result.get("anomalies") == {"inf": 1}


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_add_array_stats_returns_none_for_non_numeric(monkeypatch) -> None:
    """Non-combined path with non-numeric array: stats returns None (falsy branch)."""
    import llmdebug.serialize as serialize_module

    monkeypatch.setattr(serialize_module, "MAX_ARRAY_ELEMENTS_FOR_ANOMALIES", 3)
    arr = np.array([True, False, True, False, True])
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))
    assert "stats" not in result
    assert result.get("anomalies_omitted_reason") == "too_large"


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_no_shape_attr() -> None:
    """Array-like object without .shape still serializes."""

    class _ShapelessArray:
        __module__ = "numpy"
        __name__ = "ndarray"

        def __init__(self) -> None:
            self.dtype = "float32"
            self.size = 3
            self._data = np.array([1.0, 2.0, 3.0])

        def reshape(self, *args: int) -> np.ndarray:
            return self._data.reshape(*args)

    result = summarize_array(_ShapelessArray(), SnapshotConfig())
    assert "__array__" in result
    assert "shape" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_no_dtype_attr() -> None:
    """Array-like object without .dtype still serializes."""

    class _DtypelessArray:
        __module__ = "numpy"
        __name__ = "ndarray"

        def __init__(self) -> None:
            self.shape = (3,)
            self.size = 3
            self._data = np.array([1.0, 2.0, 3.0])

        def reshape(self, *args: int) -> np.ndarray:
            return self._data.reshape(*args)

    result = summarize_array(_DtypelessArray(), SnapshotConfig())
    assert "__array__" in result
    assert result["shape"] == [3]
    assert "dtype" not in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_combined_stats_path_falsy_stats() -> None:
    """Boolean-dtype array in combined path yields stats=None (falsy branch)."""
    arr = np.array([True, False, True])
    result = summarize_array(arr, SnapshotConfig(include_array_stats=True))
    assert "stats" not in result


def test_summarize_dataframe_dtypes_truncated() -> None:
    """DataFrame with many dtypes truncates to max_items."""
    from llmdebug.serialize import summarize_dataframe

    cols = [f"col{i}" for i in range(10)]

    class _DtypesObj:
        def items(self) -> list[tuple[str, str]]:
            return [(c, "int64") for c in cols]

    class _FakeHead:
        def __init__(self) -> None:
            self.values = type("V", (), {"tolist": lambda s: [[0] * 10]})()

    class _FakeDF:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (100, 10)
            self.columns = cols
            self.dtypes = _DtypesObj()

        def head(self, n: int) -> "_FakeHead":
            return _FakeHead()

        def __len__(self) -> int:
            return 100

    _FakeDF.__name__ = "DataFrame"
    cfg = SnapshotConfig(max_items=3)
    result = summarize_dataframe(_FakeDF(), cfg)
    assert result["dtypes_truncated"] is True
    assert len(result["dtypes"]) == 3


def test_summarize_dataframe_dtypes_empty_items() -> None:
    """DataFrame whose dtypes.items() returns empty list skips dtypes block."""
    from llmdebug.serialize import summarize_dataframe

    class _EmptyDtypes:
        def items(self) -> list[tuple[str, str]]:
            return []

    class _FakeHead:
        def __init__(self) -> None:
            self.values = type("V", (), {"tolist": lambda s: [[1, 2]]})()

    class _FakeDF:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (5, 2)
            self.columns = ["a", "b"]
            self.dtypes = _EmptyDtypes()

        def head(self, n: int) -> "_FakeHead":
            return _FakeHead()

        def __len__(self) -> int:
            return 5

    _FakeDF.__name__ = "DataFrame"
    result = summarize_dataframe(_FakeDF(), SnapshotConfig())
    assert "dtypes" not in result


def test_summarize_dataframe_dtypes_raises() -> None:
    """DataFrame whose dtypes.items() throws should silently skip dtypes."""
    from llmdebug.serialize import summarize_dataframe

    class _BadDtypes:
        def items(self) -> list[tuple[str, str]]:
            raise RuntimeError("boom")

    class _FakeHead:
        def __init__(self) -> None:
            self.values = type("V", (), {"tolist": lambda s: [[1, 2]]})()

    class _FakeDF:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (5, 2)
            self.columns = ["a", "b"]
            self.dtypes = _BadDtypes()

        def head(self, n: int) -> "_FakeHead":
            return _FakeHead()

        def __len__(self) -> int:
            return 5

    _FakeDF.__name__ = "DataFrame"
    result = summarize_dataframe(_FakeDF(), SnapshotConfig())
    assert "dtypes" not in result


def test_summarize_dataframe_null_sum_not_callable() -> None:
    """DataFrame where null_mask.sum is not callable skips null_counts."""
    from llmdebug.serialize import summarize_dataframe

    class _NullMask:
        sum = "not_callable"

    class _FakeHead:
        def __init__(self) -> None:
            self.values = type("V", (), {"tolist": lambda s: [[1]]})()

    class _FakeDF:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (3, 1)
            self.columns = ["a"]

        def isna(self) -> _NullMask:
            return _NullMask()

        def head(self, n: int) -> "_FakeHead":
            return _FakeHead()

        def __len__(self) -> int:
            return 3

    _FakeDF.__name__ = "DataFrame"
    result = summarize_dataframe(_FakeDF(), SnapshotConfig())
    assert "null_counts" not in result


def test_summarize_dataframe_null_items_not_callable() -> None:
    """DataFrame where null_counts.items is not callable skips null_counts."""
    from llmdebug.serialize import summarize_dataframe

    class _NullCounts:
        items = "not_callable"

    class _NullMask:
        def sum(self) -> _NullCounts:
            return _NullCounts()

    class _FakeHead:
        def __init__(self) -> None:
            self.values = type("V", (), {"tolist": lambda s: [[1]]})()

    class _FakeDF:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (3, 1)
            self.columns = ["a"]

        def isna(self) -> _NullMask:
            return _NullMask()

        def head(self, n: int) -> "_FakeHead":
            return _FakeHead()

        def __len__(self) -> int:
            return 3

    _FakeDF.__name__ = "DataFrame"
    result = summarize_dataframe(_FakeDF(), SnapshotConfig())
    assert "null_counts" not in result


def test_summarize_series_no_name() -> None:
    """Series without a name attribute omits the 'name' key."""
    from llmdebug.serialize import summarize_series

    class _FakeSeries:
        __module__ = "pandas.core.series"

        def __init__(self) -> None:
            self.shape = (3,)
            self.dtype = "int64"

        def head(self, n: int) -> "_FakeSeries":
            return self

        def tolist(self) -> list[int]:
            return [1, 2, 3]

        def __len__(self) -> int:
            return 3

    _FakeSeries.__name__ = "Series"
    result = summarize_series(_FakeSeries(), SnapshotConfig())
    assert "name" not in result
    assert result["dtype"] == "int64"


def test_summarize_series_name_none() -> None:
    """Series with name=None omits the 'name' key."""
    from llmdebug.serialize import summarize_series

    class _FakeSeries:
        __module__ = "pandas.core.series"

        def __init__(self) -> None:
            self.shape = (5,)
            self.dtype = "float64"
            self.name = None

        def head(self, n: int) -> "_FakeSeries":
            return self

        def tolist(self) -> list[float]:
            return [1.0, 2.0, 3.0, 4.0, 5.0]

        def __len__(self) -> int:
            return 5

    _FakeSeries.__name__ = "Series"
    result = summarize_series(_FakeSeries(), SnapshotConfig())
    assert "name" not in result


def test_summarize_series_no_dtype() -> None:
    """Series without dtype attribute omits 'dtype' key."""
    from llmdebug.serialize import summarize_series

    class _FakeSeries:
        __module__ = "pandas.core.series"

        def __init__(self) -> None:
            self.shape = (3,)
            self.name = "col_a"

        def head(self, n: int) -> "_FakeSeries":
            return self

        def tolist(self) -> list[int]:
            return [1, 2, 3]

        def __len__(self) -> int:
            return 3

    _FakeSeries.__name__ = "Series"
    result = summarize_series(_FakeSeries(), SnapshotConfig())
    assert "dtype" not in result
    assert result["name"] == "col_a"
