"""Additional tests for llmdebug public API helpers in __init__.py."""

from __future__ import annotations

import builtins
import importlib
import warnings
from dataclasses import replace
from importlib.metadata import version
from typing import Any

import pytest

import llmdebug
from llmdebug.config import SnapshotConfig, resolve_snapshot_config
from llmdebug.output import get_latest_snapshot


def test_init_module_reload_executes_top_level_exports() -> None:
    reloaded = importlib.reload(llmdebug)
    assert reloaded.__version__
    assert reloaded.__version__ == version("llmdebug")
    assert "debug_snapshot" in reloaded.__all__
    assert "load_jupyter" in reloaded.__all__


def test_maybe_warn_about_unredacted_defaults_emits_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)

    with pytest.warns(UserWarning, match="no redaction settings were provided"):
        llmdebug._maybe_warn_about_unredacted_defaults(
            redaction_profile=None,
            redact=None,
            redact_keys=None,
            redact_traceback=None,
            redact_exception_strings=None,
        )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        llmdebug._maybe_warn_about_unredacted_defaults(
            redaction_profile=None,
            redact=None,
            redact_keys=None,
            redact_traceback=None,
            redact_exception_strings=None,
        )
    assert captured == []


def test_maybe_warn_about_unredacted_defaults_no_warning_when_inputs_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        llmdebug._maybe_warn_about_unredacted_defaults(
            redaction_profile="ci",
            redact=None,
            redact_keys=None,
            redact_traceback=None,
            redact_exception_strings=None,
        )
    assert captured == []


def test_try_capture_logs_when_capture_exception_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(llmdebug, "capture_exception", _boom)
    cfg = SnapshotConfig()

    try:
        raise ValueError("trigger")
    except ValueError as exc:
        llmdebug._try_capture("api-test", exc, cfg)

    assert "capture failed for api-test: RuntimeError: capture exploded" in capsys.readouterr().err


def test_load_jupyter_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import llmdebug.jupyter_plugin as jp

    captured: list[dict[str, Any]] = []

    def _fake_install(**kwargs: Any) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(jp, "install", _fake_install)
    llmdebug.load_jupyter(enable_auto_capture=False, detail="full")

    assert captured == [{"enable_auto_capture": False, "detail": "full"}]


def test_load_jupyter_raises_runtime_error_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.endswith("jupyter_plugin"):
            raise ImportError("missing IPython")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="IPython is not installed"):
        llmdebug.load_jupyter()


def test_ipython_extension_functions_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    import llmdebug.jupyter_plugin as jp

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(jp, "load_ipython_extension", lambda ip: calls.append(("load", ip)))
    monkeypatch.setattr(jp, "unload_ipython_extension", lambda ip: calls.append(("unload", ip)))

    ip = object()
    llmdebug.load_ipython_extension(ip)
    llmdebug.unload_ipython_extension(ip)

    assert calls == [("load", ip), ("unload", ip)]


def test_debug_snapshot_accepts_snapshot_config_without_deprecation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)
    monkeypatch.setattr(llmdebug, "_legacy_capture_kwargs_warning_emitted", set())
    cfg = SnapshotConfig(out_dir=str(tmp_path), redaction_profile="ci")

    @llmdebug.debug_snapshot(config=cfg)
    def _boom() -> None:
        raise ValueError("boom")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="boom"):
            _boom()

    assert not any(issubclass(item.category, DeprecationWarning) for item in captured)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["name"] == "_boom"


def test_snapshot_config_without_redaction_settings_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)

    with pytest.warns(UserWarning, match="no redaction settings were provided"):

        @llmdebug.debug_snapshot(config=SnapshotConfig())
        def _noop() -> None:
            pass


def test_snapshot_config_public_redaction_attributes_are_concrete() -> None:
    cfg = SnapshotConfig()

    assert cfg.redact == ()
    assert tuple(cfg.redact) == ()
    assert cfg.redact_keys is False
    assert cfg.redact_traceback is False
    assert cfg.redact_exception_strings is False
    assert cfg.has_explicit_redaction_settings is False
    assert SnapshotConfig(redact=()).has_explicit_redaction_settings is True


def test_snapshot_config_replace_preserves_unrelated_inheritance() -> None:
    inherited_prod = SnapshotConfig(redaction_profile="prod")
    moved = replace(inherited_prod, out_dir="other-snapshots")

    assert moved.redact
    assert moved.redact_traceback is True
    assert moved.redact_exception_strings is True


def test_snapshot_config_profile_replace_requires_intent_aware_update() -> None:
    with pytest.raises(ValueError, match=r"SnapshotConfig\.with_updates"):
        replace(SnapshotConfig(), redaction_profile="prod")

    promoted = SnapshotConfig().with_updates(redaction_profile="prod")
    keep_strict_traceback = SnapshotConfig(redaction_profile="prod").with_updates(
        redaction_profile="ci",
        redact_traceback=True,
    )
    explicit_prod_override = SnapshotConfig(redaction_profile="ci").with_updates(
        redaction_profile="prod",
        redact_traceback=False,
    )

    assert promoted.redact
    assert promoted.redact_traceback is True
    assert keep_strict_traceback.redact_traceback is True
    assert explicit_prod_override.redact_traceback is False


def test_snapshot_config_replace_preserves_explicit_overrides() -> None:
    explicit = SnapshotConfig(
        redaction_profile="prod",
        redact=(),
        redact_traceback=False,
        redact_exception_strings=False,
    )
    moved = replace(explicit, out_dir="other-snapshots")
    overridden_after_construction = replace(SnapshotConfig(redaction_profile="prod"), redact=())

    assert moved.redact == ()
    assert moved.redact_traceback is False
    assert moved.redact_exception_strings is False
    assert overridden_after_construction.redact == ()


def test_snapshot_config_with_updates_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match="unknown fields: 'redact_tracback'"):
        SnapshotConfig().with_updates(redact_tracback=True)


def test_snapshot_config_redaction_profile_is_applied_to_capture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)
    cfg = SnapshotConfig(
        out_dir=str(tmp_path),
        redaction_profile="prod",
        include_git=False,
        include_env=False,
        include_async_context=False,
    )

    @llmdebug.debug_snapshot(config=cfg)
    def _boom() -> None:
        raise ValueError("password=secret123")

    with pytest.raises(ValueError, match="password=secret123"):
        _boom()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert "password=secret123" not in snapshot["exception"]["message"]
    assert "[REDACTED]" in snapshot["exception"]["message"]
    assert "password=secret123" not in snapshot["traceback"]
    assert snapshot["capture_config"]["redaction_profile"] == "prod"
    assert snapshot["capture_config"]["redact_traceback"] is True
    assert snapshot["capture_config"]["redact_exception_strings"] is True
    assert not any(key.startswith("_") for key in snapshot["capture_config"])


def test_snapshot_config_explicit_redaction_overrides_profile() -> None:
    resolved = resolve_snapshot_config(
        SnapshotConfig(
            redaction_profile="prod",
            redact=(),
            redact_traceback=False,
            redact_exception_strings=False,
        )
    )

    assert resolved.redact == ()
    assert resolved.redact_traceback is False
    assert resolved.redact_exception_strings is False


def test_snapshot_section_accepts_snapshot_config_without_deprecation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)
    monkeypatch.setattr(llmdebug, "_legacy_capture_kwargs_warning_emitted", set())
    cfg = SnapshotConfig(out_dir=str(tmp_path), redaction_profile="prod")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with (
            pytest.raises(RuntimeError, match="section failed"),
            llmdebug.snapshot_section("stage-x", config=cfg),
        ):
            raise RuntimeError("section failed")

    assert not any(issubclass(item.category, DeprecationWarning) for item in captured)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["name"] == "stage-x"


def test_debug_snapshot_rejects_mixed_config_and_legacy_kwargs() -> None:
    with pytest.raises(TypeError, match=r"received both config=\.\.\. and legacy capture kwargs"):
        llmdebug.debug_snapshot(config=SnapshotConfig(), out_dir=".llmdebug")


def test_legacy_capture_kwargs_warn_once_per_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", False)
    monkeypatch.setattr(llmdebug, "_legacy_capture_kwargs_warning_emitted", set())

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        llmdebug.debug_snapshot(out_dir=".llmdebug", redaction_profile="ci")
        llmdebug.debug_snapshot(out_dir=".llmdebug", redaction_profile="ci")
        with llmdebug.snapshot_section("stage", out_dir=".llmdebug", redaction_profile="ci"):
            pass
        with llmdebug.snapshot_section("stage", out_dir=".llmdebug", redaction_profile="ci"):
            pass

    deprecations = [item for item in captured if issubclass(item.category, DeprecationWarning)]
    assert len(deprecations) == 2


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 118, 130-133, 315-319
# ---------------------------------------------------------------------------


def test_resolve_snapshot_config_both_config_and_legacy_kwargs_raises() -> None:
    """Line 118: config is not None AND legacy_kwargs → TypeError."""
    with pytest.raises(TypeError, match="received both config"):
        llmdebug._resolve_snapshot_config(
            api="test",
            config=SnapshotConfig(),
            legacy_kwargs={"out_dir": "/tmp/foo"},
            stacklevel=2,
        )


def test_resolve_snapshot_config_single_unknown_kwarg_raises() -> None:
    """Lines 130-131: single unknown keyword argument → TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus_key'"):
        llmdebug._resolve_snapshot_config(
            api="test",
            config=None,
            legacy_kwargs={"bogus_key": True},
            stacklevel=2,
        )


def test_resolve_snapshot_config_multiple_unknown_kwargs_raises() -> None:
    """Lines 132-133: multiple unknown keyword arguments → TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        llmdebug._resolve_snapshot_config(
            api="test",
            config=None,
            legacy_kwargs={"foo": 1, "bar": 2},
            stacklevel=2,
        )


def test_getattr_lazy_loads_asgi_middleware() -> None:
    """Lines 315-319: __getattr__ lazy-loads LLMDebugASGIMiddleware."""
    cls = llmdebug.__getattr__("LLMDebugASGIMiddleware")
    assert cls is not None
    assert hasattr(cls, "__init__")


def test_getattr_lazy_loads_wsgi_middleware() -> None:
    """Lines 317-318: __getattr__ lazy-loads LLMDebugWSGIMiddleware."""
    cls = llmdebug.__getattr__("LLMDebugWSGIMiddleware")
    assert cls is not None
    assert hasattr(cls, "__init__")


def test_getattr_invalid_name_raises_attribute_error() -> None:
    """Line 320: __getattr__ with unknown name → AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute"):
        llmdebug.__getattr__("NonExistentThing")


def test_resolve_snapshot_config_raises_when_config_not_snapshot_config() -> None:
    """Line 118, branch [117, 118]: config is not a SnapshotConfig → TypeError."""
    with pytest.raises(TypeError, match="config must be a SnapshotConfig instance"):
        llmdebug._resolve_snapshot_config(
            api="debug_snapshot",
            config="not_a_config",
            legacy_kwargs={},
            stacklevel=2,
        )


def test_resolve_snapshot_config_empty_legacy_kwargs_skips_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branch [135, 138]: legacy_kwargs={} → warning branch skipped, returns SnapshotConfig."""
    monkeypatch.setattr(llmdebug, "_unredacted_default_warning_emitted", True)
    result = llmdebug._resolve_snapshot_config(
        api="debug_snapshot",
        config=None,
        legacy_kwargs={},
        stacklevel=2,
    )
    assert isinstance(result, SnapshotConfig)
