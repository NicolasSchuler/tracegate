"""Reload-based tests to execute module-level initialization paths under coverage."""

from __future__ import annotations

import importlib


def test_reload_core_modules_executes_module_level_code() -> None:
    import llmdebug._json as json_module
    import llmdebug._warnings as warnings_module
    import llmdebug.capture as capture_module
    import llmdebug.cli as cli_module
    import llmdebug.config as config_module
    import llmdebug.detail_filter as detail_filter_module
    import llmdebug.git_context as git_context_module
    import llmdebug.hooks as hooks_module
    import llmdebug.log_capture as log_capture_module
    import llmdebug.output as output_module
    import llmdebug.redaction_policy as redaction_policy_module
    import llmdebug.serialize as serialize_module
    import llmdebug.state_db as state_db_module

    importlib.reload(json_module)
    importlib.reload(capture_module)
    importlib.reload(cli_module)
    importlib.reload(detail_filter_module)
    importlib.reload(config_module)
    importlib.reload(redaction_policy_module)
    importlib.reload(hooks_module)
    importlib.reload(log_capture_module)
    importlib.reload(output_module)
    importlib.reload(serialize_module)
    importlib.reload(git_context_module)
    importlib.reload(state_db_module)
    importlib.reload(warnings_module)

    assert capture_module.capture_exception
    assert cli_module.main
    assert detail_filter_module.filter_snapshot
    assert json_module.dumps
    assert config_module.SnapshotConfig
    assert redaction_policy_module.PII_PATTERNS
    assert hooks_module.PII_PATTERNS
    assert log_capture_module.LLMDebugLogHandler
    assert output_module.write_bundle
    assert state_db_module.state_db_path
    assert warnings_module.emit_warning
    assert serialize_module.serialize_locals
    assert git_context_module.get_git_context
