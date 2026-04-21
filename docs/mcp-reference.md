# MCP Reference

Protocol-level reference for the `llmdebug` MCP server. Use this document when
you are integrating a client, validating JSON contracts, or handling errors
programmatically. For installation, workflow guidance, and tool overview, start
with the [README MCP section](../README.md#mcp-server).

## JSON Envelope Schema

Evidence tools default to `response_format="json"` and return a structured
envelope for both success and error cases.

### Success envelope

```json
{
  "tool": "llmdebug_diagnose",
  "format_version": "2.0",
  "evidence": {
    "exception": {"type": "", "message": "", "category": "", "suggestion": ""},
    "crash_frame": {"index": 0, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "ok"
  }
}
```

### Error envelope

When `response_format="json"`, errors also use a JSON envelope (instead of plain text):

```json
{
  "tool": "llmdebug_frame",
  "format_version": "2.0",
  "evidence": {
    "summary": "Frame index 99 out of range. Available frames: 0-1 (2 total).",
    "exception": {"type": "", "message": ""},
    "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "error"
  },
  "error": {
    "code": "FRAME_OUT_OF_RANGE",
    "message": "Frame index 99 out of range. Available frames: 0-1 (2 total)."
  }
}
```

Text mode (`response_format="text"`) returns plain-text error messages.
For JSON errors, `metadata.with_rca` reflects the request, while `rca` is omitted when the tool fails.
`metadata.evidence_schema` is always normalized to `summary` or `full` (`summary` fallback on invalid input).

## Contract Notes

- Evidence tools stay evidence-first by default: compact payloads,
  `with_rca=false`, and `summary` schema unless the caller opts in to more.
- JSON clients should branch on `metadata.status` and `error.code` rather than
  parsing free-form error text.
- Text mode is retained for interactive compatibility; JSON mode is the safer
  default for programmatic consumers.

## Parameter Glossary

| Parameter | Type | Description |
|---|---|---|
| `raw_session` | `bool` | Return the raw DebugSession envelope instead of filtered evidence |
| `with_rca` | `bool` | Include RCA state alongside snapshot data |
| `gate_mode` | enum | `off`, `soft`, or `strict` — controls RCA gate strictness (whether advancing requires evidence) |
| `exploratory` | `bool` | Mark the request as exploratory (affects RCA tracking) |
| `evidence_schema` | enum | `summary` or `full` — controls evidence payload detail level |
| `detail` | enum | `crash`, `full`, or `context` — layered disclosure level for snapshot content |
| `response_format` | enum | `text` or `json` — output format (JSON recommended for programmatic consumers) |

## Notable Parameters

- `llmdebug_show(raw_session=true)` returns the raw DebugSession envelope.
- `llmdebug_show(with_rca=true)` returns `{snapshot, rca}` JSON.
- `llmdebug_diagnose` / `llmdebug_show` require `detail` in `crash|full|context`.
  Invalid values return `INVALID_ARGUMENT` with JSON `details`:
  `{argument, value, allowed_values}` (or plain text in `response_format="text"`).
- `llmdebug_list` / `llmdebug_rca_status` / `llmdebug_rca_history` / `llmdebug_rca_advance`
  accept `response_format="text|json"` (default: `text`) and `evidence_schema="summary|full"`.
- `gate_mode=off|soft|strict` and `exploratory=true|false` are available on RCA-aware tools.

## Error Handling Examples

All examples below use `response_format="json"`. Text mode returns the
`error.message` string as plain text instead.

### No snapshots found

Returned by `llmdebug_list`, `llmdebug_diagnose`, or `llmdebug_show` when
the `.llmdebug` directory is empty or missing.

```json
{
  "tool": "llmdebug_list",
  "format_version": "2.0",
  "evidence": {
    "summary": "No snapshots found in .llmdebug.",
    "exception": {"type": "", "message": ""},
    "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "error"
  },
  "error": {
    "code": "NO_SNAPSHOTS",
    "message": "No snapshots found in .llmdebug."
  }
}
```

### Frame index out of range

Returned by `llmdebug_frame` when the requested frame index exceeds the
available frames in the snapshot.

```json
{
  "tool": "llmdebug_frame",
  "format_version": "2.0",
  "evidence": {
    "summary": "Frame index 99 out of range. Available frames: 0-4 (5 total).",
    "exception": {"type": "", "message": ""},
    "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "error"
  },
  "error": {
    "code": "FRAME_OUT_OF_RANGE",
    "message": "Frame index 99 out of range. Available frames: 0-4 (5 total)."
  }
}
```

### Invalid argument

Returned when a parameter value is not in the allowed set (e.g. a bad
`detail` or `response_format` value). The `details` object identifies the
offending argument.

```json
{
  "tool": "llmdebug_diagnose",
  "format_version": "2.0",
  "evidence": {
    "summary": "Invalid value for response_format: \"xml\". Allowed values: [\"json\", \"text\"].",
    "exception": {"type": "", "message": ""},
    "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "error"
  },
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "Invalid value for response_format: \"xml\". Allowed values: [\"json\", \"text\"].",
    "details": {
      "argument": "response_format",
      "value": "xml",
      "allowed_values": ["json", "text"]
    }
  }
}
```

### Corrupted or unreadable snapshot

Returned when a snapshot file exists but cannot be parsed or is missing
required data (e.g. a truncated JSON file or a snapshot with no frames).

```json
{
  "tool": "llmdebug_frame",
  "format_version": "2.0",
  "evidence": {
    "summary": "No frames in snapshot.",
    "exception": {"type": "", "message": ""},
    "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
    "locals_summary": [],
    "repro": {"argv": [], "nodeid": ""},
    "evidence_ids": [],
    "sections": []
  },
  "metadata": {
    "response_format": "json",
    "with_rca": false,
    "evidence_schema": "summary",
    "status": "error"
  },
  "error": {
    "code": "MALFORMED_SNAPSHOT",
    "message": "No frames in snapshot."
  }
}
```

Clients should always branch on `metadata.status` first, then inspect
`error.code` for programmatic handling. The `error.message` field is
human-readable and may change between versions; `error.code` is stable.
