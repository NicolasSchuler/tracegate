"""Pull unified diffs out of model text and repair their hunk headers."""

from __future__ import annotations

import re

from baselines.openai_compat import strip_think
from evals.eval_patches import _diff_lines


def _extract_unified_diff(text: str) -> str | None:
    """Extract every file diff from a model response as one git-style patch.

    Review findings LLM-01/03/04/07/10: fenced and unfenced sections are scanned in
    response order and all file sections are
    collected (a later diff for the same file replaces an earlier draft), hunks are read
    with their declared counts so trailing prose isn't absorbed, blank context lines are
    normalized to ``" "``, and ``/dev/null`` headers create or delete files.
    """
    s = _strip_think(text).strip("\n")
    if not s.strip():
        return None
    sections, fenced = _response_sections(s)
    by_path: dict[str, str] = {}
    for source in sections:
        for path, block in _iter_file_diffs(source.split("\n")):
            by_path.pop(path, None)  # keep the last version, in the order it appeared
            by_path[path] = block
    if not by_path and fenced:
        for path, block in _iter_file_diffs(s.split("\n")):
            by_path.pop(path, None)
            by_path[path] = block
    if not by_path:
        return None
    return "".join(by_path.values())


def _strip_think(text: str) -> str:
    return strip_think(text)


_FENCE_RE = re.compile(r"```(?:diff|patch)?[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _response_sections(text: str) -> tuple[list[str], bool]:
    """Split a response into fenced blocks and the text around them, in response order.

    Collecting only the fenced blocks dropped an unfenced diff that followed a fenced
    one and kept a fenced draft that a later unfenced revision replaced (LLM-01/LLM-04).
    The second element reports whether any fence was found, for the whole-response
    fallback scan of diffs that straddle a fence.
    """
    sections: list[str] = []
    cursor = 0
    fenced = False
    for match in _FENCE_RE.finditer(text):
        fenced = True
        outside = text[cursor : match.start()]
        if outside.strip():
            sections.append(outside)
        sections.append(match.group(1))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        sections.append(tail)
    return (sections or [text]), fenced


_DIFF_META_PREFIXES = (
    "index ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)


_MARKDOWN_BULLET_RE = re.compile(r"^[-*+] +(?:[A-Z`*_\[(]|\d+\.)")


def _is_file_header_pair(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].startswith("--- ")
        and lines[index + 1].startswith("+++ ")
    )


def _iter_file_diffs(lines: list[str]) -> list[tuple[str, str]]:
    """Return ``(path, git-style block)`` for every file section found in *lines*."""
    out: list[tuple[str, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        git_header = line.startswith("diff --git ")
        if not git_header and not _is_file_header_pair(lines, i):
            i += 1
            continue
        j = i + 1 if git_header else i
        meta: list[str] = []
        while j < n and lines[j].startswith(_DIFF_META_PREFIXES):
            meta.append(lines[j])
            j += 1
        if not _is_file_header_pair(lines, j):
            i += 1
            continue
        old_path, new_path = _strip_git_prefixes(
            _clean_path_token(lines[j][4:].strip()),
            _clean_path_token(lines[j + 1][4:].strip()),
        )
        if old_path is None or new_path is None:
            i = j + 2
            continue
        hunks, j = _read_hunks(lines, j + 2)
        if hunks:
            out.append(_git_file_block(old_path, new_path, meta, hunks))
        i = max(j, i + 1)
    return out


def _git_file_block(
    old_path: str, new_path: str, meta: list[str], hunks: list[str]
) -> tuple[str, str]:
    if old_path == "/dev/null":  # new file (LLM-10)
        path = new_path
        header = [f"diff --git a/{path} b/{path}", *meta, "--- /dev/null", f"+++ b/{path}"]
    elif new_path == "/dev/null":  # deleted file
        path = old_path
        header = [f"diff --git a/{path} b/{path}", *meta, f"--- a/{path}", "+++ /dev/null"]
    else:
        # Avoid accidental renames in model output (common mistake).
        path = old_path
        header = [f"diff --git a/{path} b/{path}", *meta, f"--- a/{path}", f"+++ b/{path}"]
    return path, "\n".join([*header, *hunks]) + "\n"


def _blank_run_continues_hunk(lines: list[str], index: int) -> bool:
    """Whether the run of blank lines at *index* is followed by more hunk body.

    Several blank context lines in a row (LLM-03) must not end extraction: looking only
    at the next line made the first blank of a pair terminate the hunk and silently drop
    every change after it. The whole run is skipped and the first non-blank line decides.
    """
    probe = index
    while probe < len(lines) and lines[probe] == "":
        probe += 1
    following = lines[probe] if probe < len(lines) else ""
    return following.startswith((" ", "+", "-", "\\"))


def _read_hunks(lines: list[str], start: int) -> tuple[list[str], int]:
    """Read the hunks of one file section; return them and the index after the section."""
    hunks: list[str] = []
    i = start
    n = len(lines)
    while i < n and lines[i].startswith("@@"):
        match = _HUNK_RE.match(lines[i])
        # An omitted count means 1; a header that doesn't parse gives no count to trust.
        want_old = int(match.group(2) if match.group(2) is not None else 1) if match else None
        want_new = int(match.group(4) if match.group(4) is not None else 1) if match else None
        hunks.append(lines[i])
        i += 1
        old_seen = new_seen = 0
        while i < n:
            body = lines[i]
            counts_met = (
                want_old is not None
                and want_new is not None
                and old_seen >= want_old
                and new_seen >= want_new
            )
            if (
                body.startswith("@@")
                or body.startswith("diff --git ")
                or _is_file_header_pair(lines, i)
            ):
                break
            if body == "":
                # A blank line inside a hunk is a context line whose space was lost (LLM-03).
                if counts_met or not _blank_run_continues_hunk(lines, i):
                    break
                body = " "
            elif body.startswith(("+", "-")) and counts_met and _MARKDOWN_BULLET_RE.match(body):
                break  # trailing markdown bullets after a complete hunk (LLM-07)
            elif not body.startswith((" ", "+", "-", "\\")):
                break
            if body.startswith(" "):
                old_seen += 1
                new_seen += 1
            elif body.startswith("-"):
                old_seen += 1
            elif body.startswith("+"):
                new_seen += 1
            hunks.append(body)
            i += 1
        while i < n and lines[i] == "" and i + 1 < n and lines[i + 1].startswith("@@"):
            i += 1
    return hunks, i


def _clean_path_token(raw: str) -> str | None:
    if not raw:
        return None
    token = raw.strip()
    if not token:
        return None
    if "\t" in token:
        token = token.split("\t", 1)[0].rstrip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        token = token[1:-1]
    token = re.sub(r"\s+(?:---|\+\+\+)$", "", token).rstrip()

    # Common weird outputs:
    # - "--- buggy.py ---" (extra trailing token)
    # - "--- a/foo.py\t2026-..." (timestamps)
    if token in {"a/", "b/"}:
        return None
    return token


def _strip_git_prefixes(old: str | None, new: str | None) -> tuple[str | None, str | None]:
    """Drop git's ``a/`` and ``b/`` prefixes only when the headers use them as a pair.

    Stripping each side unconditionally mangled a real top-level directory named ``a``
    or ``b`` in prefix-less headers such as ``--- a/mod.py`` / ``+++ a/mod.py`` (LLM-23).
    """
    if old is None or new is None:
        return old, new
    old_git = old == "/dev/null" or old.startswith("a/")
    new_git = new == "/dev/null" or new.startswith("b/")
    if not (old_git and new_git):
        return old, new
    old = old.removeprefix("a/")
    new = new.removeprefix("b/")
    return (old or None), (new or None)


def _looks_like_raw_unified_diff(text: str) -> bool:
    return _extract_unified_diff(text) is not None


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _fix_hunk_counts(diff: str) -> str:
    """Make unified diff hunks syntactically consistent.

    LLMs often emit correct hunk bodies but incorrect `@@ -l,s +l,s @@` counts.
    Git treats those diffs as corrupt. We recompute the counts from the hunk body
    and rewrite the header counts, keeping the start line numbers as-is.
    """
    lines = _diff_lines(diff)  # not splitlines(): form feeds etc. stay in their line (LLM-15)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _HUNK_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        old_start = int(m.group(1))
        new_start = int(m.group(3))
        suffix = m.group(5) or ""

        # Collect hunk body.
        body: list[str] = []
        i += 1
        while i < n:
            nxt = lines[i]
            if nxt.startswith("diff --git ") or nxt.startswith("@@ "):
                break
            if nxt.startswith("--- a/") or nxt.startswith("+++ b/"):
                break
            body.append(nxt)
            i += 1
        # LLM-03: trailing blank lines separate sections; interior ones are context lines
        # whose leading space was lost and must be counted.
        trailing_blank: list[str] = []
        while body and body[-1] == "":
            trailing_blank.append(body.pop())
        body = [" " if b == "" else b for b in body]

        old_count = 0
        new_count = 0
        for b in body:
            c = b[0]
            if c == " ":
                old_count += 1
                new_count += 1
            elif c == "-":
                old_count += 1
            elif c == "+":
                new_count += 1
            else:
                # "\ No newline..." or other metadata does not affect counts.
                pass

        # A side with no lines starts at 0 (file creation or deletion, LLM-10); otherwise
        # line numbers are 1-based.
        old_start = old_start if old_count == 0 else max(1, old_start)
        new_start = new_start if new_count == 0 else max(1, new_start)
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}")
        out.extend(body)
        out.extend(trailing_blank)

    return "\n".join(out).rstrip("\n") + "\n"
