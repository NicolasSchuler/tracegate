def read_chunks(raw_chunks: list[bytes]) -> bytes:
    """Concatenate raw byte chunks into a single payload."""
    return b"".join(raw_chunks)


def build_message(header: str, raw_chunks: list[bytes]) -> str:
    """Build a text message: header followed by decoded payload."""
    payload = read_chunks(raw_chunks)
    return header + ": " + payload


def parse_pipeline(header: str, raw_chunks: list[bytes]) -> dict[str, str]:
    """Full pipeline: build message, then split into key-value pair."""
    msg = build_message(header, raw_chunks)
    key, _, value = msg.partition(": ")
    return {"key": key, "value": value}
