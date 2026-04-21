from buggy import parse_pipeline


def test_parse_pipeline_basic() -> None:
    chunks = [b"hello", b" ", b"world"]
    result = parse_pipeline("DATA", chunks)
    assert result == {"key": "DATA", "value": "hello world"}
