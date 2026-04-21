from buggy import run


def test_pipeline_produces_doubled_values() -> None:
    assert run() == [0, 2, 4]
