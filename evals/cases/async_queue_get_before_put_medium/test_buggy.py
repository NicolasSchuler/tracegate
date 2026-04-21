from buggy import run_pipeline


def test_consumer_gets_all_items() -> None:
    assert run_pipeline() == [0, 10, 20]
