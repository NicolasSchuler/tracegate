from buggy import run_coordination


def test_both_waiters_notified() -> None:
    assert run_coordination() == ["A:done", "B:done"]
