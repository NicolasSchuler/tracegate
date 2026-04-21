from buggy import RunningTotal


def test_cached_total_updates_when_popping_last() -> None:
    tracker = RunningTotal()
    tracker.add(2)
    tracker.add(3)
    assert tracker.total_without_last() == 2
