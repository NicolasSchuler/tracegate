from buggy import append_event


def test_default_list_not_shared_between_calls() -> None:
    assert append_event("a") == ["a"]
    assert append_event("b") == ["b"]
