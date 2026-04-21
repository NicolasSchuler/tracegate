from buggy import add_event
from cache import get_buffer


def test_each_user_gets_isolated_buffer() -> None:
    add_event("alice", "A1")
    add_event("bob", "B1")
    assert get_buffer("alice") == ["A1"]
    assert get_buffer("bob") == ["B1"]
