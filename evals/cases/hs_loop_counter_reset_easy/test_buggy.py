from buggy import RingPicker


def test_ring_picker_wraps_index() -> None:
    picker = RingPicker()
    assert picker.pick(["a", "b"]) == "a"
    assert picker.pick(["a", "b"]) == "b"
    assert picker.pick(["a", "b"]) == "a"
