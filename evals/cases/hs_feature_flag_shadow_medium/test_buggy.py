from buggy import FeatureGate


def test_override_updates_gate_state() -> None:
    gate = FeatureGate()
    assert gate.is_enabled({"flag": False}) is False
    assert gate.is_enabled({"override": True}) is True
