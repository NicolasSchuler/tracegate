from buggy import Metrics, bump


def test_dataclass_default_dict_uses_factory() -> None:
    metrics = Metrics()
    assert bump(metrics, "ok") == 1
