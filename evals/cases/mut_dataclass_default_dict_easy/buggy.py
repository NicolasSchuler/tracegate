from dataclasses import dataclass


@dataclass
class Metrics:
    counters: dict[str, int] = {}


def bump(metrics: Metrics, name: str) -> int:
    metrics.counters[name] = metrics.counters.get(name, 0) + 1
    return metrics.counters[name]
