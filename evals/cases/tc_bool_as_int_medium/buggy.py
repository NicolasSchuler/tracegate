def count_passing(scores: list[float], threshold: float = 60.0) -> int:
    """Count how many scores meet or exceed the threshold."""
    return sum(score >= threshold for score in scores)


def passing_percentage(scores: list[float], threshold: float = 60.0) -> str:
    """Return percentage of passing scores as a formatted string like '75.0%'."""
    total = len(scores)
    passed = count_passing(scores, threshold)
    ratio = passed // total * 100
    return f"{ratio}%"
