from buggy import execute


def test_resource_cleaned_after_cancellation() -> None:
    result = execute()
    assert result.get("a_started"), "worker_a should have started"
    assert result.get("a_cleaned"), "worker_a resource should be cleaned up"
    resource = result.get("a_resource", {})
    assert resource.get("active") is False, "resource should be deactivated"
