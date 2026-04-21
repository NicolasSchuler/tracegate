from buggy import execute


def test_query_returns_result() -> None:
    assert execute("SELECT 1") == "result:localhost:SELECT 1"
