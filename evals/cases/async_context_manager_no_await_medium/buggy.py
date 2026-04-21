import asyncio
from typing import Any


async def connect_db() -> dict[str, Any]:
    await asyncio.sleep(0)
    return {"host": "localhost", "connected": True}


class DatabaseSession:
    def __init__(self) -> None:
        self.conn: dict[str, Any] | None = None

    async def __aenter__(self) -> "DatabaseSession":
        self.conn = connect_db()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.conn = None

    def query(self, sql: str) -> str:
        return f"result:{self.conn['host']}:{sql}"


async def run_query(sql: str) -> str:
    async with DatabaseSession() as db:
        return db.query(sql)


def execute(sql: str) -> str:
    return asyncio.run(run_query(sql))
