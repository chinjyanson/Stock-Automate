import os
import asyncio
import asyncpg
import psycopg2
from typing import List, Dict, Any, Optional

CONNECTION = os.environ.get("TIMESCALEDB_CONNECTION_STRING")

# Asynchronous Utils using asyncpg
async def async_connect() -> asyncpg.Connection:
    """Establish an asynchronous connection to TimescaleDB."""
    if not CONNECTION:
        raise ValueError("TIMESCALEDB_CONNECTION_STRING is not set.")
    return await asyncpg.connect(CONNECTION)

async def async_execute(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a query asynchronously.
    Args:
        query (str): The SQL query to execute.
        params (Optional[List[Any]]): Parameters for the query.
    Returns:
        List[Dict[str, Any]]: Query results as a list of dictionaries.
    """
    conn = await async_connect()
    try:
        rows = await conn.fetch(query, *(params or []))
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def async_insert(query: str, params: List[Any]) -> None:
    """
    Insert data asynchronously.
    Args:
        query (str): The SQL INSERT query.
        params (List[Any]): Parameters for the query.
    """
    conn = await async_connect()
    try:
        await conn.execute(query, *params)
    finally:
        await conn.close()


# Synchronous Utils using psycopg2
def sync_connect() -> psycopg2.extensions.connection:
    """Establish a synchronous connection to TimescaleDB."""
    if not CONNECTION:
        raise ValueError("TIMESCALEDB_CONNECTION_STRING is not set.")
    return psycopg2.connect(CONNECTION)


def sync_execute(query: str, params: Optional[List[Any]] = None) -> List[Any]:
    """
    Execute a query synchronously.
    Args:
        query (str): The SQL query to execute.
        params (Optional[List[Any]]): Parameters for the query.
    Returns:
        List[Any]: Query results.
    """
    conn = sync_connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or [])
            if cursor.description:  # For SELECT queries
                return cursor.fetchall()
    finally:
        conn.close()

def sync_insert(query: str, params: List[Any]) -> None:
    """
    Insert data synchronously.
    Args:
        query (str): The SQL INSERT query.
        params (List[Any]): Parameters for the query.
    """
    conn = sync_connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()
    finally:
        conn.close()
