"""
app/models/database.py — Pool de conexões PostgreSQL via psycopg2.
Fornece get_conn() como context manager para uso seguro com with.
"""
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from config.config import config

# Pool de conexões (mínimo 2, máximo 10 — adequado para uso local)
_pool: pg_pool.ThreadedConnectionPool | None = None


def init_pool() -> None:
    """Inicializa o pool. Chamado uma vez na inicialização do Flask."""
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=config.database.dsn,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )


@contextmanager
def get_conn() -> Generator:
    """
    Context manager que obtém uma conexão do pool e a devolve ao final.

    Uso:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_schema() -> None:
    """
    Aplica o schema.sql caso as tabelas ainda não existam.
    Seguro para ser chamado múltiplas vezes (usa IF NOT EXISTS).
    """
    from pathlib import Path

    schema_path = Path(__file__).resolve().parent.parent.parent / "config" / "schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("[DB] Schema aplicado com sucesso.")