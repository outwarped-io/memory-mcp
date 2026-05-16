"""Alembic env (Phase 1).

Imports ``memory_mcp.db.models.Base.metadata`` so future ``alembic revision
--autogenerate`` runs see the ORM definitions. The Phase 1 ``0001_v1_initial``
migration is hand-authored; subsequent migrations may use autogenerate.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from memory_mcp.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Always normalize to a sync driver — alembic uses the synchronous engine even
# though the application uses asyncpg. The .ini file ships with a +asyncpg
# URL for documentation symmetry; we strip it here regardless of the source.
env_url = os.environ.get("POSTGRES_URL")
ini_url = config.get_main_option("sqlalchemy.url")
raw_url = env_url or ini_url or ""
if raw_url:
    config.set_main_option("sqlalchemy.url", raw_url.replace("+asyncpg", ""))


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

