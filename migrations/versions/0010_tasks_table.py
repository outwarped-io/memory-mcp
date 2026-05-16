"""Add tasks and task graph nodes.

Revision ID: 0010_tasks_table
Revises: 0009_playbook_kind_and_macro
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010_tasks_table"
down_revision: str | None = "0009_playbook_kind_and_macro"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_GRAPH_NODE_TYPES_WITH_TASK = "('entity','memory','task')"
_GRAPH_NODE_TYPES_WITHOUT_TASK = "('entity','memory')"
_OUTBOX_TYPES_WITH_TASK = "('memory','entity','relation','env','task')"
_OUTBOX_TYPES_WITHOUT_TASK = "('memory','entity','relation','env')"

_GRAPH_EXACTLY_ONE_WITH_TASK = """
(
    (node_type = 'memory' AND memory_id IS NOT NULL AND entity_id IS NULL AND task_id IS NULL)
 OR (node_type = 'entity' AND entity_id IS NOT NULL AND memory_id IS NULL AND task_id IS NULL)
 OR (node_type = 'task' AND task_id IS NOT NULL AND entity_id IS NULL AND memory_id IS NULL)
)
"""

_GRAPH_EXACTLY_ONE_WITHOUT_TASK = """
(
    (node_type = 'memory' AND memory_id IS NOT NULL AND entity_id IS NULL)
 OR (node_type = 'entity' AND entity_id IS NOT NULL AND memory_id IS NULL)
)
"""


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tasks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            title text NOT NULL,
            description text NULL,
            status text NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','in_progress','blocked','done','cancelled')),
            priority int NOT NULL DEFAULT 50 CHECK (priority BETWEEN 1 AND 100),
            playbook_id uuid NULL REFERENCES memories(id) ON DELETE SET NULL,
            version int NOT NULL DEFAULT 1,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            created_by_agent_id uuid NULL REFERENCES agents(id) ON DELETE SET NULL,
            UNIQUE(id, env_id)
        )
    """)
    op.execute("CREATE INDEX tasks_env_status_priority_created_idx ON tasks(env_id, status, priority, created_at)")
    op.execute("CREATE INDEX tasks_env_playbook_idx ON tasks(env_id, playbook_id) WHERE playbook_id IS NOT NULL")
    op.execute("CREATE INDEX tasks_updated_desc_idx ON tasks(updated_at DESC)")

    op.execute("ALTER TABLE graph_nodes ADD COLUMN task_id uuid NULL REFERENCES tasks(id) ON DELETE CASCADE")
    op.execute("ALTER TABLE graph_nodes DROP CONSTRAINT IF EXISTS graph_nodes_exactly_one_target_chk")
    op.execute("ALTER TABLE graph_nodes DROP CONSTRAINT IF EXISTS graph_nodes_node_type_check")
    op.execute(
        "ALTER TABLE graph_nodes ADD CONSTRAINT graph_nodes_exactly_one_target_chk "
        f"CHECK {_GRAPH_EXACTLY_ONE_WITH_TASK}"
    )
    op.execute(
        "ALTER TABLE graph_nodes ADD CONSTRAINT graph_nodes_node_type_check "
        f"CHECK (node_type IN {_GRAPH_NODE_TYPES_WITH_TASK})"
    )
    op.execute(
        "CREATE UNIQUE INDEX graph_nodes_task_uniq "
        "ON graph_nodes(env_id, task_id) WHERE node_type = 'task'"
    )

    op.execute("ALTER TABLE outbox DROP CONSTRAINT IF EXISTS outbox_aggregate_type_check")
    op.execute(
        "ALTER TABLE outbox ADD CONSTRAINT outbox_aggregate_type_check "
        f"CHECK (aggregate_type IN {_OUTBOX_TYPES_WITH_TASK})"
    )


def downgrade() -> None:
    op.execute("""
        DELETE FROM relations r
        USING graph_nodes src, graph_nodes dst
        WHERE r.src_node_id = src.id
          AND r.dst_node_id = dst.id
          AND (src.node_type = 'task' OR dst.node_type = 'task')
    """)
    op.execute("DELETE FROM graph_nodes WHERE node_type = 'task'")
    op.execute("DROP INDEX IF EXISTS graph_nodes_task_uniq")
    op.execute("ALTER TABLE graph_nodes DROP CONSTRAINT IF EXISTS graph_nodes_exactly_one_target_chk")
    op.execute("ALTER TABLE graph_nodes DROP CONSTRAINT IF EXISTS graph_nodes_node_type_check")
    op.execute("ALTER TABLE graph_nodes DROP COLUMN IF EXISTS task_id")
    op.execute(
        "ALTER TABLE graph_nodes ADD CONSTRAINT graph_nodes_exactly_one_target_chk "
        f"CHECK {_GRAPH_EXACTLY_ONE_WITHOUT_TASK}"
    )
    op.execute(
        "ALTER TABLE graph_nodes ADD CONSTRAINT graph_nodes_node_type_check "
        f"CHECK (node_type IN {_GRAPH_NODE_TYPES_WITHOUT_TASK})"
    )

    op.execute("DELETE FROM outbox_delivery od USING outbox o WHERE od.event_id = o.event_id AND o.aggregate_type = 'task'")
    op.execute("DELETE FROM outbox WHERE aggregate_type = 'task'")
    op.execute("ALTER TABLE outbox DROP CONSTRAINT IF EXISTS outbox_aggregate_type_check")
    op.execute(
        "ALTER TABLE outbox ADD CONSTRAINT outbox_aggregate_type_check "
        f"CHECK (aggregate_type IN {_OUTBOX_TYPES_WITHOUT_TASK})"
    )
    op.execute("DROP TABLE IF EXISTS tasks")
