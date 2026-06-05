"""Happy-path tests for the decisions client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest
from memory_mcp_schemas.decisions import AdrExportResponse

pytestmark = pytest.mark.asyncio


async def test_adr_export(client, fake_session) -> None:
    memory_id = uuid4()
    fake_session.set_response(
        "adr_export",
        {
            "markdown": "# Fake ADR\n",
            "status": "accepted",
            "memory_id": str(memory_id),
        },
    )

    out = await client.decisions.adr_export(memory_id=memory_id)

    name, args = fake_session.calls[0]
    assert name == "adr_export"
    assert args["memory_id"] == str(memory_id)
    assert isinstance(out, AdrExportResponse)
