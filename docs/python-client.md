# Python client

## Python client

The async [`memory-mcp-client`](./client/) package wraps the Streamable-HTTP MCP tools with typed namespaces and shared schemas. It is path-installed next to the server while the package remains unpublished.

```python
async with MemoryClient("http://127.0.0.1:8080/mcp") as client:
    envs = await client.envs.list_()
```

---

