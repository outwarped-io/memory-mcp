---
name: Bug report
about: A concrete, reproducible defect in memory-mcp
title: "[BUG] "
labels: ["bug"]
---

### What happened?

<!-- One paragraph. What did you observe? What did you expect? -->

### Reproduction

```bash
# Minimal steps. Include the failing MCP tool call (tools/call payload),
# curl against /healthz or /readyz, or a unit/integration test invocation.
```

### Environment

- memory-mcp version (tag or commit):
- Install mode: [ ] `/plugin install`  [ ] manual `git clone` + `docker compose up`
- Host OS / arch:
- Docker / Docker Compose versions:
- LLM backend (if applicable): [ ] OpenAI  [ ] Azure OpenAI  [ ] none / sentence-transformers only

### Logs / diagnostics

<details>
<summary>docker compose logs --tail=200 server</summary>

```
```

</details>

<details>
<summary>/readyz output</summary>

```json
```

</details>

### Additional context

<!-- Anything else worth knowing — recent config changes, related issues, suspected component. -->
