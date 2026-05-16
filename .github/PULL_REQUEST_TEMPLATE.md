<!-- Thanks for contributing to memory-mcp! Please fill in what's relevant. -->

### What this PR does

<!-- One paragraph. Lead with the user-visible change. -->

### Why

<!-- Link to the issue, the design doc, or the rationale. -->

### How to verify

```bash
# Commands a reviewer can run locally to convince themselves the change works.
```

### Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy --strict src/memory_mcp` passes (or call out the deferred typing)
- [ ] Unit tests (`pytest -m "not integration"`) pass
- [ ] Integration tests run locally against the docker-compose stack (if applicable)
- [ ] `CHANGELOG.md` updated (under `## [Unreleased]` or the next version section)
- [ ] No new top-level dependency without explicit justification in the PR body
- [ ] No secrets, tokens, real customer data, or `.env` files committed

### Related issues

<!-- Closes #X, Refs #Y -->
