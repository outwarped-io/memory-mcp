# Contributing to memory-mcp

Thanks for taking the time to look. The project is currently maintained by
**outwarped-io contributors**; external contributions are welcome under the
guidelines below.

## Ground rules

- The project is **MIT-licensed** (see [LICENSE](LICENSE)). By submitting a
  pull request you agree your contribution may be redistributed under that
  licence.
- **v1 is LOCAL-ONLY.** Do not propose features that assume multi-tenant
  auth, public-internet exposure, or untrusted client isolation without
  first opening an issue to discuss the scope.
- **No secrets in the repo.** Tokens, real customer data, `.env` files, and
  anything subject to a content-exclusion policy must stay out of commits.
- **Backwards compatibility matters.** Existing MCP tool surfaces and their
  response shapes are part of the contract. Breaking changes need an issue,
  a design discussion, and a `CHANGELOG.md` migration note.

## Development setup

```bash
git clone https://github.com/outwarped-io/memory-mcp.git
cd memory-mcp
cp .env.example .env             # set LLM_API_KEY if you intend to exercise LLM features

# Bring up the full stack
docker compose up -d
docker compose ps                # postgres / qdrant / neo4j / server / projection-worker healthy

# Verify
curl -s localhost:8080/healthz | jq
curl -s localhost:8080/readyz | jq
```

For a local Python environment (lint / unit tests only — no docker
required):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

## What to run before opening a PR

Required:

```bash
ruff check .
ruff format --check .
mypy --strict src/memory_mcp
pytest -m "not integration"
```

Recommended (require the docker-compose stack):

```bash
pytest -m integration
```

## Commit / PR conventions

- One logical change per PR. Stack PRs if a feature lands in waves.
- Conventional-style commit subjects are appreciated (`feat:`, `fix:`,
  `chore:`, `docs:`, `refactor:`, `test:`).
- Update `CHANGELOG.md` under `## [Unreleased]` for any user-visible
  change. Release-cutting moves the `Unreleased` section to a dated
  version header.
- AI-assisted contributions are welcome. If a coding agent authored part
  of the change, keep the `Co-authored-by: Copilot
  <223556219+Copilot@users.noreply.github.com>` trailer on the relevant
  commits.

## Reporting bugs / requesting features

Open an issue using one of the templates in `.github/ISSUE_TEMPLATE/`.
For suspected security vulnerabilities, follow [SECURITY.md](SECURITY.md)
instead of filing a public issue.
