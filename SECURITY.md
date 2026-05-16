# Security Policy

## Supported versions

Only the latest tagged release on `main` is actively supported. Older
tags receive backports on a best-effort basis if the fix is small.

| Version | Supported          |
|---------|--------------------|
| Latest `v0.x` | ✅ |
| Older `v0.x` | ⚠️ Best-effort |
| Pre-v0.10    | ❌ |

## ⚠️ v1 is LOCAL-ONLY

memory-mcp v1 has **no authentication** on its MCP HTTP surface. The
docker-compose stack binds the server port to `127.0.0.1` by default,
which is the only enforcement against external access. Do not:

- Expose the MCP port to a non-loopback interface.
- Run memory-mcp on a multi-tenant or shared host without your own
  network-level isolation.
- Trust agent-supplied `X-Agent-Id` / `X-Agent-Name` headers as a
  security boundary — they are agent identity for accounting and
  attribution, not for authorisation.

If you need multi-user auth, please open a feature-request issue rather
than treating the current surface as suitable.

## Reporting a vulnerability

**Do not file a public GitHub issue for security reports.**

Instead, use **GitHub's [Private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)**
on this repository (Security tab → Report a vulnerability). The
maintainers will:

1. Acknowledge receipt as soon as practical.
2. Investigate and confirm scope.
3. Coordinate a fix and a coordinated-disclosure window.
4. Credit the reporter in the advisory unless they prefer anonymity.

For reports that fall outside the v1 LOCAL-ONLY threat model (i.e.
issues that only manifest when a deployment intentionally violates the
above), maintainers may publish an advisory marked "won't fix in v1"
and route the work to v2.
