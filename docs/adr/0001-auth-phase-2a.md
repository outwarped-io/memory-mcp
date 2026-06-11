# ADR 0001 — Phase 2a Auth Design

- **Status:** Accepted (2026-06-06)
- **Phase:** 2a (single-tenant, optional auth, k8s-native deployment)
- **Supersedes:** none
- **Superseded-by:** none
- **Related work:** Phase 2b (multi-tenant + MCP-spec OAuth 2.1 + DCR) — deferred
- **Authors:** memory-mcp maintainers

## Context

Through v0.17.2, memory-mcp ships as a local-only service. The MCP HTTP endpoint at `/mcp/` accepts unauthenticated requests, an `agent-<uuid>` row is auto-created on first contact, and that synthetic identity owns everything the caller writes. This is fine for a single developer's laptop and for the multi-CLI-on-one-host workflow we already use, but it does not generalize to:

- A shared deployment where multiple human operators talk to one memory-mcp instance.
- Pod-to-pod traffic where the *caller's* identity needs to be auditable (not just "an agent on this host").
- Any production-shaped environment where the data path must reject anonymous requests by default.

Phase 2 introduces optional authentication. Phase 2a is the **single-tenant** slice: one logical deployment, one IdP, one per-environment ACL surface, all backing stores in-cluster. Phase 2b will handle multi-tenant routing and MCP-spec OAuth 2.1 + Dynamic Client Registration; it is explicitly out of scope here.

### Constraints

1. **Backward compatibility is non-negotiable.** Existing local-dev flows (`docker compose up`, CI smoke tests, the stdio bridge against `http://127.0.0.1:8080/mcp/`) must continue working with no config change. This means an explicit "auth off" mode that the server logs loudly but does not refuse.
2. **The deployment target is k8s.** AKS is the validating target, but nothing in the design may be AKS-specific. The same chart must install on GKE / EKS / kind / k3s with a values override.
3. **No Azure-specific infrastructure.** Backing stores (Postgres / Qdrant / Neo4j) must run in-cluster. The team has explicitly rejected Azure Database for PostgreSQL Flexible Server, Qdrant Cloud, Neo4j Aura, Azure Key Vault, Managed Identity / Workload Identity, and external-secrets-operator for v1. Entra ID is the *only* Azure surface we accept, and only as an inbound identity provider.
4. **Entra ID must be one of many.** The bearer-token validator must work against any compliant OIDC provider (Keycloak, Auth0, Okta, Google, …). Entra-specific configuration is a *preset* over the generic shape, not a parallel code path.
5. **Pod-to-pod credentials are weak.** Postgres / Qdrant / Neo4j use shared secrets (connection strings, API keys, basic auth) that any pod with the secret can use. That means *NetworkPolicy* is the tenant-isolation primitive in v1, not credential scoping. The Helm chart must make this load-bearing.
6. **The ACL story has to be simple.** Per-env roles, manageable through MCP tools, bootstrapped automatically. Anything more elaborate (RBAC inheritance, policy languages, attribute-based access) is deferred.

## Decision

Phase 2a ships **five coordinated artifacts**:

1. **A new server release** with three `AUTH_MODE` values and an OIDC bearer-token validator on `/mcp/` only.
2. **An additive schema migration** (`0023_auth_phase2a`) introducing per-env ACLs and a nullable audit column.
3. **An updated stdio bridge** that mints / caches tokens via device-code flow.
4. **A Helm chart** that packages memory-mcp + the three backing stores + NetworkPolicy + Secret templates as a single `helm install`.
5. **Operator documentation** + an end-to-end smoke test that proves the whole stack works against a real IdP.

### 1. Inbound authentication — `AUTH_MODE` ∈ {`none`, `oidc`, `entra`}

A new module `src/memory_mcp/auth/oidc.py` exposes a `require_authenticated_principal` FastAPI dependency. It is mounted on `/mcp/` and nowhere else:

| Path | Auth required? |
|---|---|
| `/mcp/` | yes (when `AUTH_MODE != none`) |
| `/healthz` | no — kubelet probe must not depend on IdP availability |
| `/readyz` | no — same; response body includes `auth_mode` so operators can see it |
| `/metrics` (if/when added) | no — Prometheus scrape uses NetworkPolicy, not bearer auth |

**Three modes:**

- `AUTH_MODE=none` (default; backward-compat) — passthrough. The synthetic `agent-<uuid>` model from v0.17.x stays unchanged. On startup the server logs `WARN: AUTH_MODE=none — running in LOCAL-ONLY mode; do not expose externally`. `/readyz` echoes `auth_mode: none` so operators can monitor.
- `AUTH_MODE=oidc` (generic) — validates a bearer JWT against an arbitrary OIDC issuer. Config keys: `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL` (optional override; defaults to `{ISSUER}/.well-known/jwks.json`).
- `AUTH_MODE=entra` (preset) — sugar over `oidc`. Config keys: `ENTRA_TENANT_ID`, `ENTRA_APP_ID`. The validator computes `OIDC_ISSUER=https://login.microsoftonline.com/{ENTRA_TENANT_ID}/v2.0`, `OIDC_AUDIENCE={ENTRA_APP_ID}` (or `api://{ENTRA_APP_ID}`), and the standard Entra JWKS URL.

The JWT validator extracts a **principal id** from the token: Entra's `oid` claim (stable per-user), generic OIDC's `sub` claim. That value is stored on the request-scoped `AgentContext.principal_id`; downstream code uses it for ACL checks and audit.

**JWKS cache** lives in-process. Refresh policy: TTL (default 1h) + ETag conditional GET on refresh + stale-while-revalidate (default 1h beyond TTL if the upstream is down, logged WARN). This keeps memory-mcp from going dark if the IdP's JWKS endpoint blips.

### 2. Outbound credentials — K8s Secrets, native connection strings

memory-mcp consumes backing-store credentials the same way it does today: environment variables loaded into `Settings` at startup. The Helm chart materializes them as a single K8s `Secret` per backing store and mounts the values via `envFrom`:

| Secret key | Consumer |
|---|---|
| `DATABASE_URL` | SQLAlchemy → in-cluster Postgres StatefulSet |
| `QDRANT_API_KEY` | Qdrant client → in-cluster Qdrant StatefulSet |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j driver → in-cluster Neo4j StatefulSet |

No Workload Identity, no CSI driver, no external-secrets-operator. Rotation is operator-owned: `kubectl create secret` + `kubectl rollout restart`. We accept this v1 caveat explicitly and document it in `docs/auth.md` with a TODO `memory-mcp-external-secrets-integration` for v2.

The server logs a startup WARN if any required secret is empty (rather than crashing late on first DB call).

### 3. Authorization — per-env ACLs in `env_acls`

A new table:

```
env_acls (
  env_id        UUID NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
  principal_id  TEXT NOT NULL,
  role          TEXT NOT NULL,
  granted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  granted_by    TEXT NOT NULL,
  PRIMARY KEY (env_id, principal_id)
)
```

`role` is one of `admin` / `writer` / `reader`. Hierarchy is flat — `admin` implies `writer` implies `reader` at the dispatcher level; we do not represent the implication in the table.

**Bootstrap rule:** the first authenticated `env_create_` call automatically inserts a row `(env_id, caller.principal_id, admin, now(), 'bootstrap')`. This is the *only* admin grant that does not require an existing admin to authorize it. Subsequent admin assignments go through new tools `env_grant_` / `env_revoke_` / `env_acls_browse_`.

**Enforcement:** a decorator `@require_env_role(role)` on the tool-dispatch boundary. It pulls `principal_id` from the request `AgentContext`, looks up `env_acls`, and either lets the call through or raises an MCP error mapped to HTTP 403.

**`AUTH_MODE=none` bypasses ACL enforcement entirely.** That keeps local-dev / CI / the existing stdio bridge unchanged.

### 4. Identity binding — additive principal column

`agents` gains a nullable `principal_id TEXT` column with a partial unique index `WHERE principal_id IS NOT NULL`. The constraint prevents two agent rows from claiming the same OIDC subject; the partial-index form preserves room for synthetic / pre-auth rows that don't carry a principal.

memories / relations / tombstones each gain a nullable `created_by_principal_id TEXT` audit column. We do **not** back-fill `created_by_agent_id` UUIDs into `created_by_principal_id` for legacy rows — that would invent identity that wasn't asserted at write time. Audit queries union both columns; older rows show as agent-only, newer rows show as principal-attested.

### 5. Stdio bridge — device-code flow + on-disk token cache

The bridge (`arc-bridge`-style stdio→HTTP proxy) gains:

- A new `MEMORY_MCP_AUTH_PROVIDER` env var matching the server's `AUTH_MODE` (mirror: `none` / `oidc` / `entra`).
- `MEMORY_MCP_AUTH_ISSUER`, `MEMORY_MCP_AUTH_AUDIENCE` (+ Entra preset shortcuts) for token acquisition.
- Device-code flow on first run, prompting the user in the terminal.
- Token cache at `~/.cache/memory-mcp/auth/<deployment-id>.json` with file mode `0600`.
- Automatic refresh-token use until expiry; re-prompt on full expiry.
- `MEMORY_MCP_AUTH_PROVIDER=none` short-circuits everything (default; backward-compat).

The bridge is the only client we ship; CLI users wiring up their own clients follow the same pattern documented in `docs/auth-clients.md`.

### 6. Network isolation — NetworkPolicy as a first-class chart resource

The Helm chart ships **default-on** NetworkPolicy manifests:

- memory-mcp pod **egress**: allowed to Postgres / Qdrant / Neo4j services + DNS + the configured `OIDC_ISSUER` host. Everything else denied.
- Postgres / Qdrant / Neo4j pods **ingress**: allowed only from memory-mcp's ServiceAccount. Everything else denied.
- memory-mcp pod **ingress**: allowed from any namespace by default (operator opts in to tightening this via values).

The chart does *not* expose a `networkPolicy.enabled` toggle. NetworkPolicy is the load-bearing isolation primitive given that the backing-store credentials are shared API keys; making it optional would let an operator accidentally ship a fully open cluster. Operators who genuinely need to disable it can delete the manifest in a chart fork; we accept that friction.

The E2E smoke test (Subtask 10) includes a denial case: a sidecar pod tries to reach Postgres / Qdrant / Neo4j using the same credentials but from outside memory-mcp's ServiceAccount, and the connection must time out.

### 7. Schema migration shape

Single migration `0023_auth_phase2a`:

- `CREATE TABLE env_acls (...)` per §3 (FK to `environments(id)`).
- `ALTER TABLE agents ADD COLUMN principal_id TEXT;` + partial unique index `WHERE principal_id IS NOT NULL`.
- `ALTER TABLE memories ADD COLUMN created_by_principal_id TEXT;`
- `ALTER TABLE relations ADD COLUMN created_by_principal_id TEXT;`
- `ALTER TABLE memory_tombstones ADD COLUMN created_by_principal_id TEXT;`

All additions are nullable; existing inserts continue working unchanged. Down-migration drops the column / table; we document that the down path loses all auth state, which is acceptable for a v0.x service.

> **Note on numbering and table names.** The ADR originally reserved `0022_auth_phase2a` and referenced tables `agent_accounts` and `envs(id)`. The migration shipped as `0023_auth_phase2a` because v0.17.2 had already landed `0022_message_kind` on `main` while this ADR was in review. The actual schema (since migration 0001) uses `agents` and `environments` as table names — `agent_accounts` and `envs` were working names that never landed. Both corrections applied here; see migration `0023_auth_phase2a.py` for the implemented shape.

## Consequences

### Positive

- **Drop-in deployable.** `helm install memory-mcp` gives an operator a working stack: server + three backing stores + NetworkPolicy + Secrets + auth all wired together, with one toggle to flip between local-only and IdP-protected.
- **Backward compatible.** v0.17.x users see no change unless they opt in to `AUTH_MODE=oidc` / `entra`. CI and local-dev workflows keep working with no config edit.
- **Audit-honest.** New columns carry the principal that actually wrote each row. Legacy rows stay anonymous, which is truthful.
- **IdP-portable.** The generic OIDC shape lets us point at Keycloak in CI for cheap E2E coverage and at Entra in production with a config flip.
- **Tenant-safe at the network layer.** Default-on NetworkPolicy prevents the "shared API key" risk from leaking across pods.

### Negative

- **Operator owns secret rotation.** No automation in v1. We accept this and document it; v2 follow-up tracked.
- **Postgres in-cluster ≠ a managed database.** Backups, HA, point-in-time recovery are the operator's job. We document recommendations (volume snapshots, `pg_dump` cron) but don't ship them.
- **ACL UX is bare.** Three tools (`env_grant_` / `env_revoke_` / `env_acls_browse_`) cover the surface; group-based RBAC, attribute policies, time-bounded grants are all deferred.
- **No control-plane endpoint for tenant administration.** Adding tenants today means adding ACL rows. Multi-tenant lifecycle is Phase 2b.
- **First-bootstrap admin is whoever creates the first env.** Operator must coordinate the first call. Mitigation: document it; offer a `BOOTSTRAP_ADMIN_PRINCIPAL` env var that pre-seeds a row at startup if no admin exists (cheap follow-up; not in 2a unless feedback demands).

### Neutral

- **Three `AUTH_MODE` values, not two.** The `entra` preset is sugar — strictly redundant with `oidc` + correct config — but it's the path of least friction for the IdP we expect most operators to use, and it pins the issuer URL pattern so a typo doesn't silently widen the trust boundary.
- **Helm, not Kustomize, not raw manifests.** Helm has the broadest operator literacy; we accept Helm's templating tax over the alternatives.

## Rejected alternatives

| Option | Why rejected |
|---|---|
| Azure Database for PostgreSQL Flexible Server | Locks v1 to Azure. Rejected per constraint 3. |
| Qdrant Cloud / Neo4j Aura | Same; also pulls in cross-region network paths we don't want. |
| Azure Key Vault + CSI driver | Adds operator setup cost (CSI install, MI binding) for a benefit we don't need at this scale. |
| Managed Identity / Workload Identity | Same lock-in problem; we want NetworkPolicy + shared-secret to work on any cluster. |
| external-secrets-operator | Useful in v2; v1 stays with native K8s Secrets to minimize chart prerequisites. |
| ServiceAccount + TokenReview for pod-to-pod | Bypasses Entra device-code dance for in-cluster agents. Real benefit; deferred as a *secondary* mode in v2 (`AUTH_MODE=k8s-sa`). |
| Full MCP-spec OAuth 2.1 + Dynamic Client Registration | Entra doesn't support DCR. Deferred to Phase 2b along with the option to front Entra with Keycloak. |
| Hard-coded Entra-only validator | Forecloses Keycloak / Auth0. The generic OIDC shape costs ~50 lines extra and pays for itself the first time anyone wants to wire a non-Entra IdP. |
| Per-tool ACL (e.g. `mem_write` requires writer, `mem_search` allows reader anonymously) | Tempting but inverts the threat model: read access without authentication is the larger leak. v1 enforces auth uniformly on `/mcp/`. |
| Optional NetworkPolicy (toggle in values.yaml) | Makes operator footguns too easy. NetworkPolicy is load-bearing; it ships default-on. |
| Inline secrets in `values.yaml` | Helm values land in `Secret` resources anyway, but a `values.yaml` checked into git would leak. We document operator pattern: a `secrets-values.yaml` overlay kept out of source control, or `--set-file`. |
| Back-fill `created_by_principal_id` for legacy rows | Inventing identity. Audit honesty wins over completeness. |

## Follow-ups (post-2a)

These are intentionally **not** in scope; tracked for v2/Phase 2b:

- External-secrets-operator integration (`memory-mcp-external-secrets-integration`).
- Multi-tenant deployment shape (Phase 2b).
- MCP-spec OAuth 2.1 + Dynamic Client Registration (Phase 2b; likely via Keycloak federation).
- Server-side audit log for auth events (login, role grant, ACL change) — currently relies on app-level INFO logs.
- Refresh-token rotation in the bridge cache.
- In-cluster `AUTH_MODE=k8s-sa` for pod-to-pod traffic without device-code.
- `BOOTSTRAP_ADMIN_PRINCIPAL` env var for first-env-admin seeding.
- Group-based ACLs (Entra group claims → `env_acls.role`).

## Validation plan

Subtask 10 (`auth-2a-10-e2e-smoke`) covers the whole stack:

1. Spin up the chart against a `kind` cluster.
2. Run a mock OIDC IdP (or wire real Entra creds via operator).
3. Bridge mints a device-code token, calls `mem_write`, verifies the row carries `created_by_principal_id`.
4. Sidecar pod attempts to reach Postgres / Qdrant / Neo4j outside memory-mcp's ServiceAccount → must be denied by NetworkPolicy.
5. `AUTH_MODE=none` path runs as a regression check that nothing in 2a broke local-dev.

The smoke test runs in CI on every PR against `feat/auth-phase-2a-*` branches.

## Subtask map

| # | Slug | Output |
|---|---|---|
| 01 | `auth-2a-01-adr` | This file + a `decision` memory in env `workspace` |
| 02 | `auth-2a-02-schema` | Alembic migration `0023_auth_phase2a` + tests |
| 03 | `auth-2a-03-jwt-dep` | `auth/oidc.py` + FastAPI dep + unit/integration tests |
| 04 | `auth-2a-04-acl` | `env_acls` enforcement + `env_grant_` / `env_revoke_` / `env_acls_browse_` tools |
| 05 | `auth-2a-05-config-secrets` | `config.py` audit + `docs/deployment.md` Secret-key reference |
| 06 | `auth-2a-06-bridge-auth` | Bridge device-code + token cache |
| 07 | `auth-2a-07-helm` | `charts/memory-mcp/` |
| 08 | `auth-2a-08-network-policy` | NetworkPolicy manifests + denial verification |
| 09 | `auth-2a-09-docs` | `docs/auth.md`, `docs/auth-clients.md`, `docs/deployment-k8s.md`, README update |
| 10 | `auth-2a-10-e2e-smoke` | `make e2e-auth-smoke` kind-cluster test |

Each ships as a separate PR. Dependencies are encoded in the workspace's session SQL (`todo_deps`).
