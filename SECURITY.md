# Security policy

Agent Reflex can sit on an authorization path, so treat policy bypasses, unsafe auto-execution, secret leakage, cross-tenant data exposure, auth bypasses, and dependency compromise as security issues.

Do not include credentials, customer traces, or exploit payloads containing real secrets in a public report. Use a private security advisory in the hosting repository when available.

## Security invariants

1. Probabilistic providers never override deterministic hard policy.
2. High-risk requests abstain unless handled by a deterministic or explicitly trusted provider.
3. Invalid/non-finite outputs are rejected by the typed firewall.
4. Telemetry is best-effort and must never alter the decision path.
5. Secrets are redacted before local audit sinks; operators should still avoid sending unnecessary secrets.
6. Public HTTP serving is authenticated by default; unauthenticated mode is opt-in.
