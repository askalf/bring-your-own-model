# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

byom is pre-1.0. Security fixes land on the latest 0.1.x release; older
snapshots are not patched — upgrade to the current version.

## Reporting a Vulnerability

If you discover a security vulnerability in bring-your-own-model, please
report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email **security@askalf.org** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. **Response:** Acknowledgment within 48 hours; we aim to ship a fix for
   confirmed high-severity issues within 7 days. As a small, dependency-free
   project we cannot promise formal SLAs beyond a best-effort response.
4. We will coordinate disclosure with you before publishing a fix.

## Scope

byom is a local proxy that translates the Anthropic Messages API to and from
OpenAI (Responses API + Chat Completions). It sits between two parties it does
not control — a client sending arbitrary Anthropic-shaped request bodies and an
upstream sending arbitrary OpenAI-shaped responses/SSE. In scope:

- API-key or credential leakage in logs, errors, or forwarded responses
- Request/response translation exploits (injection via model names, message
  content, tool definitions, or SSE framing)
- SSE stream parse / framing exploits in either translation direction
- Denial of service via the proxy (unbounded output, super-linear parsing)
- Prototype pollution through hostile request or upstream bodies

Out of scope: the security of whatever OpenAI-compatible upstream you point byom
at — that URL is operator-configured and is your trust anchor, not byom's.
