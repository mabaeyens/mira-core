# Security Policy

## Supported versions

mira-core is a single-maintainer project. Only the **latest release** (and the current
`main`) is supported — please reproduce any issue on the current version before reporting.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub:

1. Go to the [**Security**](https://github.com/mabaeyens/mira-core/security) tab of this repo.
2. Click **Report a vulnerability**.
3. Describe the issue.

Helpful details to include:

- A description of the vulnerability and its impact.
- Steps to reproduce (or a proof of concept).
- Your OS, Python version, and the mira-core version.
- The active inference backend + model, if relevant.

You'll get a best-effort acknowledgement; as a one-person project, response times vary. Once
a fix ships, the advisory can be published to credit the reporter (let me know if you'd
prefer to stay anonymous).

## Scope

- The server gates all network access behind a shared bearer token — see **Access control**
  in the [README](README.md) and [docs/remote-access.md](docs/remote-access.md). Keep that
  token out of the repo and out of client config.
- Inference and data stay on the user's own hardware. Off-host access is **HTTPS-only over
  Tailscale** (the listener binds the tailnet interface); plain HTTP stays loopback-only and
  the server is never intended to be exposed to the public internet.
- User input is validated against command injection and path traversal; shell operations use
  `subprocess` with explicit argument lists. Reports of gaps here are especially welcome.
- The native clients are in [mira-apps](https://github.com/mabaeyens/mira-apps) — report
  client-side issues there.
