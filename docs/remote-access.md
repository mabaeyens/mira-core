# Remote access — choosing your network posture

Mira's server can run shell commands and read/write files, so a leaked credential means
full remote code execution on your Mac. Access is therefore locked down by default, and
nothing sensitive is ever sent in plaintext off the machine. This doc explains the
default, how to reach Mira while travelling, and the opt-in (riskier) plain-LAN mode.

## Default (secure) — nothing to configure

With an `auth_token` set (see the README's *Access control*):

- **HTTP `:8000` binds `127.0.0.1` only.** The local web UI and health checks work on
  the Mac itself; no token or message ever leaves the machine in clear text.
- **HTTPS `:8443` binds the Tailscale interface only.** The off-host socket exists
  *solely* on your tailnet, so only your enrolled devices can even open a connection,
  and the Tailscale certificate makes it trusted HTTPS with no per-device setup.
- Every route (except `/health` and the static UI) needs `Authorization: Bearer <token>`,
  and a source-IP allowlist (loopback + the `100.64.0.0/10` tailnet range) is enforced as
  defense-in-depth.

If Tailscale is **not running** when the server starts, `:8443` **fails closed to
loopback** — remote access is simply off until you start Tailscale and restart:

```bash
tailscale up
/mira-server restart        # or: launchctl kickstart -k gui/$(id -u)/com.mab.mira
```

> The bind is decided at startup. If you start Tailscale *after* the server, restart the
> server so it picks up the tailnet address.

## Travelling — use Tailscale, mind the VPN conflict

When you're away from home, your phone and your Mac are on different networks. The only
way to reach the Mac is a tunnel that terminates **at your house** — that's Tailscale.
Connect the app to `https://<your-mac>.<tailnet>.ts.net:8443`.

### Why Proton VPN (or any commercial VPN) conflicts on iPhone/iPad

- **iOS allows only one active VPN tunnel at a time.** Tailscale and Proton VPN are both
  Network Extension tunnels, so turning one on disconnects the other — they can't run
  together, and iOS has no general per-app split tunnelling.
- **Proton can't reach your Mac anyway.** A commercial VPN tunnels your traffic to *its*
  servers for privacy/geo — not to your home. So Proton and "reach Mira from afar" are
  different jobs, not alternatives.

What to do:

- **Just using Mira for a bit:** toggle Proton off, Tailscale on. Switch back after.
- **Want privacy *and* Mira at once:** run an **exit node** on your home network (e.g. the
  Mac or another always-on device) and route your phone's traffic through it in the
  Tailscale app. One tunnel does both jobs — encrypted transit out of the café *and* a
  path to your Mac — without Proton.

## Opt-in: plain-WiFi LAN access (accept the risk)

If you're on the **same Wi-Fi as the Mac** and would rather not run Tailscale, you can
re-enable direct LAN access. This sends the token and your messages over the local
network in a form other devices on that Wi-Fi can potentially capture — **plaintext,
sniffable.** Only do this on a network you trust, and prefer the Tailscale path otherwise.

In `mira.yaml`:

```yaml
# add your LAN subnet to the source allowlist
allowed_source_cidrs: ["127.0.0.0/8", "::1/128", "100.64.0.0/10", "192.168.0.0/24"]
```

…and start the server with `MIRA_HOST=0.0.0.0` so HTTP `:8000` binds the LAN. Connect the
app to `http://<mac-lan-ip>:8000`. Restart the server after changing either setting.

To go back to the secure default, remove the LAN subnet and unset `MIRA_HOST`.
