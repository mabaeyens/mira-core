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

**One thing you must configure: `allowed_hosts`.** Before the token or the source-IP
allowlist, the server checks the `Host` header (this is what blocks DNS rebinding). It
accepts loopback and any bare IP inside `allowed_source_cidrs` automatically — but a
*name* is never accepted unless you list it. Your Tailscale certificate is issued for the
MagicDNS name **only**, so remote clients have to connect by name, which means:

```yaml
# mira.yaml
allowed_hosts:
  - your-mac.tailXXXX.ts.net
```

`make install ARGS="--with-tailscale <host>"` writes this for you. If you skip it, the
server answers **403 on every request** over a perfectly healthy connection — see the
troubleshooting section below, because the symptom does not look like a 403.

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

## Running Tailscale day-to-day

Leaving Tailscale always-on is fine; a few settings keep it healthy and tight.

**On the Mac (the server) — recommended:**

- **Disable key expiry for this node.** Node keys expire (default 180 days) and a lapsed
  key silently drops the Mac off the tailnet — remote access just stops working. In the
  Tailscale admin console → *Machines* → the Mac → *Disable key expiry*. Do this so the
  server is never unreachable mid-trip.
- **Lock down with ACLs.** By default every device on your tailnet can reach every other
  device's ports. Restrict access to Mira so only your own devices can hit it. In the
  admin console *Access controls*, scope it to your own user, e.g.:

  ```jsonc
  {
    "acls": [
      // only this user's devices may reach the Mac's HTTPS port
      { "action": "accept", "src": ["autogroup:member"], "dst": ["<mac-name>:8443"] }
    ]
  }
  ```

  Tighten `src`/`dst` to taste (tag the Mac, name specific devices). The point: a single
  compromised or shared tailnet node shouldn't be able to reach the RCE-capable server.
- The control plane brokers keys only — it never sees your traffic (WireGuard is
  end-to-end encrypted) — but it does see metadata (which nodes connect).

**On the phone/iPad — always-on tradeoffs:**

- **Battery:** small when idle (WireGuard is low-power UDP); higher only if traffic is
  relayed via DERP instead of a direct connection, or if you route everything through an
  exit node. For occasional Mira use it's negligible.
- **Internet/bandwidth:** no effect on normal traffic — only tailnet (`100.x`) and
  enabled subnet routes go through the tunnel. No speed tax unless you use an exit node.
- **DNS / captive portals:** with MagicDNS overriding system DNS, café/hotel sign-in
  pages and local split-DNS can break until you toggle Tailscale off. If you hit this,
  turn off **"Override local DNS"** in Tailscale's DNS settings — you keep `*.ts.net`
  name resolution without hijacking all queries.
- You don't *have* to leave it on: since you only need it to reach Mira, flipping it on
  when needed sidesteps the battery/DNS/portal friction entirely. Always-on is also fine.

## Troubleshooting: "could not reach server" when the network is fine

The apps show one message for two very different failures — never got there, and got
there and was refused. Find out which before touching anything else. From the Mac:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://<your-mac>.<tailnet>.ts.net:8443/health
```

| Result | Meaning |
|---|---|
| `200` | Server is fine. The problem is on the client (Tailscale off, wrong URL, wrong scheme). |
| `403` | You reached it and it refused you — almost always `allowed_hosts`. See below. |
| `401` | Reached and allowed, token wrong or missing. |
| `000` + curl exit `60` | TLS failure. You're probably connecting by **IP** — the cert has no IP SAN. Use the MagicDNS name. |
| `000` + curl exit `7`/`28` | Genuinely unreachable: Tailscale down, or `:8443` never bound (check `tailscale status` and `lsof -nP -iTCP:8443 -sTCP:LISTEN`). |

**Confirming a 403 is the Host gate** — re-send the same request with the Host header
overridden to the tailnet IP. Same socket, same TLS, only the header changes:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -H "Host: 100.x.y.z:8443" \
  https://<your-mac>.<tailnet>.ts.net:8443/health
```

`200` with the IP and `403` with the name is conclusive: the name isn't in
`allowed_hosts`. Add it (see the Default section), then restart — the allowlist is read
at import, so a reload is required:

```bash
/mira-server restart
```

Why the symptom is confusing: `/health` is exempt from the *token* check but **not** from
the Host check, so the very probe clients use to test reachability is the one being
rejected. The app concludes the server is unreachable and points you at your network,
while the server is up, the tailnet is healthy, and the token is correct.

> Note: `:8443` binds the tailnet address only, so probing `https://127.0.0.1:8443` from
> the Mac fails by design. Use the MagicDNS name even when testing locally.

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
