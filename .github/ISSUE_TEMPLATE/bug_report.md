---
name: Bug report
about: Something in the mira-core server is broken
title: ''
labels: ''
assignees: ''

---

<!--
Thanks for filing an issue! Fill in the sections that apply and delete the rest.

SECURITY VULNERABILITIES: do not file them here — see SECURITY.md and use the
repo's Security tab → "Report a vulnerability".

Questions and "is this useful" feedback are welcome in Discussions instead:
https://github.com/mabaeyens/mira-core/discussions
-->

### Summary

<!-- One or two sentences describing what went wrong. -->

### Steps to reproduce

1.
2.
3.

### Expected vs. actual

- **Expected:**
- **Actual:**

### Environment

- **macOS version:** <!-- e.g. macOS 26.5.1 -->
- **Mac model / chip / RAM:** <!-- e.g. MacBook Pro M5, 32GB -->
- **Python version:** <!-- python3 --version -->
- **mira-core version:** <!-- the release tag you installed, or `git rev-parse --short HEAD` -->
- **Backend + model:** <!-- e.g. mira-mlx + mlx-community/Qwen3.6-35B-A3B-4bit; see GET /info -->
- **Interface:** <!-- CLI (`mira chat`) / web UI / iOS app / macOS app -->

<details>
<summary><code>make doctor</code> output</summary>

```
paste here — it covers most of the above in one go
```

</details>

### Logs

<!--
Server log: /tmp/com.mab.mira.log if you run the LaunchAgent, otherwise the
terminal running `mira serve`. The inference engine logs separately to
~/.local/share/mira/mira-mlx.log — include it for hangs, crashes, empty replies
or anything model-shaped.

Please skim for your auth_token and any file paths you'd rather not share.
-->

```
```

### Anything else

<!-- Screenshots for web UI issues, a conversation excerpt, what you'd already tried. -->
