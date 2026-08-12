# The mlx-lm pin: what it is, and when to move it

`pyproject.toml` does not install mlx-lm from PyPI. It installs one exact commit
of a personal fork:

```toml
[tool.uv.sources]
mlx-lm = { git = "https://github.com/mabaeyens/mlx-lm.git", rev = "291a61a…" }
```

That commit is the tip of the `mira-core-pin-vision` branch. This document exists
because "should I move the pin" has been asked three times and answered from
memory each time, and the answer depends on things that are checkable in about
five minutes.

## Why a fork at all

Twelve commits that are not on upstream `main`. They fall into four groups, and
they are not equally important:

| group | commits | what breaks without it |
|---|---|---|
| Mistral tool-call flush | `2a3a696` | Mistral-family tool calling. Empty `tool_call_end_tokens` defeats the stock EOS flush check. Upstream's own PR for this (#1373) was closed, not merged. |
| KV-cache quantization for continuous batching | `bbd8496` | `mira_mlx_kv_bits: 8`, live since 2026-07-18. Upstream PR #1584, unreviewed. |
| MoE disk-backed expert offload | `f0c66a4` and five follow-ups | The offload path that cut peak memory 18.21 → 7.25GB. Upstream PR #1588, unreviewed. |
| Vision prefill from embeddings | `839c707`, `291a61a` | `mira_mlx_vision`. Image turns cannot prefill. |

So the fork is not a patch waiting to be upstreamed. It is four features, three
of which have open PRs that no maintainer has looked at, and one (the Mistral
fix) whose upstream attempt is already closed.

## What actually breaks if you move it today

Upstream `main` was last fetched here on 2026-08-05. Against the pin it differs
by 24 commits, and two of them remove things `mira_mlx_server.py` imports by
name:

```python
from mlx_lm.generate import BatchGenerator, SequenceStateMachine, _embed_tokens
```

- **`SequenceStateMachine` no longer exists.** PR #1501 ("text-based state
  machine for tool/reasoning parsing") replaced it with `TextStateMachine`,
  which is a different API, not a rename. This is the one already recorded in
  the `pyproject.toml` comment.
- **`_embed_tokens` no longer exists either**, and this one is not recorded
  anywhere. It is a private helper — the leading underscore is upstream telling
  us it was never ours to import — and vision prefill calls it.

Everything else mira-core imports (`BatchGenerator`, `ToolCallFormatter`,
`hf_repo_to_path`, `SwitchGLU`, `QuantizedSwitchLinear`, the cache classes,
`make_sampler`, `make_logits_processors`) is still there under the same name.

A move today is therefore not "rebase and run the tests". It is a port of two
call sites, one of which has no supported replacement.

## The conditions for moving it

Move when **all** of these hold. Any one of them missing means the move costs
more than it returns.

1. **The features you rely on are upstream, or you have stopped relying on
   them.** Concretely: #1584 (or its A/B/C split), #1588 and the vision
   embedding path. If a feature is neither merged nor abandoned, moving the pin
   means carrying it as a rebase on top of a moving target — which is the exact
   situation the pin was created to end.
2. **`TextStateMachine` has a migration you have read, not guessed.** Read
   upstream's own server code for how it drives the new class. The old one was
   driven per sequence; do not assume the new one is.
3. **`_embed_tokens` has a public replacement.** If it does not, the vision path
   needs its own copy of that logic, and that is a decision to take deliberately
   rather than discover during a rebase.
4. **Something is actually gained.** A newer upstream is not a gain by itself.
   Name the commit you want and why.

## The one condition that forces a move

If upstream ships something Mira cannot do without — a model architecture the
pinned commit does not know, a correctness fix in the batching path — then the
cost above is paid whether it is convenient or not. Watch for new model support
in particular: that is the change most likely to arrive as "the model everyone
is using does not load".

**That condition is already met, found 2026-08-12.** Upstream commit `a790972`,
"Fix broadcast crash in quantized SDPA with GQA + batched padding mask
(batch >= 2)" (PR #1467, merged 2026-07-09), is one of the 24 commits the pin
does not have. Without it, `quantized_scaled_dot_product_attention` compares a
4-D mask against 5-D scores whenever a GQA model runs quantized KV at batch 2 or
more — which is Mira's production configuration — and the engine thread dies.
Verified rather than assumed: the installed copy contains zero occurrences of
that guard, and the crash reproduces standalone in a second.

It is two lines plus a test and it touches nothing else, so **the cheap answer was
to cherry-pick `a790972`, not to rebase** — that buys the correctness fix without
paying for `SequenceStateMachine` or `_embed_tokens`. **Done the same day:**
`9721b95` on `mira-core-pin-vision`, pushed as a fast-forward, `pyproject.toml`
bumped from `291a61a`, venv resynced, and three concurrent long requests survived
on the restarted server.

The general lesson, which is why this section exists at all: **the forcing
condition does not necessarily force a full move.** A single upstream commit may
be liftable on its own, and checking that first is cheaper than either rebasing or
living with the bug. Ask "does this fix stand alone?" before "is it time to move
the pin?".

Read `notes/kv-quant-batched-mask-crash-2026-08-12.md` for the mechanism.

## How to move it, when the time comes

1. Fetch upstream and read `git log --oneline <pin>..upstream/main` in full.
   Twenty-four commits is a readable number; do not skip this and rely on tests.
2. Rebase `mira-core-pin-vision` onto upstream `main` on a **new** branch name.
   Never force-push the branch `pyproject.toml` points at — that is what makes a
   pin a pin, and `mira-mistral-tool-call-fix` is force-pushed for PR prep, which
   is precisely why the pin branch is separate from it.
3. Port the two imports. Expect `SequenceStateMachine` to be a rewrite of the
   tool/reasoning parsing seam, not a substitution.
4. Run the engine-adjacent tests, then **run the engine**. The failure mode this
   guards against was a live import error, and the tests that would have caught
   it did not exist at the time.
5. Bump `rev` to the new commit, and update the comment above it to say what the
   new branch carries.

## Bookkeeping that is currently wrong

`BACKLOG.md` describes the pin as "frozen at `bbd8496`". It is not, and has not
been for some time: `bbd8496` is now ten commits back, the branch named
`mira-core-pin` stops at `65fcb4c`, and the installed rev is `291a61a` on
`mira-core-pin-vision`. The freeze held; the record of which commit it froze at
did not. When the pin moves, this document and that entry both need the new SHA,
which is an argument for keeping the SHA in exactly one place — here — and
letting the backlog point at the document instead.
