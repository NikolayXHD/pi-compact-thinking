# pi-compact-thinking

Pi extension that shrinks the thinking in the context.

- in some tasks reasoning takes more than half of the context;
- the extension compresses that part by 30–60%.

Old thinking blocks are rewritten by a background call of the same model into
short digests: decisions, rejected options, facts and open questions survive,
repetition and filler do not. Recent blocks stay raw.

## What you see

Footer: `💭 -40k/120k +76k` — reasoning tokens saved, raw reasoning tokens in
context, tokens spent by the background calls. Spinner while a rewrite runs,
`💭 off` when disabled.

- `/compact-thinking` — report;
- `/compact-thinking on` / `off`;
- `/compact-thinking dump` — raw and compacted blocks next to the session file,
  with matching headings so a diff aligns.

## Install

```bash
pi install git:github.com/NikolayXHD/pi-compact-thinking
```

Settings: `~/.pi/agent/compact-thinking.json`.

| field             | default | meaning                       |
| ----------------- | ------- | ----------------------------- |
| `enabled`         | `true`  | mode is on at session start   |
| `k`               | `5`     | recent blocks that stay raw   |
| `minTokens`       | `512`   | shorter blocks are left alone |
| `acceptanceRatio` | `0.95`  | a longer digest is rejected   |

## How it works

A digest is a separate call of the same model over the same session prefix, so
the provider reads it from the warm cache: no tools, no reasoning, the answer
wrapped in `<digest>…</digest>`.

Digests are substituted into the context by one function called in `context`
(before the main model request) and in `session_before_compact` (before Pi's
own summarizer). Nothing is written to the session, so Pi keeps deriving the
context size and the compaction threshold from the real provider `usage`.

The background job runs while the agent is busy with tools and right before the
session settles; it never competes with the model stream.

## Limits

- works while the extension is loaded, nothing is stored in the session;
- every block costs a full prefix read: a warm cache matters;
- does not replace Pi compaction, it shrinks thinking only.

## Development

`node src/probe.mjs` for pure rules, `src/TESTING.md` for session scenarios.
