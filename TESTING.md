# Test scenarios

The extension densifies old thinking blocks into digests while keeping the raw
text in the session. Digests are substituted into the context on the fly: in
`context` and in `session_before_compact`.

The automated run is `node src/probe.mjs` from the package root: pure rules for
candidate selection, acceptance, replacement assembly and status text.
Everything that lives only in the session is checked manually through a child
`pi` session and its JSONL.

The background job starts in the tool window (`tool_execution_start`) and
before settling (`agent_before_settle`); it never runs while the main model
streams.

## Config

The file is `~/.pi/agent/compact-thinking.json`. Fields: `enabled`, `k`,
`minTokens`, `acceptanceRatio`. A missing file equals the defaults; a broken
field falls back to its default, and the extension warns once per session.

## Selection and replacement

Rules live in probe.mjs. Session consequences:

- blocks shorter than `minTokens` are left alone;
- `k` counts all thinking blocks, including redacted ones, so the raw zone
  keeps the width of that many actions;
- a block whose visible text was replaced by a foreign `context_edit` is not
  touched;
- replacement changes the text of target blocks and leaves neighbouring blocks,
  text and tool calls intact;
- a digest is accepted only with a normal stop reason, without tool calls and
  not longer than `acceptanceRatio` of the source block; the answer must
  contain `<digest>…</digest>`, text outside the tags is ignored, tool-call
  markup inside the tags is rejected;
- a rejection caused by the block (length cut-off, tool call, missing or empty
  wrapper, tool markup, grown answer) is recorded and never retried; a network
  or cancellation rejection leaves the block in the queue, but not more than
  three attempts per session.

## Manual session scenarios

Run each scenario in an empty directory so the JSONL lands in its own place:
`mkdir -p /tmp/compact-thinking-check` and run `pi -p "..."` inside it.

Status. At start the footer shows `💭 -…k/…k +…k`: savings, raw reasoning sum
and generation of the background calls. While the background call runs, a
spinner follows the balloon; the disabled mode shows `💭 off`.
`/compact-thinking` prints a detailed report, `off`/`on` toggle. Sums are
recomputed at every turn boundary, so in a running session with reasoning the
second number must not stay zero.

Compaction during a task. With `k = 0` and `minTokens` around 16 ask the agent
to run several bash commands in a row. After the second answer the session
JSONL contains `compact-thinking-applied` entries; the extension writes no
`context_edit` entries at all; the assistant message entry keeps the full raw
thinking.

Substitution reaches the provider. A temporary logger extension on
`before_provider_request` writes the length and the start of
`reasoning_content` for assistant messages; in the log a raw block turns into a
digest and stays that way in later requests.

The summarizer sees digests. A temporary logger extension on
`session_before_compact` writes thinking block lengths from
`preparation.messagesToSummarize`; those must be digests, not raw text.

Context change. After a digest is applied, `/compact-thinking` shows non-zero
digests and savings, while the raw text stays in the message entry.

Tree navigation. Make several turns, then `/tree` to an earlier entry. The
state is rebuilt for the new branch: the status counts only its entries, and
digests from the abandoned branch do not enter the context.

Foreign edit. If a block's visible text was replaced by a foreign
`context_edit`, the extension leaves it alone and no `compact-thinking-applied`
entry appears for it.

Resume. Interrupt a session and resume it. Digests are restored from
`compact-thinking-applied` entries and substituted into the context again; the
status matches the number of active entries.

Disable. With `/compact-thinking off` no new extension entries appear until the
mode is enabled again.

Dump. After several compactions `/compact-thinking dump` writes `thinking-a`
and `thinking-b` next to the session JSONL; block headings match in both files,
blocks follow session order, and `thinking-b` holds what the model sees. Diff
recipes are in README, section Observability.

Provider signature. For DeepSeek and vLLM the digest keeps
`thinkingSignature: "reasoning_content"`; for a provider without a
reasoning-field signature it is dropped.
