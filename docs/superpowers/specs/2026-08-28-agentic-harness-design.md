# Agentic Harness — Design Spec

## Purpose

Extend the existing Telegram → local LLM bot into a real agent: a harness
that runs a model/tool loop until the model produces a final answer, with
an `exec` tool that runs commands on the host machine, file tools scoped to
a per-session workspace, loadable skills (`SKILL.md`), persistent
conversation sessions, and context compaction when the window fills.

The current app answers each message independently in one LLM call. This
spec replaces that single call with a bounded loop, and replaces the
stateless `build_messages()` seam with a persisted session.

Assumed backend for now: Ollama at `http://localhost:11434/v1` running
`qwen3:1.7b`. Everything here must still work against MLX and vLLM by
config change only — but the small-model assumption drives several
decisions below (see Design notes).

## Non-functional requirements

- Carries forward from the original spec: swappable LLM backend via config,
  no SDKs (raw HTTP), a single failure never crashes the poll loop,
  every LLM call killable, dependencies via `uv`.
- Only allowlisted Telegram **user** ids may talk to the bot at all.
- Every `exec` command is confirmed by the user before it runs, unless an
  explicit escape hatch has been engaged.
- A tool that has not been confirmed cannot touch anything outside the
  session workspace.
- One slow or blocked chat never delays another chat.
- Every model request/response is recoverable after the fact for debugging.

## Non-goals

- Multi-bot support; rich Markdown formatting; rate-limit handling beyond
  backoff (all carried forward from the original spec).
- An automated test suite (manual verification only).
- Streaming partial model output into Telegram (progress is reported per
  tool call, not per token).
- Sandboxing or containerizing `exec`. The trust boundary is the user
  allowlist plus per-command confirmation, deliberately and explicitly.
- Restart-resumption of an *in-flight* turn. Sessions persist; a turn
  interrupted by a restart is abandoned.

## Design notes (why these choices)

**The model is small.** A 1.7B model chooses tools poorly, emits malformed
tool calls, and writes weak summaries. Three decisions follow: the tool-call
parser accepts a text fallback when the model ignores the `tools` API; the
skill system keeps an explicit override so you can force a skill when the
model won't pick it; and summarization is the *last* compaction tier, not
the first.

**Approval-gated paths are unrestricted; unapproved paths are scoped.**
`exec` can touch anything on the host, but always asks. `read_file` /
`write_file` never ask, so they are code-enforced to the workspace root.
This invariant is what makes "confirm exec only" safe — if `write_file`
can escape the workspace, the confirmation gate is decorative.

**Ollama defaults `num_ctx` to 4096** regardless of the model's advertised
window. `LLM_CONTEXT_TOKENS` must describe the *server's* configured
window, not the model's theoretical maximum, or compaction will trigger
far too late.

## Configuration

Existing variables are unchanged except where noted. New ones:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ALLOWED_USER_IDS` | yes | — | Comma-separated Telegram **user** ids permitted to use the bot |
| `DB_PATH` | no | `./oh-my-bot.db` | SQLite database file |
| `WORKSPACE_ROOT` | no | `./workspaces` | Parent dir for per-session workspaces |
| `SKILLS_DIR` | no | `./skills` | Directory of `<name>/SKILL.md` skill packages |
| `MAX_LOOP_ITERATIONS` | no | `5` | Model→tool rounds per user message before the breaker trips |
| `MAX_LLM_RETRIES` | no | `3` | Transport-level retries of a failed LLM call |
| `MAX_CONSECUTIVE_TOOL_FAILURES` | no | `3` | Same-tool failures in a row before the breaker trips |
| `EXEC_TIMEOUT_SECONDS` | no | `30` | Per-command wall clock before the child is killed *(open, see Q5)* |
| `EXEC_MAX_OUTPUT_BYTES` | no | `8192` | Truncation cap on captured output *(open, see Q5)* |
| `APPROVAL_TIMEOUT_SECONDS` | no | `600` | How long a pending approval waits before auto-denying |
| `LLM_CONTEXT_TOKENS` | no | `4096` | The **server's** configured context window |
| `REASONING_TAGS` | no | `think,thinking,reasoning,thought,reflection,scratchpad` | Tags whose contents are stripped from replies; empty disables stripping |
| `STOP_SEQUENCES` | no | `<\|im_end\|>,<\|endoftext\|>,<\|eot_id\|>,<end_of_turn>` | End-of-turn markers sent to the server and truncated on; empty disables truncation |
| `COMPACT_THRESHOLD_PCT` | no | `75` | Percent of the window that triggers compaction *(open, see Q6)* |

`MAX_WORKERS` changes meaning: it no longer bounds concurrent messages, it
bounds concurrent **chat actors**. Given the allowlist is small, this is
effectively unbounded in practice.

`ALLOWED_USER_IDS` is checked against `message.from.id`, never
`message.chat.id` — in a group those differ, and a chat-id check would
authorize every member of an allowed group.

## Dependencies

No new third-party dependencies. `sqlite3`, `subprocess`, `threading`,
`pathlib`, and `re` are all stdlib. `SKILL.md` frontmatter is parsed by
hand (a few lines of `key: value` splitting) rather than adding PyYAML.

## Project layout

```
oh-my-bot/
├── skills/
│   └── <name>/
│       ├── SKILL.md            # frontmatter + instructions
│       └── *.py, *.sh          # optional sibling scripts
├── workspaces/<chat_id>/<session_id>/   # gitignored
├── oh-my-bot.db                          # gitignored
└── src/oh_my_bot/
    ├── config.py               # (grows)
    ├── telegram_client.py      # (grows: keyboards, callbacks)
    ├── llm_client.py           # (changes: complete() signature)
    ├── worker.py               # (shrinks: killable-child helper only)
    ├── main.py                 # (changes: poll loop becomes a router)
    ├── store.py                # SQLite access
    ├── session.py              # Session object, /new, history
    ├── actors.py               # per-chat actor threads and queues
    ├── agent.py                # the agentic loop and circuit breakers
    ├── approvals.py            # pending approvals, always-allow patterns
    ├── context.py              # token accounting, tiered compaction
    ├── skills.py               # skill discovery and loading
    └── tools/
        ├── __init__.py         # registry, JSON schemas, dispatch
        ├── exec.py             # exec (approval-gated)
        ├── files.py            # read_file / write_file (workspace-scoped)
        └── skill.py            # skill (progressive disclosure)
```

Conventions carried forward: `src` layout, relative imports, and a one-line
comment above every function describing its purpose.

## Components

### `main.py` — the router

The poll loop stops dispatching to a thread pool and becomes a router:

- Reject any update whose `from.id` is not in `ALLOWED_USER_IDS` *(behavior
  on rejection is open, see Q2)*.
- `message` updates → append to that chat's actor queue.
- `callback_query` updates → hand to `approvals.resolve()` for that chat.
  These already arrive from `getUpdates` by default; the current code drops
  them at `worker.py:62-64` via the `"text" not in message` early return.
- Poll-level errors keep the existing capped-backoff retry.

### `actors.py` — per-chat actors

One long-lived thread plus one `queue.Queue` per chat, created on first
message. The actor owns the session and drains its queue strictly FIFO: a
message arriving mid-turn waits its turn rather than interrupting.

Blocking is now expected and safe — an actor waiting on an approval blocks
only its own chat. This is the reason the thread pool had to go: under the
old model a pending approval would hold a pool slot for as long as the user
took to tap a button.

*Idle-actor reaping is open (see Q8).*

### `agent.py` — the loop

One turn, for one user message:

1. Load session history; compact if over threshold (`context.py`).
2. Call the model with the tool schemas and the skill index.
3. If the response has no tool calls → send it as the final answer, done.
4. For each tool call: gate on approval if it is `exec`, execute, post a
   progress message, append the result to the history.
5. Increment the iteration counter and go to 2.

Three independent circuit breakers, each with its own budget and its own
user-facing message so a dead turn explains itself in the chat:

| Breaker | Trips when | Message |
|---|---|---|
| `MAX_LOOP_ITERATIONS` | 5 model→tool rounds elapse without a final answer | "I hit my step limit after N steps." |
| `MAX_LLM_RETRIES` | the LLM call fails N times in a row | "I couldn't reach the AI service." |
| `MAX_CONSECUTIVE_TOOL_FAILURES` | the same tool errors N times in a row | "A tool kept failing; stopping here." |

On any trip, whatever work completed stays in the session history, so a
follow-up message continues from there rather than starting over.

### `llm_client.py` — tool-calling connector

`complete(messages, tools) -> AssistantMessage` replaces
`complete(messages) -> str`. The return carries both `content` and
`tool_calls`; `build_messages()` is deleted, its role taken by `session.py`.

Tool calls are read from the response's native `tool_calls` field. When
that field is absent but the content contains a fenced tool-call block,
the block is parsed instead. This covers both a backend that ignores the
`tools` parameter and — more commonly at this model size — a backend that
accepts it while the model answers in prose anyway. *Detection strategy is
open (see Q4).*

A response that is neither a clean tool call nor parseable text is treated
as a tool failure and counts against `MAX_CONSECUTIVE_TOOL_FAILURES`.

### `tools/exec.py`

One-shot `bash -c` per call, in a fresh child process, cwd set to the
session workspace. No state persists between calls: the model must re-`cd`
every time, and shell exports do not survive.

- Killable and timeout-bounded, reusing `worker.py`'s existing child-process
  pattern.
- Output truncated to `EXEC_MAX_OUTPUT_BYTES` with an explicit marker so the
  model knows it was cut. *(stdout/stderr merged or separate is open, Q5.)*
- The child's environment is **scrubbed** of `TELEGRAM_BOT_TOKEN`, every other
  key defined in `.env`, and anything whose name looks like a credential.
  Without this, one `env` prints the bot token into a chat. Note the limit of
  this control: `cat .env` still reads the file off disk, because `exec` is
  unrestricted by design. Secrets on disk are protected by the approval gate,
  not by scrubbing.
- Every call is approval-gated (below).

### `tools/files.py`

`read_file(path)` and `write_file(path, content)`. Neither prompts for
approval, so both resolve their argument against `WORKSPACE_ROOT/<chat>/<session>/`
and reject anything that escapes it *after* symlink resolution. This check
is the load-bearing security control of the whole design: without it,
`write_file("../../.ssh/authorized_keys")` bypasses the confirmation gate
entirely.

They exist because a 1.7B model handles a `write_file` tool call far better
than it handles heredoc quoting inside `exec`.

### `approvals.py`

Every `exec` call sends the command to the chat with an inline keyboard:
**Allow** / **Deny** / **Always allow this pattern**. The actor blocks on a
`threading.Event`; the router's `callback_query` branch sets it.

- **Always allow this pattern** stores a pattern in SQLite so repeats stop
  asking. *(Pattern granularity — exact string, first token, prefix glob —
  is a detail to settle during implementation.)*
- **`/auto`** suspends confirmation for the rest of the session. Reset by
  `/new`.
- **Approval timeout** — after `APPROVAL_TIMEOUT_SECONDS` with no tap, the
  request is auto-denied, the chat is told the turn expired, and the actor
  is freed. Without this a forgotten prompt pins a chat forever.
- **Deny** — *semantics open (see Q1)*.

### `session.py` and `store.py`

A session is the unit `/new` resets. SQLite holds:

| Table | Contents |
|---|---|
| `sessions` | chat_id, active session id, auto-approve flag, token-ratio calibration |
| `messages` | session_id, seq, role, content, tool_calls, tool_call_id |
| `approvals` | pattern, scope, created_at |
| `traces` | session_id, ts, raw request, raw response |

Connections are per-thread with WAL mode enabled — `sqlite3` connections
are not shareable across threads, and each actor is its own thread.

`traces` is not optional in practice: it is the only way to answer "why did
the model do that" when debugging a small model.

**`/new`** starts a fresh session: new context, new empty workspace
directory, auto-approve off, session-scoped always-allow patterns cleared.
Because the old workspace is discarded, workspaces are keyed per *session*,
not per chat. The old directory is renamed to an archive path rather than
recursively deleted — a recursive delete on a path built from ids is not
worth getting wrong on a host-unrestricted bot.

Other commands: `/stop` (cancel the running turn), `/status` (tokens used,
iterations, auto-approve state), `/skills`, `/skill <name>`, `/compact`.

### `context.py` — token accounting and compaction

**Counting:** estimate as `len(text) / ratio`, where `ratio` starts at 4.0
and is corrected after every response from the backend's reported
`usage.prompt_tokens`. No dependency, no tokenizer coupling, and it
converges on accurate for whatever model is actually loaded. The ratio is
stored per model so it survives restarts and re-learns on `LLM_MODEL` change.

**Compaction** runs when the estimate exceeds `COMPACT_THRESHOLD_PCT` of
`LLM_CONTEXT_TOKENS`, in three tiers, stopping as soon as it is under:

1. **Squeeze tool outputs.** Replace the bodies of old tool results with
   `[N bytes elided]`, keeping the command that produced them. Free, no
   model call, and tool output is where the bulk of the growth is.
2. **Trim.** Drop the oldest turns. The system prompt and the first user
   message of the session are pinned and never evicted.
3. **Summarize.** Ask the model to summarize the dropped span and splice
   the summary in. Last resort, because summary quality at 1.7B is poor.

Separately and always: every tool result is capped on the way *in* (see
`EXEC_MAX_OUTPUT_BYTES`). That cap alone removes most of the pressure; the
tiers above handle what is left. *(Whether compaction may run mid-loop or
only between turns is open, see Q6.)*

### `skills.py` and `tools/skill.py`

Skills live at `SKILLS_DIR/<name>/SKILL.md`, with optional sibling scripts
the agent invokes through `exec` (which means running a skill's script goes
through the normal approval gate).

`SKILL.md` is frontmatter plus a Markdown body:

```markdown
---
name: check-disk
description: Investigate disk usage and find large files.
---

<instructions the model follows>
```

*(Whether to also honor an `allowed-tools` key is open, see Q7.)*

**Progressive disclosure:** only each skill's `name` and `description` go
into the system prompt. A `skill(name)` tool loads the full body on demand
and appends it to the context. `/skill <name>` forces one manually — both
an override for when the model picks wrong, and a way to measure whether
the model's own choosing works at all.

### `telegram_client.py`

Gains: `reply_markup` support on `sendMessage` for inline keyboards, and
`answerCallbackQuery` so the tapped button stops spinning. Existing
chunking, empty-reply fallback, and token redaction are unchanged.

**Progress reporting:** one message per tool call — the command, then its
truncated result — followed by the final answer. Noisy in the chat, and
exactly what you want while debugging a small model's decisions.

## Data flow (one turn)

1. Router receives a `message`, checks `from.id` against the allowlist,
   enqueues it on the chat's actor.
2. Actor picks it up, loads the session, compacts if needed.
3. Agent loop calls the model with tool schemas + skill index.
4. Model returns `exec("ls -la")`. Agent posts an inline keyboard and blocks.
5. User taps **Allow**; the router resolves the pending approval and the
   actor wakes.
6. Command runs in a killable child with a scrubbed env; output is captured
   and truncated; a progress message is posted.
7. Result is appended to the history and persisted; loop returns to 3.
8. Model returns content with no tool calls → sent as the final answer.
9. Session is persisted; actor takes the next queued message.

## Error handling summary

Extends the original table:

| Failure | Behavior |
|---|---|
| Update from a non-allowlisted user | Rejected before any processing *(exact behavior open, Q2)* |
| Model emits an unparseable tool call | Counted as a tool failure; loop retries with the error appended |
| Loop exceeds `MAX_LOOP_ITERATIONS` | Breaker message; partial work stays in history |
| Tool fails `MAX_CONSECUTIVE_TOOL_FAILURES` times | Breaker message; turn ends |
| `exec` exceeds `EXEC_TIMEOUT_SECONDS` | Child killed; timeout reported to the model as the tool result |
| Approval never answered | Auto-denied at `APPROVAL_TIMEOUT_SECONDS`; actor freed |
| `write_file` path escapes the workspace | Rejected; returned to the model as a tool error |
| Context overflow despite compaction | Trim harder and retry once; then fail the turn with a clear message |
| SQLite write fails | Logged; the turn continues in memory (never lose a reply over a DB error) |
| Actor thread dies unexpectedly | Logged; actor recreated on the chat's next message |

## Build order

Each phase leaves the bot working:

1. **Sessions.** SQLite, session history, `/new`, `/status`. The bot becomes
   conversational with no new attack surface and no loop yet.
2. **The loop.** Actors, `agent.py`, tool registry, `exec` with approvals,
   file tools with the workspace guard, per-tool-call progress.
3. **Breakers and traces.** The three budgets, the `traces` table.
4. **Compaction.** Token estimation with usage correction, the three tiers.
5. **Skills.** Discovery, the skill index, the `skill` tool, `/skill`.

Skills come last because they are the least useful until the loop is solid,
and their success depends on a loop you already trust.

## Open questions

1. **Deny semantics.** Does Deny abort the turn outright, or does the model
   get "the user denied that command" and one more attempt at a different
   approach (charged against the tool-failure budget)?
2. **Non-allowlisted users.** Silent drop, or an explicit "not authorized"
   reply? Silence leaks less about the bot's existence.
3. **Group chats.** Supported at all, or DMs only? Determines whether the
   actor keys on chat id or user id.
4. **Native-tools detection.** Startup probe, config flag, or purely
   opportunistic (parse text whenever `tool_calls` is absent)?
5. **`exec` limits.** Final values for timeout and output cap, and whether
   stderr is returned separately or merged into stdout.
6. **Compaction trigger.** The threshold percentage, and whether compaction
   may run mid-loop or only between user turns.
7. **Skill frontmatter.** Mirror Claude Code's `name` / `description` /
   `allowed-tools`, or keep just name and description?
8. **Actor lifecycle.** Threads live for the process's lifetime, or are
   reaped after an idle period?

## Testing (manual)

- **Allowlist:** message the bot from a non-allowlisted account, confirm the
  rejection path; confirm an allowlisted account still works.
- **Approval:** trigger an `exec`, confirm the keyboard appears; test Allow,
  Deny, Always-allow (then confirm the repeat does not ask), `/auto`, and
  letting one expire.
- **Workspace escape:** prompt the model toward
  `write_file("../../escape.txt")`, confirm it is rejected and reported as a
  tool error.
- **Env scrubbing:** ask the bot to `cat .env`, confirm no bot token appears.
- **Breakers:** each of the three, confirming the distinct message and that
  partial work survives into the next turn.
- **Isolation:** leave an approval pending in one chat, confirm a second
  chat still gets replies.
- **FIFO:** send a second message mid-turn, confirm it is answered after the
  first turn completes, not merged into it.
- **Compaction:** run a long tool-heavy session past the threshold, confirm
  each tier fires in order and the conversation stays coherent.
- **`/new`:** confirm context, workspace, and auto-approve all reset.
- **Restart:** restart mid-session, confirm history is intact and an
  in-flight turn is cleanly abandoned.
