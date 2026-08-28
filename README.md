# oh-my-bot

A Telegram bot that is an AI agent for your own machine. It runs a model/tool loop until the model has an answer: it can execute shell commands (with your per-command approval), read and write files in a per-conversation workspace, and follow skills you write as Markdown. Conversations persist in SQLite and compact themselves when they outgrow the context window.

No SDKs — raw HTTP to both Telegram and the LLM. The connector speaks the OpenAI-compatible chat-completions API, so it works with MLX, Ollama, or vLLM by changing config only, never code.

**It runs commands on the host, unsandboxed, by design.** The trust boundary is a Telegram user allowlist plus confirmation of every command before it runs. Read "Security model" below before pointing it at anything you care about.

## Prerequisites

- Python 3.9+
- [`uv`](https://docs.astral.sh/uv/)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- One of: an Apple Silicon Mac (MLX), Ollama installed (any platform), or a CUDA GPU (vLLM) — see "Start a local LLM server" below

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure `.env`

Create a `.env` file in the project root (already gitignored):

```bash
TELEGRAM_BOT_TOKEN=your-token-from-botfather
ALLOWED_USER_IDS=123456789
```

Both are required and the bot refuses to start without them. `ALLOWED_USER_IDS` is a
comma-separated list of numeric Telegram **user** ids — message
[@userinfobot](https://t.me/userinfobot) to find yours. It is the user id, not a chat id:
in a group those differ, and a chat-id allowlist would authorise every member of that group.
Everyone else is ignored in silence.

Optional overrides (defaults shown):

```bash
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=qwen3:1.7b
MAX_WORKERS=4
LLM_TIMEOUT_SECONDS=60
POLL_TIMEOUT_SECONDS=30
REASONING_TAGS=think,thinking,reasoning,thought,reflection,scratchpad
STOP_SEQUENCES=<|im_end|>,<|endoftext|>,<|eot_id|>,<end_of_turn>
LLM_MAX_TOKENS=2048
LLM_CONTEXT_TOKENS=16384
COMPACT_THRESHOLD_PCT=75
DB_PATH=./oh-my-bot.db
WORKSPACE_ROOT=./workspaces
SKILLS_DIR=./skills
MAX_LOOP_ITERATIONS=5
MAX_LLM_RETRIES=3
MAX_CONSECUTIVE_TOOL_FAILURES=3
EXEC_TIMEOUT_SECONDS=30
EXEC_MAX_OUTPUT_BYTES=8192
APPROVAL_TIMEOUT_SECONDS=600
```

### 3. Start a local LLM server

Pick one backend. All three speak the same OpenAI-compatible `/chat/completions` API, so the app's code never changes — only `.env`.

#### Option A: MLX — for Apple Silicon

Install `mlx-lm` as a standalone tool (keeps it out of the app's own dependencies):

```bash
uv tool install mlx-lm
```

Start the server, pointing at a Hugging Face MLX model repo — **not** an Ollama-style name like `qwen3:1.7b`, `mlx_lm.server` needs the actual repo id:

```bash
mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --port 8080
```

First run downloads the model from Hugging Face; it's cached after that. Leave this running in its own terminal.

Set `.env` to match:

```bash
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=mlx-community/Qwen3-1.7B-4bit
```

#### Option B: Ollama — any platform, easiest setup

Install it from [ollama.com/download](https://ollama.com/download), or on macOS:

```bash
brew install ollama
```

Start the server (installers usually already run this as a background service — only needed if it's not already running):

```bash
ollama serve
```

In another terminal, pull a model. Ollama's tag-style naming (`qwen3:1.7b`) is what `config.py`'s own default expects, so this is the lowest-friction option:

```bash
ollama pull qwen3:1.7b
```

Set `.env`:

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:1.7b
```

#### Option C: vLLM — Linux + NVIDIA GPU, best throughput

vLLM needs a CUDA-capable GPU — not an option on Apple Silicon. Install and run:

```bash
uv tool install vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000
```

First run downloads the model from Hugging Face (some models require accepting a license on Hugging Face first and setting `HF_TOKEN`). Set `.env`:

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
```

### 4. Run the bot

```bash
uv run oh-my-bot
```

(equivalently: `uv run python -m oh_my_bot.main`)

It long-polls Telegram, forwards each message to the LLM server, and replies. Message your bot on Telegram to try it.

## Stopping everything

If you ran the bot and/or the LLM server in the foreground (as shown above), just `Ctrl-C` in each terminal.

If something's running in the background instead, find and stop it by process name:

```bash
# The bot
pgrep -fl "oh_my_bot.main"
kill <pid>

# MLX
pgrep -fl "mlx_lm.server"
kill <pid>

# vLLM
pgrep -fl "vllm serve"
kill <pid>
```

Ollama is different — `ollama serve` usually runs as a persistent background service (started by the installer via `launchd` on macOS, `systemd` on Linux), not something you start/stop per session:

```bash
# macOS (Homebrew service)
brew services stop ollama

# Linux (systemd)
sudo systemctl stop ollama

# Or, if you started `ollama serve` manually in a terminal
killall ollama
```

## Swapping in another model

### Another MLX model (e.g. TinyLlama)

1. Stop `mlx_lm.server` (Ctrl-C).
2. Start it with the new model's repo id:
   ```bash
   mlx_lm.server --model mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit --port 8080
   ```
3. Update `.env`:
   ```bash
   LLM_MODEL=mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit
   ```
4. Restart the bot (`Ctrl-C` then `uv run oh-my-bot`) so it picks up the new `.env` value.

Browse [mlx-community on Hugging Face](https://huggingface.co/mlx-community) for other pre-converted models — quantized ones (`-4bit`, `-8bit`) run faster and use less memory.

### Reasoning models

Models like Qwen3 and DeepSeek-R1 return their chain of thought inline in the reply, wrapped in a
tag. The bot strips those blocks so they reach neither the user nor the stored conversation
history (where they would eat the context window across turns). The tag name differs by model, so
it is configuration rather than code:

```bash
REASONING_TAGS=think,reasoning        # only strip these two
REASONING_TAGS=                       # strip nothing
```

Set it to an empty value if you ever want to see the raw reasoning while debugging.

### Generation budget

`mlx_lm.server` caps a reply at **512 tokens** by default, and Ollama has its own limit. That is
not enough for a reasoning model: Qwen3 can spend 500+ tokens thinking before it writes a word of
the answer, so generation stops mid-thought and the reply cleans down to nothing. `LLM_MAX_TOKENS`
(default 2048) is sent as `max_tokens` to prevent that; set it to `0` to defer to the server.

If a reply still comes back empty with `finish_reason: length`, the model used its whole budget
thinking. Raising `LLM_MAX_TOKENS` helps; for Qwen3 specifically, appending `/no_think` to a
message disables thinking for that turn — in local testing the same question took 684 tokens with
thinking and 44 without.

### Context window and compaction

`LLM_CONTEXT_TOKENS` must match the **model's** window, not a backend default — the two models in
this README differ by 20x:

```bash
# Read it straight from the model's own config
python3 -c "import json,glob;print(json.load(open(glob.glob('$HOME/.cache/huggingface/hub/models--*Qwen3-1.7B*/snapshots/*/config.json')[0]))['max_position_embeddings'])"
```

| Model | Max context |
|---|---|
| TinyLlama 1.1B | 2,048 |
| Llama 3 8B / Gemma 2 | 8,192 |
| Qwen2.5 / Mistral 7B v0.3 | 32,768 |
| Qwen3 (all sizes) | 40,960 |
| Llama 3.1+ / Gemma 3 4B+ | 131,072 |

Don't simply set the maximum: on Apple Silicon the practical limit is KV-cache memory. For
Qwen3-1.7B that is roughly 0.94 GB at 8k, 1.9 GB at 16k, and 4.7 GB at 40k.

When the estimated prompt passes `COMPACT_THRESHOLD_PCT` of the window, the history is compacted
in three tiers — old tool outputs are elided, then the oldest turns are dropped, then what was
dropped is summarized back in. It compacts to below the trigger rather than exactly to it, so the
next few turns do not each pay for another compaction.

**These three settings interact.** Whatever the threshold leaves over has to hold the reply:

```
LLM_CONTEXT_TOKENS x (1 - COMPACT_THRESHOLD_PCT/100)  >=  LLM_MAX_TOKENS
```

The bot warns at startup when that is violated, because the symptom otherwise looks like a model
problem rather than a configuration one.

### End-of-turn markers

Chat templates delimit turns with special tokens — `<|im_end|>` for ChatML/Qwen, `<|eot_id|>` for
Llama 3, `<end_of_turn>` for Gemma. The tokenizer should stop there and never show them to you. If
one appears in a reply, the server failed to stop, and whatever follows it is usually a *fabricated
next turn* rather than part of the answer.

The bot handles this twice over: it sends `STOP_SEQUENCES` to the server as the `stop` parameter,
and if a marker still comes back it truncates the reply at that point.

```bash
STOP_SEQUENCES=<|im_end|>       # only this marker
STOP_SEQUENCES=                 # disable truncation
```

Leaked `<|...|>` control tokens are always removed from replies regardless of this setting — they
are never legitimate answer text. The unmodified response is still recorded in the `traces` table.

### Switching backends entirely (MLX ↔ Ollama ↔ vLLM)

See "Start a local LLM server" above for each backend's setup. Switching is just: stop the old server, start the new one, update `LLM_BASE_URL`/`LLM_MODEL` in `.env` to match, and restart the bot. No code changes, ever — that's the point of the connector's config-only design.

## How it works

Each message runs as one **turn**: the model is called with the available tools, any tool calls it
makes are executed and their results fed back, and the loop repeats until it produces an answer.

```
Telegram ──poll──▶ main.py (router)
                     ├── button tap ──▶ approvals.py ──wakes──▶ the blocked actor
                     └── message ─────▶ actors.py (one thread per chat)
                                            └─▶ agent.py: the loop
                                                  ├─▶ context.py   compact if needed
                                                  ├─▶ llm_client.py ──▶ worker.py (killable subprocess)
                                                  ├─▶ tools/  exec | read_file | write_file | skill
                                                  └─▶ store.py     history + traces
```

```
oh-my-bot/
├── .env                    # gitignored
├── oh-my-bot.db            # gitignored: sessions, history, approvals, traces
├── skills/<name>/SKILL.md  # your skills
├── workspaces/<chat>/<session>/   # gitignored: where file tools and exec run
└── src/oh_my_bot/
    ├── config.py  session.py  store.py  actors.py  agent.py
    ├── approvals.py  context.py  skills.py
    ├── llm_client.py  telegram_client.py  worker.py  main.py
    └── tools/  base.py  exec.py  files.py  skill.py
```

| File | Responsibility |
|---|---|
| `config.py` | Settings from `.env`, and a startup check that the token budgets are consistent |
| `main.py` | Long-poll loop and router: taps to approvals, messages to actors |
| `actors.py` | One long-lived thread and queue per chat |
| `agent.py` | The loop and its three circuit breakers |
| `approvals.py` | Inline-keyboard confirmation, always-allow patterns, timeouts |
| `tools/` | The tool registry, `exec`, the workspace-scoped file tools, and `skill` |
| `context.py` | Token estimation and three-tier compaction |
| `skills.py` | `SKILL.md` discovery and frontmatter parsing |
| `session.py` | Per-chat sessions, history, `/new` and the other commands |
| `store.py` | SQLite: sessions, messages, approvals, token ratios, traces |
| `llm_client.py` | The OpenAI-compatible connector and reply cleaning |
| `worker.py` | Runs one LLM call in its own killable subprocess |
| `telegram_client.py` | Raw HTTP to the Telegram Bot API |

**Why one thread per chat rather than a thread pool.** A turn blocks while it waits for you to tap
Allow, and that wait is unbounded. Under a pool, a pending approval would hold a worker slot for as
long as you took to answer. Each chat now owns its thread, so a chat waiting on you delays only
itself. Messages that arrive mid-turn queue and run in order.

**Why each LLM call still gets its own process.** A hung or slow request can be killed on timeout
rather than abandoned like a stuck thread. `exec` gets the same treatment, in its own process group,
so a timeout also kills anything the command spawned.

**Circuit breakers.** Three independent budgets stop a turn that is going nowhere, each with its own
message so a dead turn explains itself: `MAX_LOOP_ITERATIONS` model→tool rounds,
`MAX_LLM_RETRIES` transport failures, and `MAX_CONSECUTIVE_TOOL_FAILURES` same-tool errors in a row.
Transport retries do not consume loop iterations — a flaky server should not eat your reasoning
steps. Partial work stays in the history, so a follow-up message continues from where it stopped.

## Security model

`exec` runs commands on this machine as you, with no sandbox. That is deliberate — it is what makes
the bot useful — so the boundary is elsewhere:

- **Only allowlisted Telegram user ids are answered at all**, and only in private chats. Button taps
  are checked against the same allowlist, or anyone could approve your pending command by guessing
  its request id.
- **Every `exec` command is confirmed before it runs**, with Allow / Deny / Always-allow-*program*.
  "Always allow `ls`" permits any later `ls`, never `rm`. `/auto` suspends confirmation until
  `/new`. An unanswered request auto-denies after `APPROVAL_TIMEOUT_SECONDS`.
- **`read_file` and `write_file` are never confirmed**, so they are code-enforced to the session
  workspace — every path is resolved, symlinks included, and anything outside is refused. This is
  what stops the unconfirmed tools being an easier route around the gate.
- **Secrets are stripped from the environment** of every command: keys defined in `.env`, plus
  anything whose name looks like a credential. The limit of this control is worth knowing: `cat .env`
  still reads the file off disk. Secrets on disk are protected by the approval gate, not by
  scrubbing — so think before approving a command that reads them, or keep `.env` elsewhere.

## Commands

| Command | Effect |
|---|---|
| `/new` | Fresh session: new context, new empty workspace, confirmations back on. The old workspace is archived, not deleted |
| `/status` | Session id, message count, auto-approve state, workspace path |
| `/auto` | Stop confirming commands until `/new` |
| `/compact` | Force a compaction pass now |
| `/skills` | List installed skills |
| `/skill <name>` | Load a skill into the conversation directly |

## Skills

A skill is a folder in `skills/` holding a `SKILL.md` of instructions, plus any scripts it needs:

```
skills/
└── check-disk/
    ├── SKILL.md
    └── (optional scripts the agent can run through exec)
```

```markdown
---
name: check-disk
description: Investigate disk usage and find what is taking up space.
---

Instructions the model should follow for this kind of task.
```

Only the **name and description** of each skill go into the system prompt. The model calls the
`skill` tool to load the full instructions when it decides a task needs them — so adding skills
costs almost nothing in context until one is used. Scripts a skill bundles are run through `exec`,
which means they still pass the normal approval gate; a bundled script is arbitrary code like any
other.

`/skills` lists what is installed. `/skill <name>` loads one into the conversation directly,
bypassing the model's choice — useful when it picks wrong, and the way to tell whether it is
choosing well at all.

**On small models:** whether a 1.7B model reaches for the right skill unaided is genuinely
uncertain. Qwen3-1.7B tends to call a tool *named after* the skill instead of calling `skill` with
that name, so the bot treats such a call as the skill load it obviously meant. If it still never
picks a skill on its own, use `/skill <name>` and treat progressive disclosure as a bigger-model
feature.

## Debugging

Every model request and response is recorded in the `traces` table, including calls that failed.
When the bot does something inexplicable, this is what tells you what the model actually saw and
actually said — rather than what the Telegram messages imply.

Most recent exchanges, newest first (`failed` is non-empty when the call did not succeed):

```bash
sqlite3 oh-my-bot.db "
SELECT t.id, s.chat_id, json_extract(t.response, '\$.status') AS failed
FROM traces t JOIN sessions s USING (session_id) ORDER BY t.id DESC LIMIT 10;"
```

The full raw response of the last call — where you can see whether the model emitted native
`tool_calls`, wrote a fenced block instead, or spent its whole budget inside `<think>`:

```bash
sqlite3 oh-my-bot.db "SELECT response FROM traces ORDER BY id DESC LIMIT 1;"
```

The shape of the last prompt, which is how you spot a malformed history (an orphaned `tool`
message, or a conversation that has grown past the context window):

```bash
sqlite3 oh-my-bot.db "
SELECT json_extract(value, '\$.role')
FROM traces, json_each(json_extract(traces.request, '\$.messages'))
WHERE traces.id = (SELECT MAX(id) FROM traces);"
```

Traces are never pruned automatically and they store full prompts, so the database grows with use.
To reclaim space:

```bash
sqlite3 oh-my-bot.db "DELETE FROM traces WHERE created_at < strftime('%s','now','-7 days'); VACUUM;"
```

## Known limitations

- **`exec` is unsandboxed by design.** The allowlist and per-command approval are the only
  boundary. Approving a destructive command runs it.
- **Private chats only.** Group chats are dropped; a host-unrestricted agent in a group is a much
  larger trust surface.
- **A turn in flight is abandoned on restart.** Sessions and history persist, but a turn that was
  mid-tool when the process died is not resumed.
- **Small models choose tools poorly.** Qwen3-1.7B calls a tool named after a skill rather than
  calling `skill` with that name (handled), and may ignore tools entirely. `/skill <name>` is the
  override. Progressive disclosure is more reliable on bigger models.
- **Reasoning is expensive at small sizes.** Qwen3 can spend 500+ tokens thinking before answering.
  Appending `/no_think` to a message skips it.
- **No streaming.** Progress is reported per tool call, not per token.
- **Traces are never pruned** — see Debugging for the cleanup query.
- **`mlx_lm.server` is not recommended for production** (per its own startup warning) — fine for
  personal/local use.
