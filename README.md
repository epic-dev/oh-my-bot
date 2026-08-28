# oh-my-bot

A tiny Telegram bot that forwards messages to a local LLM and replies with the answer. No SDKs — raw HTTP to both Telegram and the LLM. The LLM connector speaks the OpenAI-compatible chat-completions API, so it works with MLX, Ollama, or vLLM by changing config only, never code.

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
```

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

```
oh-my-bot/
├── .env                  # gitignored
├── pyproject.toml
├── README.md
└── src/
    └── oh_my_bot/
        ├── __init__.py
        ├── config.py
        ├── telegram_client.py
        ├── llm_client.py
        ├── worker.py
        └── main.py
```

| File | Responsibility |
|---|---|
| `src/oh_my_bot/config.py` | Loads and validates settings from `.env` |
| `src/oh_my_bot/telegram_client.py` | Raw HTTP calls to the Telegram Bot API |
| `src/oh_my_bot/llm_client.py` | OpenAI-compatible LLM connector |
| `src/oh_my_bot/worker.py` | Thread pool, per-chat message ordering, and running each LLM call in its own killable subprocess |
| `src/oh_my_bot/main.py` | The long-poll loop wiring everything together |

Each LLM call runs in its own OS process so a hung or slow request can be killed on timeout without affecting other users — it doesn't just get abandoned like a stuck thread would.

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

- Conversation history is kept per chat in SQLite and survives restarts; `/new` starts a fresh session. Only allowlisted Telegram user ids (`ALLOWED_USER_IDS`) may use the bot, and only in private chats.
- A burst of more than `MAX_WORKERS` messages from one chat can briefly delay replies to other chats (the per-chat lock holds a thread-pool slot for the LLM call's full duration).
- `mlx_lm.server` is not recommended for production use as-is (per its own startup warning) — fine for personal/local use.
