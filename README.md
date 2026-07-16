# bring-your-own-model — run Claude Code on any model

Claude Code speaks one wire protocol: the Anthropic Messages API. This is a
tiny local proxy that translates that protocol **to and from OpenAI** (the
Responses API and Chat Completions), so you can point Claude Code at GPT-5.6
Sol, gpt-4o, or any OpenAI-compatible endpoint — and drive the full agentic
harness with a different model underneath.

Tools, streaming, reasoning, the whole loop. Claude Code never knows.

Zero runtime dependencies — just Node's built-in `http` and two pure
translator modules. The whole thing is a few hundred lines you can read.

The demo is GPT-5.6 Sol reading a failing test, reasoning about *why* it fails,
and fixing its own code — live inside Claude Code, over the OpenAI wire:

**[▶ Watch the demo](demo/cc-sol-live-cap.mp4)** — `demo/cc-sol-live-cap.mp4`

## Quickstart

```bash
npm install
npm run build

# your OpenAI key (or any OpenAI-compatible provider's)
export OPENAI_API_KEY=sk-...

# start the bridge, forcing every request to gpt-5.6-sol
node dist/cli.js --model gpt-5.6-sol --port 8788
```

Then point Claude Code at it — in a separate shell:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8788
export ANTHROPIC_API_KEY=any-placeholder
claude
```

Use Claude Code exactly as normal. Every turn now runs on your chosen model.

> The package is `bring-your-own-model`; after `npm install -g .` (or once
> published) the CLI is just `byom --model … --port …`. `ANTHROPIC_API_KEY`
> can be any non-empty string — the bridge never forwards it; it authenticates
> to the provider with your `OPENAI_API_KEY`.

## Flags

```
byom --model <id> [options]
```

| Flag              | Default                     | Meaning                                                        |
| ----------------- | --------------------------- | -------------------------------------------------------------- |
| `--model <id>`    | *(required)*                | The model every request routes to, e.g. `gpt-5.6-sol`, `gpt-4o`. |
| `--fast-model <id>` | *(unset)*                 | Cheaper model for Claude Code's haiku-tier sub-agent requests (Explore/Task fan-outs), e.g. `gpt-4o`. Unset = everything runs on `--model`. |
| `--port <n>`      | `8788`                      | Port the bridge listens on.                                    |
| `--base-url <url>`| `https://api.openai.com/v1` | OpenAI-compatible base URL (Groq, OpenRouter, LiteLLM, Ollama). |
| `-v`, `--verbose` | off                         | Log each request's target model + usage / reasoning tokens.    |
| `-h`, `--help`    | —                           | Show help.                                                     |

`OPENAI_API_KEY` is read from the environment (required). For a local provider
that needs no key, set it to any non-empty placeholder.

A per-request override is supported: prefix the request model with
`openai:<model>` (or `groq:` / `openrouter:` / `compat:` / `local:`) to route
that one turn to a different model id. Otherwise `--model` always wins (with
`--fast-model`, haiku-tier requests route to the cheap model instead) — this
bridge only ever speaks to the configured OpenAI-compatible backend.

Why `--fast-model`: Claude Code deliberately runs its throwaway sub-agent
turns (Explore, Task fan-outs) on its cheap haiku tier. With a single forced
`--model`, those all silently upgrade to your premium model — on a fan-out
task that multiplies cost for work that never needed the big model.

## Which models

- **GPT-5.6 Sol / gpt-5.x / o-series** on api.openai.com → OpenAI **Responses
  API** (reasoning + tools together). This is the path in the demo, tested end
  to end.
- **gpt-4o and any OpenAI-compatible endpoint** (Groq, OpenRouter, LiteLLM, a
  local Ollama exposing `/v1`) → **Chat Completions**. Point `--base-url` at the
  provider. Implemented and unit-tested, but only lightly exercised live —
  issues / PRs welcome.

The bridge auto-picks the surface: Responses when the base URL is
api.openai.com **and** the model matches `gpt-5*` / `o<n>*`; Chat Completions
otherwise.

## How it works

Claude Code sends `POST /v1/messages` (Anthropic Messages). The bridge:

1. Translates the request to OpenAI's shape (system → instructions, `tool_use`
   to/from `tool_calls`/`function_call`, images, tool results, `thinking` →
   `reasoning`, `max_tokens` with reasoning headroom, …).
2. Forwards it to the provider with **your** key.
3. Translates the reply back to Anthropic Messages — non-streaming and the full
   streaming SSE state machine (text, tool-call arguments, reasoning), so Claude
   Code's UI, tool loop, and token accounting all work unchanged.

Two translators (`src/translate-responses.ts`, `src/translate-chat.ts`) are
pure functions with offline unit tests. `src/proxy.ts` is the HTTP server that
wires them to the network; `src/cli.ts` is the entry point. No dependencies at
runtime.

## "How do I know it's really the other model?"

Claude Code's harness tells the model *"you are Claude,"* so if you ask it, it
will say it's Claude — the model faithfully adopts the harness identity. Don't
trust its self-report; trust the wire:

- Run with `-v` and watch every request go to the provider with the real model
  id, and the usage line report `reasoning=<n>` — a field the Anthropic API
  never emits.
- Pull the key mid-session and Claude Code stops. There is no Claude underneath.
- Or check your provider's own usage dashboard.

## Limitations

- Prompt caching differs by provider; long sessions build large contexts that
  can hit a provider's per-minute rate limits — start a fresh session to shrink
  it.
- `count_tokens` is estimated, not exact (no tokenizer dependency).
- Images inside tool results are dropped (provider tool messages are text-only).
- Assistant `thinking` blocks from prior turns are not replayed upstream (no
  faithful inbound slot); reasoning the model returns *is* surfaced.

## Develop

```bash
npm install
npm run build        # tsc → dist/
npm test             # builds, then runs the translator + smoke tests
npm run dev -- --model gpt-5.6-sol   # run from source via tsx
```

The test suite is offline: pure translator unit tests plus a server smoke test
that stubs `fetch` with canned OpenAI payloads — no API key, no network.

## License

MIT © Thomas Sprayberry
