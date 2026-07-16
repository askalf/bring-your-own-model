/**
 * bring-your-own-model (byom) — a minimal, dependency-free HTTP bridge that lets an
 * Anthropic-speaking client (Claude Code, on the `/v1/messages` Messages
 * API) drive an OpenAI / OpenAI-compatible model.
 *
 * The server owns three things and nothing more:
 *   1. Routing. `POST /v1/messages` and `POST /v1/messages/count_tokens`
 *      are the only translated routes; `GET /` is a health check; every
 *      other request gets an Anthropic-shaped 404.
 *   2. The gate (`resolveOpenAITarget`). Standalone always routes to the
 *      configured OpenAI-compatible backend — that is its only job — so the
 *      gate reduces to "which model id, and which OpenAI surface". The
 *      forced `--model` wins unless a single request overrides it with an
 *      `openai:<model>` prefix.
 *   3. The transport (`handleTranslatedOpenAI`). Translate the inbound
 *      Messages request to the Chat Completions or Responses shape, POST it
 *      to the backend with the operator's `OPENAI_API_KEY`, and translate
 *      the reply back to the Messages shape — non-streaming JSON or the full
 *      streaming SSE state machine, plus Anthropic-shaped errors.
 *
 * All the wire translation lives in the two pure, unit-tested modules
 * (`translate-chat.ts`, `translate-responses.ts`); this file is only plumbing.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

import {
  anthropicToOpenAIRequest,
  openAIToAnthropicResponse,
  openAIStreamToAnthropicSSE,
  parseOpenAISSELine,
  formatAnthropicSSE,
  estimateTokenCount,
  type AnthropicRequest,
  type OpenAIChatResponse,
} from './translate-chat.js';
import {
  anthropicToResponsesRequest,
  responsesToAnthropicResponse,
  responsesStreamToAnthropicSSE,
  createResponsesSSEParser,
  formatResponsesAnthropicSSE,
  type ResponsesResponse,
} from './translate-responses.js';

// ─────────────────────────────────────────────────────────────────────
// Config + constants
// ─────────────────────────────────────────────────────────────────────

export const DEFAULT_PORT = 8788;
export const DEFAULT_BASE_URL = 'https://api.openai.com/v1';
/** Upstream fetch ceiling. Reasoning models can think for a while. */
export const DEFAULT_UPSTREAM_TIMEOUT_MS = 600_000;
/** Reject request bodies larger than this (Claude Code contexts are big, but bounded). */
const MAX_BODY_BYTES = 64 * 1024 * 1024;
const LOG_PREFIX = '[byom]';

/** Minimal hardening headers echoed on every response. */
const SECURITY_HEADERS: Record<string, string> = {
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
};

export interface BridgeConfig {
  /** Forced target model — every request routes here (e.g. `gpt-5.6-sol`). */
  model: string;
  /**
   * Optional cheap model for haiku-tier requests. Claude Code deliberately
   * runs throwaway sub-agent (Explore/Task) turns on its haiku tier; with a
   * single forced `--model` those would all hit the premium model. When set,
   * an inbound request whose model matches /haiku/i routes here instead.
   * Unset → every request routes to `model` (unchanged behavior).
   */
  fastModel?: string;
  /** OpenAI-compatible base URL, e.g. `https://api.openai.com/v1`. */
  baseUrl: string;
  /** Backend API key (from `OPENAI_API_KEY`). */
  apiKey: string;
  /** Emit the `-v` proof log (per-request target + usage/reasoning tokens). */
  verbose?: boolean;
  /** Upstream fetch timeout in ms. */
  upstreamTimeoutMs?: number;
}

/** The backend a translated request is forwarded to. */
export interface BackendConfig {
  apiKey: string;
  baseUrl: string;
}

/** Resolved routing decision for one translated request. */
export interface OpenAITarget {
  /** Real model id (any recognized `openai:`-family prefix already stripped). */
  model: string;
  /** Which OpenAI surface to translate to. */
  api: 'responses' | 'chat';
  /**
   * Which routing tier picked `model`: `fast` when a haiku-tier request was
   * routed to `--fast-model`, `primary` otherwise (the `--model` default, or
   * an explicit per-request override). Surfaced in the `-v` log.
   */
  tier: 'fast' | 'primary';
}

// ─────────────────────────────────────────────────────────────────────
// Provider prefix + surface selection
// ─────────────────────────────────────────────────────────────────────

// A `<provider>:<model>` prefix on the request `model` forces a per-request
// model id. Only recognized prefixes are parsed, so an ollama-style
// `llama3:8b` (no recognized prefix) passes through untouched and is treated
// as a bare model name that reaches the configured backend as-is.
const PROVIDER_PREFIXES: Record<string, 'openai' | 'claude'> = {
  openai: 'openai',
  openrouter: 'openai',
  groq: 'openai',
  compat: 'openai',
  local: 'openai',
  claude: 'claude',
  anthropic: 'claude',
};

export function parseProviderPrefix(
  model: string,
): { provider: 'openai' | 'claude'; model: string } | null {
  const idx = model.indexOf(':');
  if (idx <= 0) return null;
  const prefix = model.slice(0, idx).toLowerCase();
  const provider = PROVIDER_PREFIXES[prefix];
  if (!provider) return null;
  const stripped = model.slice(idx + 1);
  if (!stripped) return null;
  return { provider, model: stripped };
}

/**
 * True when `baseUrl`'s host is api.openai.com (OpenAI proper), where the
 * reasoning-era models require the Responses shape and reject `max_tokens`.
 * Any parse failure → false (treat as a generic OpenAI-compat provider).
 */
export function isOpenAIProperBase(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).host === 'api.openai.com';
  } catch {
    return false;
  }
}

/** Reasoning-era OpenAI models (gpt-5*, o-series) — tools+reasoning need Responses. */
export function isReasoningEraModel(model: string): boolean {
  return /^(gpt-5|o[0-9])/.test(model.toLowerCase());
}

/**
 * Pick the OpenAI surface: the Responses API only when the backend is
 * api.openai.com AND the model is reasoning-era (gpt-5*, o-series) — that
 * combination needs Responses for tools+reasoning together. Every other
 * case is Chat Completions, the shape every OpenAI-compat provider serves.
 */
export function pickApi(baseUrl: string, model: string): 'responses' | 'chat' {
  return isOpenAIProperBase(baseUrl) && isReasoningEraModel(model) ? 'responses' : 'chat';
}

/**
 * GATE for the Anthropic→OpenAI translate path. Only the Messages family
 * (`/v1/messages`, `/v1/messages/count_tokens`) is eligible.
 *
 * This standalone bridge ALWAYS routes to the configured OpenAI-compatible
 * backend — so the forced `--model` (`forcedModel`) is the target for every
 * request. A single request can override just the model id by naming an
 * `openai:`-family prefix (`openai:` / `openrouter:` / `groq:` / `compat:` /
 * `local:`) in its `model` field. A `claude:` / `anthropic:` prefix, a bare
 * `claude-*` name, or no model at all all fall through to the forced model:
 * there is no Claude upstream here, so the operator's `--model` is honored.
 *
 * Tier-aware routing (`fastModel`): Claude Code deliberately runs its
 * throwaway sub-agent (Explore/Task) turns on the haiku tier and the main
 * conversation on opus/sonnet/fable. When a fast-model is configured, an
 * inbound request whose model matches /haiku/i is routed to that cheaper
 * model instead of the forced `--model`. An explicit per-request `openai:`
 * prefix still wins, and a misconfigured `claude:`-prefixed fast-model
 * degrades safely to the primary.
 */
export function resolveOpenAITarget(params: {
  path: string;
  model: string | undefined;
  forcedModel: string;
  baseUrl: string;
  fastModel?: string | null;
}): OpenAITarget | null {
  if (params.path !== '/v1/messages' && params.path !== '/v1/messages/count_tokens') {
    return null;
  }

  let realModel = params.forcedModel;
  let tier: 'fast' | 'primary' = 'primary';
  const reqModel = typeof params.model === 'string' ? params.model : '';
  const prefix = parseProviderPrefix(reqModel);
  if (prefix && prefix.provider === 'openai') {
    realModel = prefix.model; // explicit per-request OpenAI model override
  } else if (params.fastModel && /haiku/i.test(reqModel)) {
    // Haiku-tier request → the cheap fast-model. Tolerate an `openai:`-family
    // prefix on the flag value; a `claude:` prefix has no meaning here and
    // falls through to the primary.
    const fastPrefix = parseProviderPrefix(params.fastModel);
    const fast = fastPrefix
      ? (fastPrefix.provider === 'openai' ? fastPrefix.model : null)
      : params.fastModel; // bare model id
    if (fast) {
      realModel = fast;
      tier = 'fast';
    }
  }

  if (!realModel) return null;
  return { model: realModel, api: pickApi(params.baseUrl, realModel), tier };
}

// ─────────────────────────────────────────────────────────────────────
// Anthropic-shaped errors + verbose proof log
// ─────────────────────────────────────────────────────────────────────

/** An Anthropic-shaped error body (the shape Claude Code parses). */
function anthropicErrorBody(type: string, message: string): string {
  return JSON.stringify({ type: 'error', error: { type, message } });
}

/**
 * Anthropic-shaped error for a failed OpenAI-compat forward. No upstream
 * detail leaks to the client (server logs carry it under `-v`).
 */
function anthropicTranslateError(status: number): string {
  const type =
    status === 429 ? 'rate_limit_error'
    : status === 400 || status === 422 ? 'invalid_request_error'
    : status === 401 || status === 403 ? 'authentication_error'
    : status >= 500 ? 'api_error'
    : 'invalid_request_error';
  return anthropicErrorBody(type, `Upstream OpenAI-compat backend error (status ${status}).`);
}

/**
 * Verbose-only proof suffix: token counts (incl. the OpenAI-only
 * `reasoning_tokens` / `cached_tokens`) pulled from a raw upstream usage
 * object (Responses or Chat shape). reasoning_tokens is a field the
 * Anthropic API never emits — seeing it is proof the turn ran on OpenAI.
 */
function usageProofSuffix(u: unknown): string {
  if (!u || typeof u !== 'object') return '';
  const g = u as {
    input_tokens?: number; output_tokens?: number;
    prompt_tokens?: number; completion_tokens?: number;
    input_tokens_details?: { cached_tokens?: number };
    output_tokens_details?: { reasoning_tokens?: number };
    prompt_tokens_details?: { cached_tokens?: number };
    completion_tokens_details?: { reasoning_tokens?: number };
  };
  const inTok = g.input_tokens ?? g.prompt_tokens ?? '?';
  const outTok = g.output_tokens ?? g.completion_tokens ?? '?';
  const reasoning =
    g.output_tokens_details?.reasoning_tokens ?? g.completion_tokens_details?.reasoning_tokens ?? 0;
  const cached =
    g.input_tokens_details?.cached_tokens ?? g.prompt_tokens_details?.cached_tokens ?? 0;
  return `  [usage in=${inTok} out=${outTok} reasoning=${reasoning} cached=${cached}]`;
}

// ─────────────────────────────────────────────────────────────────────
// Translate-and-forward transport
// ─────────────────────────────────────────────────────────────────────

/**
 * Translate an inbound Anthropic Messages request, forward it to the
 * OpenAI-compat backend, and translate the reply back:
 *   - non-streaming → a single Anthropic Messages JSON (200).
 *   - streaming     → Anthropic SSE, produced incrementally from the upstream
 *                     SSE via the matching reverse streamer.
 * Client-abort and the upstream timeout share one AbortController; a non-2xx
 * or transport error becomes an Anthropic-shaped error (JSON before headers
 * are sent, a clean stream close after). Owns the whole response — the caller
 * returns immediately after it.
 */
export async function handleTranslatedOpenAI(
  req: IncomingMessage,
  res: ServerResponse,
  body: AnthropicRequest,
  target: OpenAITarget,
  backend: BackendConfig,
  opts: { corsOrigin: string; verbose: boolean; upstreamTimeoutMs: number; reqNum: number },
): Promise<void> {
  const { corsOrigin, verbose, upstreamTimeoutMs, reqNum } = opts;
  const stream = body.stream === true;
  const base = backend.baseUrl.replace(/\/$/, '');
  const openaiProper = isOpenAIProperBase(backend.baseUrl);
  const reasoningEra = isReasoningEraModel(target.model);

  // Translate the inbound Messages request into the chosen OpenAI shape.
  let url: string;
  let payload: Record<string, unknown>;
  if (target.api === 'responses') {
    url = `${base}/responses`;
    payload = anthropicToResponsesRequest(body, target.model) as unknown as Record<string, unknown>;
  } else {
    url = `${base}/chat/completions`;
    const oaReq = anthropicToOpenAIRequest(body, target.model, {
      // api.openai.com reasoning models reject `max_tokens` — emit
      // `max_completion_tokens` instead. (Reasoning models on openai.com route
      // to Responses, so on the chat path this only bites a non-openai.com
      // provider that also demands the newer field.)
      useMaxCompletionTokens: openaiProper && reasoningEra,
      // reasoning_effort is only valid on reasoning-era models — a
      // non-reasoning chat model (gpt-4o, …) 400s on the field, so gate it
      // on the CHOSEN model (a fast-tier gpt-4o sub-agent request omits it).
      emitReasoningEffort: reasoningEra,
    });
    const p = oaReq as unknown as Record<string, unknown>;
    // gpt-5.x (and the o-series) REJECT function tools together with reasoning
    // on /chat/completions. When a reasoning-era model reaches the chat path (a
    // non-openai.com provider serving one) with tools, force reasoning off so
    // the request is accepted. Non-reasoning models are left exactly as the
    // translator produced them — they must not carry a reasoning field they
    // don't support.
    if (reasoningEra && Array.isArray(oaReq.tools) && oaReq.tools.length > 0) {
      p.reasoning_effort = 'none';
    }
    payload = p;
  }

  // Swap in the backend key, forward only the OpenRouter attribution hints.
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${backend.apiKey}`,
    'Accept': stream ? 'text/event-stream' : 'application/json',
  };
  for (const h of ['x-title', 'http-referer', 'x-openrouter-app']) {
    const v = req.headers[h];
    if (typeof v === 'string') headers[h] = v;
  }

  // Client-abort + upstream timeout share one controller: a client disconnect
  // or the ceiling both abort the upstream fetch so we stop reading a response
  // nobody will consume.
  const abort = new AbortController();
  let clientClosed = false;
  const timeout = setTimeout(() => abort.abort(), upstreamTimeoutMs);
  const onClose = () => {
    clientClosed = true;
    try { abort.abort(); } catch { /* already aborting */ }
  };
  req.on('close', onClose);

  try {
    if (verbose) {
      console.log(`${LOG_PREFIX} #${reqNum} → openai translate (${target.api}) ${url} model=${target.model} tier=${target.tier} stream=${stream}`);
    }
    const upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: abort.signal,
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '');
      if (verbose) {
        console.error(`${LOG_PREFIX} #${reqNum} openai translate upstream ${upstream.status}: ${detail.slice(0, 300)}`);
      }
      // Headers are not sent yet (we buffer before writing), so even a stream
      // request can return a JSON error — Anthropic clients handle a non-200
      // JSON body on a streaming request.
      if (!res.headersSent) {
        res.writeHead(upstream.status, {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': corsOrigin,
          ...SECURITY_HEADERS,
        });
        res.end(anthropicTranslateError(upstream.status));
      } else if (!res.writableEnded) {
        try { res.end(); } catch { /* already closed */ }
      }
      return;
    }

    if (!stream) {
      const raw = await upstream.text();
      let parsed: unknown;
      try { parsed = JSON.parse(raw); } catch { parsed = {}; }
      const anthropicResp = target.api === 'responses'
        ? responsesToAnthropicResponse(parsed as ResponsesResponse, target.model)
        : openAIToAnthropicResponse(parsed as OpenAIChatResponse, target.model);
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': corsOrigin,
        ...SECURITY_HEADERS,
      });
      res.end(JSON.stringify(anthropicResp));
      if (verbose) {
        console.log(`${LOG_PREFIX} #${reqNum} openai translate (${target.api}) 200 (non-stream)${usageProofSuffix((parsed as { usage?: unknown }).usage)}`);
      }
      return;
    }

    // Streaming: Anthropic SSE, produced incrementally from the upstream SSE.
    res.writeHead(200, {
      ...SECURITY_HEADERS,
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'Access-Control-Allow-Origin': corsOrigin,
    });

    const reader = upstream.body?.getReader();
    const decoder = new TextDecoder();
    let rawStreamUsage: unknown = null; // captured off the terminal event for the verbose usage/reasoning proof line
    let terminalStatus: string | null = null; // 'completed' | 'incomplete' | 'failed' | error — surfaced so a silent empty turn is diagnosable
    if (target.api === 'responses') {
      // Responses SSE is a typed event stream — the buffered parser owns
      // framing (multi-line records, CRLF, cross-chunk boundaries).
      const parser = createResponsesSSEParser();
      const translator = responsesStreamToAnthropicSSE({ requestModel: target.model });
      if (reader) {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          for (const ev of parser.push(decoder.decode(value, { stream: true }))) {
            const et = (ev as { type?: string }).type;
            // Capture usage + status from ANY terminal event, not just
            // `completed` — `incomplete` (budget exhausted, often reasoning)
            // and `failed`/`error` (e.g. insufficient_quota) otherwise vanish
            // into a silent empty turn with no usage.
            if (et === 'response.completed' || et === 'response.incomplete' || et === 'response.failed') {
              const r = (ev as { response?: { usage?: unknown; status?: string; error?: { code?: string; message?: string }; incomplete_details?: { reason?: string } } }).response;
              rawStreamUsage = r?.usage ?? rawStreamUsage;
              // Prefer the concrete error message (rate-limit / context-length /
              // etc.) so `failed` isn't opaque; fall back to reason/status.
              terminalStatus = r?.error?.message ?? r?.error?.code ?? r?.incomplete_details?.reason ?? r?.status ?? et.split('.')[1] ?? et;
            } else if (et === 'error') {
              const e = ev as { message?: string; code?: string };
              terminalStatus = `error:${e.message ?? e.code ?? 'unknown'}`;
            }
            for (const aev of translator.push(ev)) res.write(formatResponsesAnthropicSSE(aev));
          }
        }
        for (const ev of parser.push(decoder.decode())) {
          for (const aev of translator.push(ev)) res.write(formatResponsesAnthropicSSE(aev));
        }
        for (const ev of parser.flush()) {
          for (const aev of translator.push(ev)) res.write(formatResponsesAnthropicSSE(aev));
        }
      }
      for (const aev of translator.end()) res.write(formatResponsesAnthropicSSE(aev));
    } else {
      // Chat-completions SSE is data-only lines; buffer to newline boundaries.
      const translator = openAIStreamToAnthropicSSE({ requestModel: target.model });
      let buffer = '';
      const pump = (final: boolean): void => {
        let nl: number;
        while ((nl = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          const chunk = parseOpenAISSELine(line);
          if (chunk?.usage) rawStreamUsage = chunk.usage;
          if (chunk) for (const aev of translator.push(chunk)) res.write(formatAnthropicSSE(aev));
        }
        if (final && buffer.trim().length > 0) {
          const chunk = parseOpenAISSELine(buffer);
          if (chunk) for (const aev of translator.push(chunk)) res.write(formatAnthropicSSE(aev));
          buffer = '';
        }
      };
      if (reader) {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          pump(false);
        }
        buffer += decoder.decode();
        pump(true);
      }
      for (const aev of translator.end()) res.write(formatAnthropicSSE(aev));
    }
    if (!res.writableEnded) res.end();
    const statusFlag = terminalStatus && terminalStatus !== 'completed' ? `  <<${terminalStatus}>>` : '';
    if (statusFlag) {
      console.warn(`${LOG_PREFIX} #${reqNum} openai translate (${target.api}) ended NON-completed: ${terminalStatus} — the client sees an empty/partial turn. (out-of-credits? raise max_tokens? upstream error?)`);
    }
    if (verbose) {
      console.log(`${LOG_PREFIX} #${reqNum} openai translate (${target.api}) stream complete${statusFlag}${usageProofSuffix(rawStreamUsage)}`);
    }
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    if (clientClosed || abort.signal.aborted) {
      if (verbose) console.log(`${LOG_PREFIX} #${reqNum} openai translate aborted (${clientClosed ? 'client closed' : 'upstream timeout'})`);
      if (!res.writableEnded) { try { res.end(); } catch { /* already closed */ } }
    } else {
      if (verbose) console.error(`${LOG_PREFIX} #${reqNum} openai translate error: ${detail}`);
      if (!res.headersSent) {
        res.writeHead(502, {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': corsOrigin,
          ...SECURITY_HEADERS,
        });
        res.end(anthropicTranslateError(502));
      } else if (!res.writableEnded) {
        try { res.end(); } catch { /* already closed */ }
      }
    }
  } finally {
    clearTimeout(timeout);
    req.off('close', onClose);
  }
}

// ─────────────────────────────────────────────────────────────────────
// HTTP server
// ─────────────────────────────────────────────────────────────────────

/** Read the full request body as a string, enforcing a size cap. */
function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function writeJson(res: ServerResponse, status: number, corsOrigin: string, jsonBody: string): void {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': corsOrigin,
    ...SECURITY_HEADERS,
  });
  res.end(jsonBody);
}

/**
 * Build the request handler bound to a config. Exposed for tests that want to
 * mount the handler on their own server; most callers use `createBridgeServer`.
 */
export function createRequestHandler(
  config: BridgeConfig,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const backend: BackendConfig = { apiKey: config.apiKey, baseUrl: config.baseUrl };
  const verbose = config.verbose === true;
  const upstreamTimeoutMs = config.upstreamTimeoutMs ?? DEFAULT_UPSTREAM_TIMEOUT_MS;
  let counter = 0;

  return async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const reqNum = ++counter;
    const method = req.method ?? 'GET';
    const rawUrl = req.url ?? '/';
    const path = rawUrl.split('?')[0];
    const corsOrigin = typeof req.headers.origin === 'string' ? req.headers.origin : '*';

    try {
      // CORS preflight (Claude Code is not a browser, but keep it correct).
      if (method === 'OPTIONS') {
        res.writeHead(204, {
          'Access-Control-Allow-Origin': corsOrigin,
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'content-type, authorization, x-api-key, anthropic-version, anthropic-beta',
          'Access-Control-Max-Age': '86400',
          ...SECURITY_HEADERS,
        });
        res.end();
        return;
      }

      // Health check.
      if (method === 'GET' && (path === '/' || path === '/health')) {
        writeJson(res, 200, corsOrigin, JSON.stringify({
          ok: true,
          service: 'bring-your-own-model',
          model: config.model,
          ...(config.fastModel ? { fastModel: config.fastModel } : {}),
          baseUrl: config.baseUrl,
          api: pickApi(config.baseUrl, config.model),
        }));
        return;
      }

      const isMessages = method === 'POST' && path === '/v1/messages';
      const isCount = method === 'POST' && path === '/v1/messages/count_tokens';
      if (!isMessages && !isCount) {
        writeJson(res, 404, corsOrigin, anthropicErrorBody('not_found_error', `Unknown route: ${method} ${path}`));
        return;
      }

      // Read + parse the Anthropic Messages body.
      let raw: string;
      try {
        raw = await readRequestBody(req);
      } catch {
        writeJson(res, 413, corsOrigin, anthropicErrorBody('invalid_request_error', 'Request body too large.'));
        return;
      }
      let body: AnthropicRequest;
      try {
        body = JSON.parse(raw) as AnthropicRequest;
      } catch {
        writeJson(res, 400, corsOrigin, anthropicErrorBody('invalid_request_error', 'Request body is not valid JSON.'));
        return;
      }

      // count_tokens: local estimate, no upstream call (no tokenizer dependency).
      if (isCount) {
        writeJson(res, 200, corsOrigin, JSON.stringify({ input_tokens: estimateTokenCount(body) }));
        return;
      }

      // Messages: resolve the target and translate-and-forward.
      const target = resolveOpenAITarget({
        path,
        model: body.model,
        forcedModel: config.model,
        baseUrl: config.baseUrl,
        fastModel: config.fastModel,
      });
      if (!target) {
        writeJson(res, 400, corsOrigin, anthropicErrorBody('invalid_request_error', 'No target model is configured.'));
        return;
      }
      await handleTranslatedOpenAI(req, res, body, target, backend, {
        corsOrigin,
        verbose,
        upstreamTimeoutMs,
        reqNum,
      });
    } catch (err) {
      // Last-resort guard: never let a handler rejection crash the server.
      const detail = err instanceof Error ? err.message : String(err);
      if (verbose) console.error(`${LOG_PREFIX} #${reqNum} handler error: ${detail}`);
      if (!res.headersSent) {
        writeJson(res, 500, corsOrigin, anthropicErrorBody('api_error', 'Internal bridge error.'));
      } else if (!res.writableEnded) {
        try { res.end(); } catch { /* already closed */ }
      }
    }
  };
}

/** Create the bridge HTTP server (not yet listening). */
export function createBridgeServer(config: BridgeConfig): Server {
  return createServer(createRequestHandler(config));
}
