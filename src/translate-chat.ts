/**
 * Anthropic Messages ⇄ OpenAI Chat Completions translation.
 *
 * Pure data transforms — no network, no fs, no timers. The wiring layer
 * (src/proxy.ts) lets an Anthropic-speaking client (Claude Code on
 * `/v1/messages`) drive an OpenAI-compat model: it translates the
 * inbound Messages request with `anthropicToOpenAIRequest`, POSTs it to
 * `{baseUrl}/chat/completions`, and translates the reply back with
 * `openAIToAnthropicResponse` (non-streaming) or
 * `openAIStreamToAnthropicSSE` (streaming).
 *
 * Targets the Chat Completions shape, not the Responses API — chat
 * completions is the shape every OpenAI-compat provider (OpenAI, Groq,
 * OpenRouter, LiteLLM, Ollama) actually serves.
 *
 * The Anthropic-side output mirrors the Messages SSE event shapes Claude
 * Code consumes: message_start,
 * content_block_start/delta/stop, message_delta, message_stop; text via
 * `text_delta`, tool arguments via `input_json_delta`, one content block
 * open at a time, indices strictly increasing, event name === data.type.
 *
 * Deliberate lossy edges (all documented at the relevant function):
 *   - `cache_control` is dropped silently (no chat-completions analog).
 *   - assistant `thinking` / `redacted_thinking` blocks are dropped.
 *   - Anthropic server tools (entries without `input_schema`) are skipped.
 *   - images inside `tool_result` content are dropped (OpenAI `tool`
 *     messages are string-only); their sibling text blocks survive.
 */

// ─────────────────────────────────────────────────────────────────────
// Anthropic-side types (Messages API, as Claude Code sends it)
// ─────────────────────────────────────────────────────────────────────

export interface AnthropicTextBlock {
  type: 'text';
  text: string;
  cache_control?: unknown;
}

export interface AnthropicImageBlock {
  type: 'image';
  source:
    | { type: 'base64'; media_type: string; data: string }
    | { type: 'url'; url: string };
  cache_control?: unknown;
}

export interface AnthropicToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
  cache_control?: unknown;
}

export interface AnthropicToolResultBlock {
  type: 'tool_result';
  tool_use_id: string;
  content?: string | Array<Record<string, unknown>>;
  is_error?: boolean;
  cache_control?: unknown;
}

export type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicImageBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock
  | { type: string; [key: string]: unknown };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
  /** Server-tool discriminator (`bash_20250124`, …). Untranslatable. */
  type?: string;
  cache_control?: unknown;
}

export interface AnthropicToolChoice {
  type: 'auto' | 'any' | 'tool' | 'none';
  name?: string;
  disable_parallel_tool_use?: boolean;
}

export interface AnthropicThinkingConfig {
  type: 'enabled' | 'disabled';
  budget_tokens?: number;
}

export interface AnthropicRequest {
  model: string;
  system?: string | AnthropicTextBlock[];
  messages: AnthropicMessage[];
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
  thinking?: AnthropicThinkingConfig;
  metadata?: { user_id?: string | null };
}

export type AnthropicStopReason =
  | 'end_turn'
  | 'max_tokens'
  | 'tool_use'
  | 'stop_sequence';

export interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface AnthropicResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  content: Array<AnthropicTextBlock | AnthropicToolUseBlock>;
  stop_reason: AnthropicStopReason | null;
  stop_sequence: string | null;
  usage: AnthropicUsage;
}

// ─────────────────────────────────────────────────────────────────────
// OpenAI-side types (Chat Completions)
// ─────────────────────────────────────────────────────────────────────

export interface OpenAIContentPartText {
  type: 'text';
  text: string;
}

export interface OpenAIContentPartImage {
  type: 'image_url';
  image_url: { url: string };
}

export type OpenAIContentPart = OpenAIContentPartText | OpenAIContentPartImage;

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface OpenAIChatMessage {
  role: 'system' | 'developer' | 'user' | 'assistant' | 'tool';
  content: string | OpenAIContentPart[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export interface OpenAIFunctionTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export type OpenAIToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } };

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIChatMessage[];
  tools?: OpenAIFunctionTool[];
  tool_choice?: OpenAIToolChoice;
  parallel_tool_calls?: boolean;
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  stream?: boolean;
  stream_options?: { include_usage: boolean };
  reasoning_effort?: 'low' | 'medium' | 'high';
  user?: string;
}

export interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface OpenAIChatResponse {
  id?: string;
  object?: string;
  model?: string;
  choices?: Array<{
    index?: number;
    message?: {
      role?: string;
      content?: string | null;
      refusal?: string | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason?: string | null;
  }>;
  usage?: OpenAIUsage | null;
}

/** One parsed `data:` payload from a chat-completions SSE stream. */
export interface OpenAIChatChunk {
  id?: string;
  object?: string;
  model?: string;
  choices?: Array<{
    index?: number;
    delta?: {
      role?: string;
      content?: string | null;
      refusal?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: OpenAIUsage | null;
}

// ─────────────────────────────────────────────────────────────────────
// Request translation: Anthropic → OpenAI
// ─────────────────────────────────────────────────────────────────────

export interface AnthropicToOpenAIOptions {
  /**
   * Emit `max_completion_tokens` instead of `max_tokens`. Default false:
   * `max_tokens` is the field every OpenAI-compat provider accepts.
   * api.openai.com's reasoning-era models (gpt-5.x, o-series) reject
   * `max_tokens` outright, so the wiring layer should set this when the
   * backend is OpenAI proper and the model is a reasoning model.
   */
  useMaxCompletionTokens?: boolean;
  /**
   * Role for the flattened system prompt. Default 'system' (universal);
   * 'developer' is the OpenAI-preferred spelling for reasoning models.
   */
  systemRole?: 'system' | 'developer';
  /**
   * When the request streams, ask the upstream for a final usage chunk
   * via `stream_options: {include_usage: true}`. Default true. Disable
   * for strict providers that reject `stream_options`.
   */
  includeStreamUsage?: boolean;
}

/**
 * Thinking-budget → reasoning_effort thresholds. Claude Code's thinking
 * tiers land at ~4k ("think"), ~10k ("think hard") and 31999
 * ("ultrathink") budget_tokens, so the cut points sit between those
 * tiers:
 *
 *   budget_tokens ≤ 4096            → 'low'
 *   4096 < budget_tokens ≤ 16384    → 'medium'
 *   budget_tokens > 16384           → 'high'
 *
 * `thinking` absent, disabled, or without a positive budget → no
 * `reasoning_effort` in the output (upstream default applies).
 */
export const REASONING_EFFORT_LOW_MAX = 4096;
export const REASONING_EFFORT_MEDIUM_MAX = 16384;

function thinkingToReasoningEffort(
  thinking: AnthropicThinkingConfig | undefined,
): 'low' | 'medium' | 'high' | undefined {
  if (!thinking || thinking.type !== 'enabled') return undefined;
  const budget = thinking.budget_tokens;
  if (typeof budget !== 'number' || !Number.isFinite(budget) || budget <= 0) {
    return undefined;
  }
  if (budget <= REASONING_EFFORT_LOW_MAX) return 'low';
  if (budget <= REASONING_EFFORT_MEDIUM_MAX) return 'medium';
  return 'high';
}

/** Flatten a Messages `system` field (string or text-block array). */
function flattenSystem(system: AnthropicRequest['system']): string {
  if (typeof system === 'string') return system;
  if (!Array.isArray(system)) return '';
  const parts: string[] = [];
  for (const block of system) {
    // cache_control dropped silently — no chat-completions analog.
    if (block && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  return parts.join('\n\n');
}

/**
 * Stringify a tool_result's content for an OpenAI `tool` message, which
 * only carries a string. Text blocks are joined with newlines; images
 * and other non-text blocks are dropped unless there is no text at all,
 * in which case the raw array is JSON-stringified so the data survives.
 */
function toolResultContentToString(
  content: AnthropicToolResultBlock['content'],
): string {
  if (content === undefined || content === null) return '';
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return JSON.stringify(content);
  const texts: string[] = [];
  for (const block of content) {
    if (block && block.type === 'text' && typeof block.text === 'string') {
      texts.push(block.text);
    }
  }
  if (texts.length > 0) return texts.join('\n');
  return content.length > 0 ? JSON.stringify(content) : '';
}

function imageBlockToPart(block: AnthropicImageBlock): OpenAIContentPartImage | null {
  const source = block.source;
  if (!source || typeof source !== 'object') return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * Translate one Anthropic user message (block form) into OpenAI
 * messages. tool_result blocks become individual `role:'tool'` messages
 * and are emitted FIRST — Anthropic guarantees results for the previous
 * assistant turn's tool_use lead the next user message, and OpenAI
 * requires the `tool` replies to directly follow the assistant message
 * that carried the tool_calls. Remaining text/image blocks collapse into
 * one user message (plain string when it is a single text block).
 */
function translateUserBlocks(blocks: AnthropicContentBlock[]): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  const parts: OpenAIContentPart[] = [];

  for (const block of blocks) {
    if (!block || typeof block !== 'object') continue;
    if (block.type === 'tool_result') {
      const tr = block as AnthropicToolResultBlock;
      out.push({
        role: 'tool',
        tool_call_id: typeof tr.tool_use_id === 'string' ? tr.tool_use_id : '',
        content: toolResultContentToString(tr.content),
      });
    } else if (block.type === 'text' && typeof (block as AnthropicTextBlock).text === 'string') {
      parts.push({ type: 'text', text: (block as AnthropicTextBlock).text });
    } else if (block.type === 'image') {
      const part = imageBlockToPart(block as AnthropicImageBlock);
      if (part) parts.push(part);
    }
    // Unknown block types (document, search_result, …) are dropped.
  }

  const only = parts.length === 1 ? parts[0] : undefined;
  if (only && only.type === 'text') {
    out.push({ role: 'user', content: only.text });
  } else if (parts.length > 0) {
    out.push({ role: 'user', content: parts });
  }
  return out;
}

/**
 * Translate one Anthropic assistant message (block form). Text blocks
 * join with a blank line; tool_use blocks become `tool_calls` with
 * JSON-stringified arguments. thinking / redacted_thinking blocks are
 * dropped — chat completions has no inbound slot for prior reasoning.
 */
function translateAssistantBlocks(blocks: AnthropicContentBlock[]): OpenAIChatMessage {
  const texts: string[] = [];
  const toolCalls: OpenAIToolCall[] = [];

  for (const block of blocks) {
    if (!block || typeof block !== 'object') continue;
    if (block.type === 'text' && typeof (block as AnthropicTextBlock).text === 'string') {
      texts.push((block as AnthropicTextBlock).text);
    } else if (block.type === 'tool_use') {
      const tu = block as AnthropicToolUseBlock;
      toolCalls.push({
        id: typeof tu.id === 'string' ? tu.id : '',
        type: 'function',
        function: {
          name: typeof tu.name === 'string' ? tu.name : '',
          arguments: JSON.stringify(tu.input ?? {}),
        },
      });
    }
  }

  const msg: OpenAIChatMessage = {
    role: 'assistant',
    content: texts.length > 0 ? texts.join('\n\n') : null,
  };
  if (toolCalls.length > 0) msg.tool_calls = toolCalls;
  return msg;
}

function translateToolChoice(
  choice: AnthropicToolChoice | undefined,
): OpenAIToolChoice | undefined {
  if (!choice || typeof choice !== 'object') return undefined;
  switch (choice.type) {
    case 'auto':
      return 'auto';
    case 'none':
      return 'none';
    case 'any':
      return 'required';
    case 'tool':
      return typeof choice.name === 'string'
        ? { type: 'function', function: { name: choice.name } }
        : 'required';
    default:
      return undefined;
  }
}

/**
 * Translate an Anthropic Messages request into an OpenAI Chat
 * Completions request body for `{baseUrl}/chat/completions`.
 *
 * Anthropic `input_schema` is already JSON Schema, which is what OpenAI
 * `parameters` expects — it passes through unchanged. Tool entries
 * without an `input_schema` (Anthropic server tools such as
 * `web_search_20250305`) have no chat-completions equivalent and are
 * skipped. `cache_control` is dropped wherever it appears.
 */
export function anthropicToOpenAIRequest(
  body: AnthropicRequest,
  targetModel: string,
  options: AnthropicToOpenAIOptions = {},
): OpenAIChatRequest {
  const systemRole = options.systemRole ?? 'system';
  const messages: OpenAIChatMessage[] = [];

  const systemText = flattenSystem(body.system);
  if (systemText.length > 0) {
    messages.push({ role: systemRole, content: systemText });
  }

  for (const msg of Array.isArray(body.messages) ? body.messages : []) {
    if (!msg || typeof msg !== 'object') continue;
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role === 'assistant' ? 'assistant' : 'user', content: msg.content });
      continue;
    }
    if (!Array.isArray(msg.content)) continue;
    if (msg.role === 'assistant') {
      messages.push(translateAssistantBlocks(msg.content));
    } else {
      messages.push(...translateUserBlocks(msg.content));
    }
  }

  const out: OpenAIChatRequest = { model: targetModel, messages };

  if (Array.isArray(body.tools)) {
    const tools: OpenAIFunctionTool[] = [];
    for (const tool of body.tools) {
      if (!tool || typeof tool.name !== 'string') continue;
      if (!tool.input_schema || typeof tool.input_schema !== 'object') continue;
      const fn: OpenAIFunctionTool['function'] = {
        name: tool.name,
        parameters: tool.input_schema,
      };
      if (typeof tool.description === 'string') fn.description = tool.description;
      tools.push({ type: 'function', function: fn });
    }
    if (tools.length > 0) out.tools = tools;
  }

  const toolChoice = translateToolChoice(body.tool_choice);
  if (toolChoice !== undefined && out.tools) out.tool_choice = toolChoice;
  if (body.tool_choice?.disable_parallel_tool_use === true && out.tools) {
    out.parallel_tool_calls = false;
  }

  if (typeof body.max_tokens === 'number' && body.max_tokens > 0) {
    // Clamp to a safe chat-completions output ceiling. Clients (Claude Code)
    // size max_tokens for their own big-context model (up to 64000); most chat
    // models cap far lower (gpt-4o: 16384) and return 400 on an over-large value.
    const CHAT_MAX_OUTPUT = 16384;
    const capped = Math.min(body.max_tokens, CHAT_MAX_OUTPUT);
    if (options.useMaxCompletionTokens) out.max_completion_tokens = capped;
    else out.max_tokens = capped;
  }
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    // OpenAI caps `stop` at 4 entries; keep the first 4.
    out.stop = body.stop_sequences.slice(0, 4);
  }
  if (body.stream === true) {
    out.stream = true;
    if (options.includeStreamUsage !== false) {
      out.stream_options = { include_usage: true };
    }
  }

  const effort = thinkingToReasoningEffort(body.thinking);
  if (effort) out.reasoning_effort = effort;

  const userId = body.metadata?.user_id;
  if (typeof userId === 'string' && userId.length > 0) out.user = userId;

  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Response translation: OpenAI → Anthropic (non-streaming)
// ─────────────────────────────────────────────────────────────────────

function mapFinishReason(reason: string | null | undefined): AnthropicStopReason {
  switch (reason) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'stop':
    case 'content_filter':
    default:
      // 'stop' is the honest end_turn; content_filter has no Anthropic
      // analog, and an unknown/missing reason degrades to end_turn too.
      return 'end_turn';
  }
}

/**
 * Parse tool-call arguments defensively. Upstreams occasionally emit
 * truncated or malformed JSON (notably on `length` stops); rather than
 * throw mid-response, unparseable arguments degrade to `{}` — the
 * client's own tool validation reports the missing fields.
 */
function safeParseArguments(raw: string | undefined): Record<string, unknown> {
  if (typeof raw !== 'string' || raw.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

function anthropicMessageId(openaiId: string | undefined): string {
  if (typeof openaiId === 'string' && openaiId.length > 0) {
    return openaiId.startsWith('msg_') ? openaiId : `msg_${openaiId}`;
  }
  return 'msg_openai_translate';
}

/**
 * Translate a non-streaming Chat Completions response into an Anthropic
 * Messages response. `requestModel` is echoed back as `model` so the
 * client sees the model name it asked for, not the upstream's alias.
 * Only `choices[0]` is translated (the bridge never requests n > 1). A
 * `refusal` string surfaces as a text block so the client sees why the
 * turn produced no content.
 */
export function openAIToAnthropicResponse(
  resp: OpenAIChatResponse,
  requestModel: string,
): AnthropicResponse {
  const choice = resp.choices?.[0];
  const message = choice?.message;
  const content: Array<AnthropicTextBlock | AnthropicToolUseBlock> = [];

  if (typeof message?.content === 'string' && message.content.length > 0) {
    content.push({ type: 'text', text: message.content });
  }
  if (typeof message?.refusal === 'string' && message.refusal.length > 0) {
    content.push({ type: 'text', text: message.refusal });
  }
  for (const call of message?.tool_calls ?? []) {
    if (!call || typeof call !== 'object') continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id.length > 0 ? call.id : 'toolu_openai_translate',
      name: call.function?.name ?? '',
      input: safeParseArguments(call.function?.arguments),
    });
  }

  return {
    id: anthropicMessageId(resp.id),
    type: 'message',
    role: 'assistant',
    model: requestModel,
    content,
    stop_reason: mapFinishReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage?.prompt_tokens ?? 0,
      output_tokens: resp.usage?.completion_tokens ?? 0,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────
// Streaming translation: OpenAI chunk objects → Anthropic SSE events
// ─────────────────────────────────────────────────────────────────────

export type AnthropicStreamEvent =
  | { type: 'message_start'; message: AnthropicResponse }
  | {
      type: 'content_block_start';
      index: number;
      content_block:
        | { type: 'text'; text: string }
        | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> };
    }
  | {
      type: 'content_block_delta';
      index: number;
      delta:
        | { type: 'text_delta'; text: string }
        | { type: 'input_json_delta'; partial_json: string };
    }
  | { type: 'content_block_stop'; index: number }
  | {
      type: 'message_delta';
      delta: { stop_reason: AnthropicStopReason; stop_sequence: null };
      usage: { output_tokens: number; input_tokens?: number };
    }
  | { type: 'message_stop' };

export interface OpenAIStreamTranslatorOptions {
  /** Model name echoed in message_start. Defaults to the first chunk's `model`. */
  requestModel?: string;
}

export interface OpenAIStreamTranslator {
  /** Feed one parsed chat-completions chunk; returns the Anthropic events it produced. */
  push(chunk: OpenAIChatChunk): AnthropicStreamEvent[];
  /** Signal upstream end-of-stream; returns the closing events. Idempotent. */
  end(): AnthropicStreamEvent[];
}

/**
 * Build a streaming translator from parsed OpenAI chat-completion
 * chunks to Anthropic SSE event objects.
 *
 * Pure state machine: callers parse the upstream SSE themselves (or use
 * `parseOpenAISSELine`), push each chunk object, and serialize the
 * returned events (`formatAnthropicSSE` produces the wire framing).
 *
 * Event discipline mirrors what Claude Code's Messages streaming
 * consumer expects on the other side:
 *
 *   - message_start fires once, on the first chunk (role-only chunks
 *     count), with an empty-content message envelope.
 *   - Exactly one content block is open at any time; indices are
 *     strictly increasing. Text opens a `text` block; each distinct
 *     OpenAI tool_call index opens a `tool_use` block (id + name arrive
 *     on that first fragment) and subsequent argument fragments become
 *     `input_json_delta` events. Opening a new block closes the
 *     previous one with content_block_stop.
 *   - A `finish_reason` closes the open block. The final
 *     message_delta (stop_reason + usage) and message_stop are emitted
 *     by `end()`, because OpenAI's usage chunk (requested via
 *     `stream_options.include_usage`) arrives AFTER finish_reason —
 *     emitting message_delta earlier would lose the token counts.
 *   - `usage.input_tokens` rides in message_delta usage when the
 *     upstream reported it; message_start necessarily carries 0/0
 *     since chat completions reports usage only at stream end.
 *
 * Degenerate inputs: `refusal` deltas are treated as text; a fragment
 * for an already-closed tool index (no real upstream interleaves
 * these) still emits against that block's original index rather than
 * dropping data; an empty stream still yields a well-formed
 * message_start → message_delta → message_stop sequence from end().
 * Only choices[0] is translated.
 */
export function openAIStreamToAnthropicSSE(
  options: OpenAIStreamTranslatorOptions = {},
): OpenAIStreamTranslator {
  let started = false;
  let ended = false;
  let model = options.requestModel;
  let messageId = 'msg_openai_translate';
  let nextIndex = 0;
  let open:
    | { kind: 'text'; index: number }
    | { kind: 'tool'; index: number; openaiIndex: number }
    | null = null;
  /** OpenAI tool_call index → Anthropic content block index. */
  const toolBlockIndex = new Map<number, number>();
  let finishReason: string | null = null;
  let usage: OpenAIUsage | null = null;
  let syntheticToolSeq = 0;

  function ensureStarted(chunk: OpenAIChatChunk | null, events: AnthropicStreamEvent[]): void {
    if (started) return;
    started = true;
    messageId = anthropicMessageId(chunk?.id);
    const resolvedModel =
      model ?? (typeof chunk?.model === 'string' && chunk.model.length > 0 ? chunk.model : 'unknown');
    model = resolvedModel;
    events.push({
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        model: resolvedModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }

  function closeOpenBlock(events: AnthropicStreamEvent[]): void {
    if (open) {
      events.push({ type: 'content_block_stop', index: open.index });
      open = null;
    }
  }

  function pushText(text: string, events: AnthropicStreamEvent[]): void {
    if (!open || open.kind !== 'text') {
      closeOpenBlock(events);
      open = { kind: 'text', index: nextIndex++ };
      events.push({
        type: 'content_block_start',
        index: open.index,
        content_block: { type: 'text', text: '' },
      });
    }
    events.push({
      type: 'content_block_delta',
      index: open.index,
      delta: { type: 'text_delta', text },
    });
  }

  return {
    push(chunk: OpenAIChatChunk): AnthropicStreamEvent[] {
      const events: AnthropicStreamEvent[] = [];
      if (ended || !chunk || typeof chunk !== 'object') return events;

      if (chunk.usage && typeof chunk.usage === 'object') usage = chunk.usage;

      const choice = chunk.choices?.[0];
      if (!choice || typeof choice !== 'object') {
        // Usage-only chunk (stream_options.include_usage) — recorded above.
        return events;
      }

      ensureStarted(chunk, events);
      const delta = choice.delta;

      const text =
        typeof delta?.content === 'string' && delta.content.length > 0
          ? delta.content
          : typeof delta?.refusal === 'string' && delta.refusal.length > 0
            ? delta.refusal
            : null;
      if (text !== null) pushText(text, events);

      for (const call of delta?.tool_calls ?? []) {
        if (!call || typeof call !== 'object') continue;
        const openaiIndex = typeof call.index === 'number' ? call.index : 0;
        let blockIndex = toolBlockIndex.get(openaiIndex);
        if (blockIndex === undefined) {
          closeOpenBlock(events);
          blockIndex = nextIndex++;
          toolBlockIndex.set(openaiIndex, blockIndex);
          open = { kind: 'tool', index: blockIndex, openaiIndex };
          const id =
            typeof call.id === 'string' && call.id.length > 0
              ? call.id
              : `toolu_openai_${syntheticToolSeq++}`;
          events.push({
            type: 'content_block_start',
            index: blockIndex,
            content_block: {
              type: 'tool_use',
              id,
              name: call.function?.name ?? '',
              input: {},
            },
          });
        }
        const args = call.function?.arguments;
        if (typeof args === 'string' && args.length > 0) {
          events.push({
            type: 'content_block_delta',
            index: blockIndex,
            delta: { type: 'input_json_delta', partial_json: args },
          });
        }
      }

      if (typeof choice.finish_reason === 'string' && choice.finish_reason.length > 0) {
        finishReason = choice.finish_reason;
        closeOpenBlock(events);
      }

      return events;
    },

    end(): AnthropicStreamEvent[] {
      if (ended) return [];
      ended = true;
      const events: AnthropicStreamEvent[] = [];
      ensureStarted(null, events);
      closeOpenBlock(events);

      const usageOut: { output_tokens: number; input_tokens?: number } = {
        output_tokens: usage?.completion_tokens ?? 0,
      };
      if (typeof usage?.prompt_tokens === 'number') {
        usageOut.input_tokens = usage.prompt_tokens;
      }
      events.push({
        type: 'message_delta',
        delta: { stop_reason: mapFinishReason(finishReason), stop_sequence: null },
        usage: usageOut,
      });
      events.push({ type: 'message_stop' });
      return events;
    },
  };
}

/**
 * Convenience: parse one line of a chat-completions SSE stream.
 * Returns the parsed chunk for `data: {...}` lines; null for event
 * headers, blank keep-alives, `data: [DONE]`, and unparseable payloads.
 * The wiring layer owns real SSE framing (multi-line buffering, CRLF) —
 * this is a per-line helper for the common well-formed case.
 */
export function parseOpenAISSELine(line: string): OpenAIChatChunk | null {
  if (typeof line !== 'string') return null;
  const trimmed = line.endsWith('\r') ? line.slice(0, -1) : line;
  if (!trimmed.startsWith('data:')) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return null;
  try {
    return JSON.parse(payload) as OpenAIChatChunk;
  } catch {
    return null;
  }
}

/**
 * Serialize one Anthropic stream event into SSE wire framing:
 * `event: <type>\ndata: <json>\n\n` — the same `event:`/`data:` pairing
 * Claude Code's Messages stream expects (an `event:` line, a `data:`
 * JSON line, then a blank-line separator).
 */
export function formatAnthropicSSE(event: AnthropicStreamEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}

// ─────────────────────────────────────────────────────────────────────
// Token estimation
// ─────────────────────────────────────────────────────────────────────

/** chars-per-token divisor for the estimate. */
const CHARS_PER_TOKEN = 4;
/** Flat per-image charge — vision inputs bill ~1.6k tokens at CC's usual sizes. */
const IMAGE_TOKEN_ESTIMATE = 1600;
/** Per-message structural overhead (role tags, separators). */
const PER_MESSAGE_OVERHEAD = 4;

/**
 * Rough token estimate for count_tokens emulation: serialized text
 * chars / 4, plus a flat per-image charge and a small per-message
 * overhead. Base64 image payloads are deliberately NOT counted as text
 * (that would overweight them ~15x). This is an ESTIMATE for budgeting
 * and display — calibration against real tokenizers comes later.
 */
export function estimateTokenCount(body: AnthropicRequest): number {
  let chars = 0;
  let images = 0;
  let messageCount = 0;

  const countBlock = (block: AnthropicContentBlock): void => {
    if (!block || typeof block !== 'object') return;
    switch (block.type) {
      case 'text':
        if (typeof (block as AnthropicTextBlock).text === 'string') {
          chars += (block as AnthropicTextBlock).text.length;
        }
        break;
      case 'image':
        images++;
        break;
      case 'tool_use': {
        const tu = block as AnthropicToolUseBlock;
        chars += (tu.name ?? '').length;
        try {
          chars += JSON.stringify(tu.input ?? {}).length;
        } catch {
          /* circular input — skip */
        }
        break;
      }
      case 'tool_result': {
        const tr = block as AnthropicToolResultBlock;
        const flattened = toolResultContentToString(tr.content);
        chars += flattened.length;
        if (Array.isArray(tr.content)) {
          for (const inner of tr.content) {
            if (inner && typeof inner === 'object' && inner.type === 'image') images++;
          }
        }
        break;
      }
      default:
        try {
          chars += JSON.stringify(block).length;
        } catch {
          /* skip */
        }
    }
  };

  chars += flattenSystem(body.system).length;

  for (const msg of Array.isArray(body.messages) ? body.messages : []) {
    if (!msg || typeof msg !== 'object') continue;
    messageCount++;
    if (typeof msg.content === 'string') {
      chars += msg.content.length;
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) countBlock(block);
    }
  }

  for (const tool of Array.isArray(body.tools) ? body.tools : []) {
    if (!tool || typeof tool !== 'object') continue;
    try {
      chars += JSON.stringify({
        name: tool.name,
        description: tool.description,
        input_schema: tool.input_schema,
      }).length;
    } catch {
      /* skip */
    }
  }

  const estimate =
    Math.ceil(chars / CHARS_PER_TOKEN) +
    images * IMAGE_TOKEN_ESTIMATE +
    messageCount * PER_MESSAGE_OVERHEAD;
  return Math.max(1, estimate);
}
