#!/usr/bin/env node
// In-process tests for src/openai-translate.ts — the pure Anthropic
// Messages ⇄ OpenAI Chat Completions translation layer. No network, no
// fs, no credentials: hand-written fixtures in, shape assertions out.
//
// Run: node --test test/openai-translate.test.mjs
// (also runs standalone under plain `node`, so the all.test.mjs driver
// picks it up like every other file in test/)

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  anthropicToOpenAIRequest,
  openAIToAnthropicResponse,
  openAIStreamToAnthropicSSE,
  parseOpenAISSELine,
  formatAnthropicSSE,
  estimateTokenCount,
  REASONING_EFFORT_LOW_MAX,
  REASONING_EFFORT_MEDIUM_MAX,
} from '../dist/translate-chat.js';

// ─────────────────────────────────────────────────────────────────────
// anthropicToOpenAIRequest
// ─────────────────────────────────────────────────────────────────────

test('system array flattens to one system message, cache_control dropped', () => {
  const out = anthropicToOpenAIRequest(
    {
      model: 'claude-opus-4-5',
      max_tokens: 512,
      system: [
        { type: 'text', text: 'You are Claude Code.', cache_control: { type: 'ephemeral' } },
        { type: 'text', text: 'Be terse.' },
      ],
      messages: [{ role: 'user', content: 'hi' }],
    },
    'gpt-5.6-sol',
  );

  assert.equal(out.model, 'gpt-5.6-sol');
  assert.deepEqual(out.messages[0], {
    role: 'system',
    content: 'You are Claude Code.\n\nBe terse.',
  });
  assert.deepEqual(out.messages[1], { role: 'user', content: 'hi' });
  assert.ok(!JSON.stringify(out).includes('cache_control'), 'cache_control must not survive');
  assert.equal(out.max_tokens, 512);
  assert.equal(out.max_completion_tokens, undefined);
});

test('string system + developer role option', () => {
  const out = anthropicToOpenAIRequest(
    { model: 'm', max_tokens: 1, system: 'sys prompt', messages: [{ role: 'user', content: 'x' }] },
    'gpt-5.6-sol',
    { systemRole: 'developer' },
  );
  assert.deepEqual(out.messages[0], { role: 'developer', content: 'sys prompt' });
});

test('multi-turn tool_use → tool_result round trip', () => {
  const out = anthropicToOpenAIRequest(
    {
      model: 'claude-opus-4-5',
      max_tokens: 1024,
      messages: [
        { role: 'user', content: 'Weather in Paris?' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Let me check.' },
            { type: 'tool_use', id: 'toolu_01', name: 'get_weather', input: { city: 'Paris' } },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_01',
              content: [{ type: 'text', text: '18C, sunny' }],
            },
            { type: 'text', text: 'thanks — and Berlin?' },
          ],
        },
      ],
      tools: [
        {
          name: 'get_weather',
          description: 'Get current weather',
          input_schema: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
          },
        },
      ],
    },
    'gpt-5.6-sol',
  );

  assert.equal(out.messages.length, 4);
  assert.deepEqual(out.messages[0], { role: 'user', content: 'Weather in Paris?' });

  const asst = out.messages[1];
  assert.equal(asst.role, 'assistant');
  assert.equal(asst.content, 'Let me check.');
  assert.equal(asst.tool_calls.length, 1);
  assert.deepEqual(asst.tool_calls[0], {
    id: 'toolu_01',
    type: 'function',
    function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
  });

  // tool reply must directly follow the assistant tool_calls message,
  // BEFORE the remaining user text from the same Anthropic message.
  assert.deepEqual(out.messages[2], {
    role: 'tool',
    tool_call_id: 'toolu_01',
    content: '18C, sunny',
  });
  assert.deepEqual(out.messages[3], { role: 'user', content: 'thanks — and Berlin?' });

  assert.equal(out.tools.length, 1);
  assert.deepEqual(out.tools[0], {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get current weather',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
    },
  });
});

test('assistant tool_use with no text gets content: null', () => {
  const out = anthropicToOpenAIRequest(
    {
      model: 'm',
      max_tokens: 1,
      messages: [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [{ type: 'tool_use', id: 't1', name: 'f', input: {} }],
        },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1' }] },
      ],
    },
    'gpt-5.6-sol',
  );
  assert.equal(out.messages[1].content, null);
  assert.equal(out.messages[1].tool_calls.length, 1);
  // empty tool_result content becomes an empty string, not undefined
  assert.deepEqual(out.messages[2], { role: 'tool', tool_call_id: 't1', content: '' });
});

test('image blocks become image_url parts (base64 → data URL, url passthrough)', () => {
  const out = anthropicToOpenAIRequest(
    {
      model: 'm',
      max_tokens: 1,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' },
            },
            { type: 'image', source: { type: 'url', url: 'https://example.com/x.jpg' } },
          ],
        },
      ],
    },
    'gpt-5.6-sol',
  );

  const content = out.messages[0].content;
  assert.ok(Array.isArray(content));
  assert.deepEqual(content[0], { type: 'text', text: 'What is this?' });
  assert.deepEqual(content[1], {
    type: 'image_url',
    image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
  });
  assert.deepEqual(content[2], {
    type: 'image_url',
    image_url: { url: 'https://example.com/x.jpg' },
  });
});

test('tool_choice variants map to OpenAI spellings', () => {
  const base = {
    model: 'm',
    max_tokens: 1,
    messages: [{ role: 'user', content: 'x' }],
    tools: [{ name: 'f', input_schema: { type: 'object' } }],
  };
  const tc = (tool_choice) =>
    anthropicToOpenAIRequest({ ...base, tool_choice }, 'gpt-5.6-sol').tool_choice;

  assert.equal(tc({ type: 'auto' }), 'auto');
  assert.equal(tc({ type: 'any' }), 'required');
  assert.equal(tc({ type: 'none' }), 'none');
  assert.deepEqual(tc({ type: 'tool', name: 'f' }), { type: 'function', function: { name: 'f' } });
  assert.equal(tc(undefined), undefined);

  const par = anthropicToOpenAIRequest(
    { ...base, tool_choice: { type: 'auto', disable_parallel_tool_use: true } },
    'gpt-5.6-sol',
  );
  assert.equal(par.parallel_tool_calls, false);
});

test('server tools without input_schema are skipped', () => {
  const out = anthropicToOpenAIRequest(
    {
      model: 'm',
      max_tokens: 1,
      messages: [{ role: 'user', content: 'x' }],
      tools: [
        { type: 'web_search_20250305', name: 'web_search' },
        { name: 'real_tool', input_schema: { type: 'object' } },
      ],
    },
    'gpt-5.6-sol',
  );
  assert.equal(out.tools.length, 1);
  assert.equal(out.tools[0].function.name, 'real_tool');
});

test('thinking budget maps to reasoning_effort at documented thresholds (opt-in)', () => {
  const eff = (thinking) =>
    anthropicToOpenAIRequest(
      { model: 'm', max_tokens: 1, messages: [{ role: 'user', content: 'x' }], thinking },
      'gpt-5.6-sol',
      { emitReasoningEffort: true },
    ).reasoning_effort;

  assert.equal(eff(undefined), undefined);
  assert.equal(eff({ type: 'disabled' }), undefined);
  assert.equal(eff({ type: 'enabled', budget_tokens: 1024 }), 'low');
  assert.equal(eff({ type: 'enabled', budget_tokens: REASONING_EFFORT_LOW_MAX }), 'low');
  assert.equal(eff({ type: 'enabled', budget_tokens: REASONING_EFFORT_LOW_MAX + 1 }), 'medium');
  assert.equal(eff({ type: 'enabled', budget_tokens: REASONING_EFFORT_MEDIUM_MAX }), 'medium');
  assert.equal(eff({ type: 'enabled', budget_tokens: REASONING_EFFORT_MEDIUM_MAX + 1 }), 'high');
  assert.equal(eff({ type: 'enabled', budget_tokens: 31999 }), 'high');
});

test('reasoning_effort is NOT emitted unless opted in (gpt-4o 400 regression)', () => {
  // A non-reasoning chat model (gpt-4o, …) rejects reasoning_effort with a
  // 400, so the default translation must omit it even when the client sends
  // a thinking budget — the wiring opts in only for reasoning-era models.
  const body = {
    model: 'm',
    max_tokens: 1,
    messages: [{ role: 'user', content: 'x' }],
    thinking: { type: 'enabled', budget_tokens: 31999 },
  };
  assert.equal(anthropicToOpenAIRequest(body, 'gpt-4o').reasoning_effort, undefined);
  assert.equal(
    anthropicToOpenAIRequest(body, 'gpt-4o', { emitReasoningEffort: false }).reasoning_effort,
    undefined,
  );
});

test('max_tokens is clamped to the chat-completions ceiling (gpt-4o 400 regression)', () => {
  // Claude Code sizes max_tokens for its own big-context model (64000); most
  // chat models cap far lower (gpt-4o: 16384) and 400 on an over-large value.
  const body = (maxTokens) => ({
    model: 'm',
    max_tokens: maxTokens,
    messages: [{ role: 'user', content: 'x' }],
  });
  assert.equal(anthropicToOpenAIRequest(body(64000), 'gpt-4o').max_tokens, 16384);
  assert.equal(anthropicToOpenAIRequest(body(16384), 'gpt-4o').max_tokens, 16384);
  assert.equal(anthropicToOpenAIRequest(body(512), 'gpt-4o').max_tokens, 512);
  // The clamp applies to the max_completion_tokens spelling too.
  assert.equal(
    anthropicToOpenAIRequest(body(64000), 'o3', { useMaxCompletionTokens: true }).max_completion_tokens,
    16384,
  );
});

test('max_completion_tokens flag, sampling params, stream_options', () => {
  const body = {
    model: 'm',
    max_tokens: 2048,
    temperature: 0.7,
    top_p: 0.9,
    stop_sequences: ['\n\nHuman:'],
    stream: true,
    metadata: { user_id: 'u-123' },
    messages: [{ role: 'user', content: 'x' }],
  };

  const dflt = anthropicToOpenAIRequest(body, 'gpt-5.6-sol');
  assert.equal(dflt.max_tokens, 2048);
  assert.equal(dflt.max_completion_tokens, undefined);
  assert.equal(dflt.temperature, 0.7);
  assert.equal(dflt.top_p, 0.9);
  assert.deepEqual(dflt.stop, ['\n\nHuman:']);
  assert.equal(dflt.stream, true);
  assert.deepEqual(dflt.stream_options, { include_usage: true });
  assert.equal(dflt.user, 'u-123');

  const mct = anthropicToOpenAIRequest(body, 'gpt-5.6-sol', { useMaxCompletionTokens: true });
  assert.equal(mct.max_completion_tokens, 2048);
  assert.equal(mct.max_tokens, undefined);

  const noUsage = anthropicToOpenAIRequest(body, 'gpt-5.6-sol', { includeStreamUsage: false });
  assert.equal(noUsage.stream_options, undefined);
});

// ─────────────────────────────────────────────────────────────────────
// openAIToAnthropicResponse
// ─────────────────────────────────────────────────────────────────────

test('non-streaming response with tool_calls, including bad-JSON guard', () => {
  const out = openAIToAnthropicResponse(
    {
      id: 'chatcmpl-abc123',
      object: 'chat.completion',
      model: 'gpt-5.6-sol',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'Sure.',
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
              },
              {
                id: 'call_2',
                type: 'function',
                function: { name: 'get_weather', arguments: '{"city": TRUNC' },
              },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
      usage: { prompt_tokens: 42, completion_tokens: 17, total_tokens: 59 },
    },
    'claude-opus-4-5',
  );

  assert.equal(out.id, 'msg_chatcmpl-abc123');
  assert.equal(out.type, 'message');
  assert.equal(out.role, 'assistant');
  assert.equal(out.model, 'claude-opus-4-5');
  assert.equal(out.stop_reason, 'tool_use');
  assert.equal(out.stop_sequence, null);
  assert.deepEqual(out.usage, { input_tokens: 42, output_tokens: 17 });

  assert.equal(out.content.length, 3);
  assert.deepEqual(out.content[0], { type: 'text', text: 'Sure.' });
  assert.deepEqual(out.content[1], {
    type: 'tool_use',
    id: 'call_1',
    name: 'get_weather',
    input: { city: 'Paris' },
  });
  // unparseable arguments degrade to {} rather than throwing
  assert.deepEqual(out.content[2], {
    type: 'tool_use',
    id: 'call_2',
    name: 'get_weather',
    input: {},
  });
});

test('finish_reason → stop_reason mapping', () => {
  const stop = (finish_reason) =>
    openAIToAnthropicResponse(
      { choices: [{ message: { content: 'x' }, finish_reason }] },
      'm',
    ).stop_reason;

  assert.equal(stop('stop'), 'end_turn');
  assert.equal(stop('length'), 'max_tokens');
  assert.equal(stop('tool_calls'), 'tool_use');
  assert.equal(stop('content_filter'), 'end_turn');
  assert.equal(stop(null), 'end_turn');
});

// ─────────────────────────────────────────────────────────────────────
// openAIStreamToAnthropicSSE
// ─────────────────────────────────────────────────────────────────────

// A captured-shape chat.completions stream: role preamble, two text
// deltas, one tool call whose arguments arrive split across chunks, a
// finish_reason chunk, then the usage-only chunk that
// stream_options.include_usage appends.
const STREAM_FIXTURE = [
  { id: 'chatcmpl-s1', object: 'chat.completion.chunk', model: 'gpt-5.6-sol',
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: { content: 'Check' }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: { content: 'ing.' }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: { tool_calls: [
    { index: 0, id: 'call_9', type: 'function', function: { name: 'get_weather', arguments: '' } },
  ] }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: { tool_calls: [
    { index: 0, function: { arguments: '{"city":' } },
  ] }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: { tool_calls: [
    { index: 0, function: { arguments: '"Paris"}' } },
  ] }, finish_reason: null }] },
  { id: 'chatcmpl-s1', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
  { id: 'chatcmpl-s1', choices: [], usage: { prompt_tokens: 50, completion_tokens: 12 } },
];

test('streaming: text + split tool call → well-formed, ordered Anthropic events', () => {
  const translator = openAIStreamToAnthropicSSE({ requestModel: 'claude-opus-4-5' });
  const events = [];
  for (const chunk of STREAM_FIXTURE) events.push(...translator.push(chunk));
  events.push(...translator.end());

  assert.deepEqual(
    events.map((e) => e.type),
    [
      'message_start',
      'content_block_start',   // text block, index 0
      'content_block_delta',   // "Check"
      'content_block_delta',   // "ing."
      'content_block_stop',    // text closes when the tool block opens
      'content_block_start',   // tool_use block, index 1
      'content_block_delta',   // {"city":
      'content_block_delta',   // "Paris"}
      'content_block_stop',    // closed by finish_reason
      'message_delta',
      'message_stop',
    ],
  );

  const start = events[0];
  assert.equal(start.message.id, 'msg_chatcmpl-s1');
  assert.equal(start.message.model, 'claude-opus-4-5');
  assert.equal(start.message.role, 'assistant');
  assert.deepEqual(start.message.content, []);
  assert.equal(start.message.stop_reason, null);
  assert.deepEqual(start.message.usage, { input_tokens: 0, output_tokens: 0 });

  assert.deepEqual(events[1], {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  assert.deepEqual(events[2], {
    type: 'content_block_delta',
    index: 0,
    delta: { type: 'text_delta', text: 'Check' },
  });
  const textOut = events
    .filter((e) => e.type === 'content_block_delta' && e.delta.type === 'text_delta')
    .map((e) => e.delta.text)
    .join('');
  assert.equal(textOut, 'Checking.');
  assert.deepEqual(events[4], { type: 'content_block_stop', index: 0 });

  assert.deepEqual(events[5], {
    type: 'content_block_start',
    index: 1,
    content_block: { type: 'tool_use', id: 'call_9', name: 'get_weather', input: {} },
  });
  const argJson = events
    .filter((e) => e.type === 'content_block_delta' && e.delta.type === 'input_json_delta')
    .map((e) => e.delta.partial_json)
    .join('');
  assert.deepEqual(JSON.parse(argJson), { city: 'Paris' });
  for (const e of events.filter((x) => x.type === 'content_block_delta' && x.delta.type === 'input_json_delta')) {
    assert.equal(e.index, 1, 'tool deltas carry the tool block index');
  }
  assert.deepEqual(events[8], { type: 'content_block_stop', index: 1 });

  assert.deepEqual(events[9], {
    type: 'message_delta',
    delta: { stop_reason: 'tool_use', stop_sequence: null },
    usage: { output_tokens: 12, input_tokens: 50 },
  });
  assert.deepEqual(events[10], { type: 'message_stop' });

  // structural invariants: every start has a stop, one block open at a time
  let openIndex = null;
  let maxIndex = -1;
  for (const e of events) {
    if (e.type === 'content_block_start') {
      assert.equal(openIndex, null, 'no nested content blocks');
      assert.ok(e.index > maxIndex, 'indices strictly increase');
      openIndex = e.index;
      maxIndex = e.index;
    } else if (e.type === 'content_block_delta') {
      assert.equal(e.index, openIndex, 'deltas target the open block');
    } else if (e.type === 'content_block_stop') {
      assert.equal(e.index, openIndex, 'stop matches the open block');
      openIndex = null;
    }
  }
  assert.equal(openIndex, null, 'all blocks closed at end');

  // end() is idempotent
  assert.deepEqual(translator.end(), []);
});

test('streaming: parallel tool calls get distinct, sequential blocks', () => {
  const translator = openAIStreamToAnthropicSSE();
  const events = [];
  const chunks = [
    { id: 'chatcmpl-p', model: 'gpt-5.6-sol', choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] },
    { choices: [{ index: 0, delta: { tool_calls: [
      { index: 0, id: 'call_a', type: 'function', function: { name: 'f_a', arguments: '{"a":1}' } },
    ] }, finish_reason: null }] },
    { choices: [{ index: 0, delta: { tool_calls: [
      { index: 1, id: 'call_b', type: 'function', function: { name: 'f_b', arguments: '{"b":2}' } },
    ] }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
  ];
  for (const c of chunks) events.push(...translator.push(c));
  events.push(...translator.end());

  assert.deepEqual(
    events.map((e) => e.type),
    [
      'message_start',
      'content_block_start', 'content_block_delta', 'content_block_stop',
      'content_block_start', 'content_block_delta', 'content_block_stop',
      'message_delta', 'message_stop',
    ],
  );
  assert.equal(events[1].content_block.id, 'call_a');
  assert.equal(events[1].index, 0);
  assert.equal(events[4].content_block.id, 'call_b');
  assert.equal(events[4].index, 1);
  // no usage chunk arrived → output_tokens 0, input_tokens omitted
  assert.deepEqual(events[7].usage, { output_tokens: 0 });
  assert.equal(events[7].delta.stop_reason, 'tool_use');
});

test('streaming: empty stream still yields a well-formed envelope from end()', () => {
  const translator = openAIStreamToAnthropicSSE({ requestModel: 'claude-opus-4-5' });
  const events = translator.end();
  assert.deepEqual(
    events.map((e) => e.type),
    ['message_start', 'message_delta', 'message_stop'],
  );
  assert.equal(events[1].delta.stop_reason, 'end_turn');
});

// ─────────────────────────────────────────────────────────────────────
// SSE helpers
// ─────────────────────────────────────────────────────────────────────

test('parseOpenAISSELine handles data, [DONE], headers, keep-alives, CRLF', () => {
  const chunk = parseOpenAISSELine('data: {"id":"chatcmpl-1","choices":[]}');
  assert.deepEqual(chunk, { id: 'chatcmpl-1', choices: [] });

  const crlf = parseOpenAISSELine('data: {"id":"chatcmpl-2","choices":[]}\r');
  assert.equal(crlf.id, 'chatcmpl-2');

  assert.equal(parseOpenAISSELine('data: [DONE]'), null);
  assert.equal(parseOpenAISSELine('event: something'), null);
  assert.equal(parseOpenAISSELine(': keep-alive'), null);
  assert.equal(parseOpenAISSELine(''), null);
  assert.equal(parseOpenAISSELine('data: {not json'), null);
});

test('formatAnthropicSSE frames event name + data + blank line like Claude Code expects', () => {
  const wire = formatAnthropicSSE({ type: 'message_stop' });
  assert.equal(wire, 'event: message_stop\ndata: {"type":"message_stop"}\n\n');

  const delta = formatAnthropicSSE({
    type: 'content_block_delta',
    index: 0,
    delta: { type: 'text_delta', text: 'hi' },
  });
  assert.ok(delta.startsWith('event: content_block_delta\ndata: '));
  assert.ok(delta.endsWith('\n\n'));
  const parsed = JSON.parse(delta.slice(delta.indexOf('data: ') + 6));
  assert.equal(parsed.delta.text, 'hi');
});

// ─────────────────────────────────────────────────────────────────────
// estimateTokenCount
// ─────────────────────────────────────────────────────────────────────

test('estimateTokenCount: chars/4 heuristic, images flat-rated, floor of 1', () => {
  // 400 chars of user text → 100 tokens + 4 per-message overhead
  const textOnly = estimateTokenCount({
    model: 'm',
    messages: [{ role: 'user', content: 'x'.repeat(400) }],
  });
  assert.equal(textOnly, 104);

  // an image adds a flat 1600, not base64-length/4
  const withImage = estimateTokenCount({
    model: 'm',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'x'.repeat(400) },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'A'.repeat(100_000) } },
        ],
      },
    ],
  });
  assert.equal(withImage, 104 + 1600);

  // system + tools count; result is never below 1
  assert.equal(estimateTokenCount({ model: 'm', messages: [] }), 1);
  const withTools = estimateTokenCount({
    model: 'm',
    system: 'y'.repeat(40),
    messages: [{ role: 'user', content: 'hi' }],
    tools: [{ name: 'f', description: 'd', input_schema: { type: 'object' } }],
  });
  assert.ok(withTools > 10, 'tools and system contribute to the estimate');
});
