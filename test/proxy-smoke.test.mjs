#!/usr/bin/env node
// End-to-end smoke test for the bridge server plumbing — NO network, NO API
// key. `globalThis.fetch` is stubbed to return canned OpenAI payloads, then a
// real HTTP request is POSTed to a listening server on an ephemeral port and
// the Anthropic-shaped reply is asserted. This proves routing + translate +
// forward + translate-back wire up correctly without touching a real provider.
//
// The client request uses node:http (NOT fetch, which is stubbed for the
// upstream leg). Run: node --test test/proxy-smoke.test.mjs (build first).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';

import { createBridgeServer, resolveOpenAITarget } from '../dist/proxy.js';

// A non-secret placeholder key (deliberately NOT of the `sk-…` real shape).
const TEST_KEY = 'test-key-123';

// ─────────────────────────────────────────────────────────────────────
// Harness helpers
// ─────────────────────────────────────────────────────────────────────

/** Start a bridge server on an ephemeral port; resolve { server, port }. */
function startServer(config) {
  const server = createBridgeServer(config);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

function closeServer(server) {
  return new Promise((resolve) => server.close(resolve));
}

/** POST a JSON body to the running bridge; resolve { status, text }. */
function postJson(port, path, bodyObj) {
  const payload = Buffer.from(JSON.stringify(bodyObj), 'utf8');
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json', 'content-length': payload.length } },
      (res) => {
        let text = '';
        res.setEncoding('utf8');
        res.on('data', (c) => { text += c; });
        res.on('end', () => resolve({ status: res.statusCode, text }));
      },
    );
    req.on('error', reject);
    req.end(payload);
  });
}

/**
 * Install a stubbed `globalThis.fetch` for the duration of `fn`. `factory`
 * receives (url, init) and returns a Response. The last call's (url, init) are
 * recorded on the returned `calls` array so the test can assert the outbound
 * leg. Always restores the real fetch.
 */
async function withStubbedFetch(factory, fn) {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return factory(String(url), init);
  };
  try {
    return await fn(calls);
  } finally {
    globalThis.fetch = original;
  }
}

/** Parse an Anthropic SSE body into [{ type, data }] events. */
function parseAnthropicSSE(text) {
  const events = [];
  for (const record of text.split('\n\n')) {
    if (!record.trim()) continue;
    let type;
    let data;
    for (const line of record.split('\n')) {
      if (line.startsWith('event:')) type = line.slice(6).trim();
      else if (line.startsWith('data:')) data = line.slice(5).trim();
    }
    if (data) events.push({ type, data: JSON.parse(data) });
  }
  return events;
}

const sseFrame = (type, obj) => `event: ${type}\ndata: ${JSON.stringify({ type, ...obj })}\n\n`;

// ─────────────────────────────────────────────────────────────────────
// Responses API path (api.openai.com + gpt-5* ⇒ /responses) — non-streaming
// ─────────────────────────────────────────────────────────────────────

const RESPONSES_JSON = {
  id: 'resp_smoke1',
  object: 'response',
  model: 'gpt-5.6-sol',
  status: 'completed',
  output: [
    { id: 'rs_1', type: 'reasoning', summary: [{ type: 'summary_text', text: 'I should list files.' }] },
    { id: 'msg_1', type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'Listing now.' }] },
    { id: 'fc_1', type: 'function_call', call_id: 'call_smoke', name: 'Bash', arguments: '{"command":"ls -a"}', status: 'completed' },
  ],
  usage: { input_tokens: 100, output_tokens: 20, output_tokens_details: { reasoning_tokens: 8 } },
};

test('Responses non-streaming → well-formed Anthropic message (thinking + text + tool_use)', async () => {
  const { server, port } = await startServer({ model: 'gpt-5.6-sol', baseUrl: 'https://api.openai.com/v1', apiKey: TEST_KEY });
  try {
    const { status, text } = await withStubbedFetch(
      () => new Response(JSON.stringify(RESPONSES_JSON), { status: 200, headers: { 'content-type': 'application/json' } }),
      (calls) =>
        postJson(port, '/v1/messages', {
          model: 'claude-opus-4-8',
          max_tokens: 256,
          messages: [{ role: 'user', content: 'list files' }],
          tools: [{ name: 'Bash', input_schema: { type: 'object', properties: { command: { type: 'string' } } } }],
        }).then((r) => {
          // outbound leg went to /responses with the configured key
          assert.equal(calls.length, 1);
          assert.ok(calls[0].url.endsWith('/responses'), `forwarded to Responses API, got ${calls[0].url}`);
          assert.equal(calls[0].init.headers.Authorization, `Bearer ${TEST_KEY}`);
          return r;
        }),
    );

    assert.equal(status, 200);
    const body = JSON.parse(text);
    assert.equal(body.type, 'message');
    assert.equal(body.role, 'assistant');
    assert.equal(body.model, 'gpt-5.6-sol', 'echoes the target model');
    assert.equal(body.stop_reason, 'tool_use');
    assert.deepEqual(body.usage, { input_tokens: 100, output_tokens: 20 });

    const types = body.content.map((b) => b.type);
    assert.deepEqual(types, ['thinking', 'text', 'tool_use']);
    const tool = body.content.find((b) => b.type === 'tool_use');
    assert.equal(tool.name, 'Bash');
    assert.equal(tool.id, 'call_smoke');
    assert.deepEqual(tool.input, { command: 'ls -a' });
  } finally {
    await closeServer(server);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Responses API path — streaming SSE
// ─────────────────────────────────────────────────────────────────────

const RESPONSES_SSE =
  sseFrame('response.created', { response: { id: 'resp_s', model: 'gpt-5.6-sol', status: 'in_progress' } }) +
  sseFrame('response.output_item.added', { output_index: 0, item: { id: 'fc_1', type: 'function_call', call_id: 'call_s', name: 'Bash', arguments: '' } }) +
  sseFrame('response.function_call_arguments.delta', { output_index: 0, item_id: 'fc_1', delta: '{"command":' }) +
  sseFrame('response.function_call_arguments.delta', { output_index: 0, item_id: 'fc_1', delta: '"uname -a"}' }) +
  sseFrame('response.output_item.done', { output_index: 0, item: { id: 'fc_1', type: 'function_call', call_id: 'call_s', name: 'Bash', arguments: '{"command":"uname -a"}' } }) +
  sseFrame('response.completed', { response: { id: 'resp_s', model: 'gpt-5.6-sol', status: 'completed', usage: { input_tokens: 50, output_tokens: 12 } } });

test('Responses streaming SSE → well-formed Anthropic event stream (tool_use)', async () => {
  const { server, port } = await startServer({ model: 'gpt-5.6-sol', baseUrl: 'https://api.openai.com/v1', apiKey: TEST_KEY });
  try {
    const { status, text } = await withStubbedFetch(
      () => new Response(RESPONSES_SSE, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
      () =>
        postJson(port, '/v1/messages', {
          model: 'claude-opus-4-8',
          max_tokens: 256,
          stream: true,
          messages: [{ role: 'user', content: 'what OS is this?' }],
          tools: [{ name: 'Bash', input_schema: { type: 'object', properties: { command: { type: 'string' } } } }],
        }),
    );

    assert.equal(status, 200);
    const events = parseAnthropicSSE(text);
    assert.equal(events[0].type, 'message_start');
    assert.equal(events[events.length - 1].type, 'message_stop');

    const start = events.find((e) => e.type === 'content_block_start');
    assert.equal(start.data.content_block.type, 'tool_use');
    assert.equal(start.data.content_block.id, 'call_s');
    assert.equal(start.data.content_block.name, 'Bash');

    const argJson = events
      .filter((e) => e.type === 'content_block_delta' && e.data.delta.type === 'input_json_delta')
      .map((e) => e.data.delta.partial_json)
      .join('');
    assert.deepEqual(JSON.parse(argJson), { command: 'uname -a' });

    const delta = events.find((e) => e.type === 'message_delta');
    assert.equal(delta.data.delta.stop_reason, 'tool_use');
    assert.deepEqual(delta.data.usage, { output_tokens: 12, input_tokens: 50 });
  } finally {
    await closeServer(server);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Chat Completions path (non-openai.com base ⇒ /chat/completions)
// ─────────────────────────────────────────────────────────────────────

const CHAT_JSON = {
  id: 'chatcmpl_smoke',
  object: 'chat.completion',
  model: 'llama-3.3-70b',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: 'Here you go.',
        tool_calls: [{ id: 'call_c', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
      },
      finish_reason: 'tool_calls',
    },
  ],
  usage: { prompt_tokens: 30, completion_tokens: 10 },
};

test('Chat Completions non-streaming → well-formed Anthropic message (text + tool_use)', async () => {
  const { server, port } = await startServer({ model: 'llama-3.3-70b', baseUrl: 'https://api.groq.com/openai/v1', apiKey: TEST_KEY });
  try {
    const { status, text } = await withStubbedFetch(
      () => new Response(JSON.stringify(CHAT_JSON), { status: 200, headers: { 'content-type': 'application/json' } }),
      (calls) =>
        postJson(port, '/v1/messages', {
          model: 'claude-opus-4-8',
          max_tokens: 256,
          messages: [{ role: 'user', content: 'weather in Paris?' }],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: { city: { type: 'string' } } } }],
        }).then((r) => {
          assert.ok(calls[0].url.endsWith('/chat/completions'), `forwarded to Chat Completions, got ${calls[0].url}`);
          return r;
        }),
    );

    assert.equal(status, 200);
    const body = JSON.parse(text);
    assert.equal(body.type, 'message');
    assert.equal(body.stop_reason, 'tool_use');
    assert.deepEqual(body.content.map((b) => b.type), ['text', 'tool_use']);
    const tool = body.content.find((b) => b.type === 'tool_use');
    assert.equal(tool.name, 'get_weather');
    assert.deepEqual(tool.input, { city: 'Paris' });
    assert.deepEqual(body.usage, { input_tokens: 30, output_tokens: 10 });
  } finally {
    await closeServer(server);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Upstream error surfaces as an Anthropic-shaped error (no key leak)
// ─────────────────────────────────────────────────────────────────────

test('upstream non-2xx → Anthropic-shaped error, no upstream detail leaked', async () => {
  const { server, port } = await startServer({ model: 'gpt-5.6-sol', baseUrl: 'https://api.openai.com/v1', apiKey: TEST_KEY });
  try {
    const { status, text } = await withStubbedFetch(
      () => new Response('{"error":{"message":"insufficient_quota — secret detail"}}', { status: 429, headers: { 'content-type': 'application/json' } }),
      () => postJson(port, '/v1/messages', { model: 'claude-opus-4-8', max_tokens: 8, messages: [{ role: 'user', content: 'hi' }] }),
    );

    assert.equal(status, 429);
    const body = JSON.parse(text);
    assert.equal(body.type, 'error');
    assert.equal(body.error.type, 'rate_limit_error');
    assert.ok(!text.includes('secret detail'), 'upstream error detail must not leak to the client');
  } finally {
    await closeServer(server);
  }
});

// ─────────────────────────────────────────────────────────────────────
// count_tokens is a local estimate (no upstream call)
// ─────────────────────────────────────────────────────────────────────

test('count_tokens returns a local input_tokens estimate without calling upstream', async () => {
  const { server, port } = await startServer({ model: 'gpt-5.6-sol', baseUrl: 'https://api.openai.com/v1', apiKey: TEST_KEY });
  try {
    const { status, text } = await withStubbedFetch(
      () => { throw new Error('count_tokens must not hit the network'); },
      () => postJson(port, '/v1/messages/count_tokens', { model: 'claude-opus-4-8', messages: [{ role: 'user', content: 'x'.repeat(400) }] }),
    );
    assert.equal(status, 200);
    const body = JSON.parse(text);
    assert.equal(typeof body.input_tokens, 'number');
    assert.ok(body.input_tokens > 0);
  } finally {
    await closeServer(server);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Tier-aware routing (--fast-model) — pure resolveOpenAITarget units
// ─────────────────────────────────────────────────────────────────────

test('fast-model: haiku-tier request routes to the cheap model', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'claude-haiku-4-5-20251001',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
    fastModel: 'gpt-4o',
  });
  assert.equal(target.model, 'gpt-4o');
  assert.equal(target.tier, 'fast');
  assert.equal(target.api, 'chat'); // surface follows the CHOSEN model
});

test('fast-model: non-haiku request stays on the primary', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'claude-fable-5',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
    fastModel: 'gpt-4o',
  });
  assert.equal(target.model, 'gpt-5.6-sol');
  assert.equal(target.tier, 'primary');
  assert.equal(target.api, 'responses');
});

test('fast-model: unset keeps everything on the primary (back-compat)', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'claude-haiku-4-5-20251001',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
  });
  assert.equal(target.model, 'gpt-5.6-sol');
  assert.equal(target.tier, 'primary');
});

test('fast-model: openai:-family prefix on the flag value is stripped', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'claude-haiku-4-5-20251001',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
    fastModel: 'openai:gpt-4o',
  });
  assert.equal(target.model, 'gpt-4o');
  assert.equal(target.tier, 'fast');
});

test('fast-model: a claude:-prefixed value degrades safely to the primary', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'claude-haiku-4-5-20251001',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
    fastModel: 'claude:haiku',
  });
  assert.equal(target.model, 'gpt-5.6-sol');
  assert.equal(target.tier, 'primary');
});

test('fast-model: an explicit per-request openai: override wins over tiering', () => {
  const target = resolveOpenAITarget({
    path: '/v1/messages',
    model: 'openai:gpt-4.1-mini',
    forcedModel: 'gpt-5.6-sol',
    baseUrl: 'https://api.openai.com/v1',
    fastModel: 'gpt-4o',
  });
  assert.equal(target.model, 'gpt-4.1-mini');
  assert.equal(target.tier, 'primary');
});
