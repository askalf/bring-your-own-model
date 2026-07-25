// Fuzz openAIToAnthropicResponse — the mapper byom runs over a full upstream
// OpenAI chat-completions response body to build the Anthropic Messages
// response the client sees. The upstream body is wire input byom does not
// control. Contract: NEVER throw on a hostile body (missing/primitive choices,
// message, tool_calls, usage; prototype-named fields), and always return a
// well-formed Anthropic message envelope — role 'assistant', an array of
// text/tool_use blocks each with the required fields, and numeric usage.
import { openAIToAnthropicResponse } from '../dist/translate-chat.js';

function checkResponse(r) {
  if (!r || typeof r !== 'object') throw new Error('response is not an object');
  if (r.type !== 'message' || r.role !== 'assistant') {
    throw new Error(`malformed envelope: ${JSON.stringify({ type: r.type, role: r.role })}`);
  }
  if (!Array.isArray(r.content)) throw new Error('content is not an array');
  for (const block of r.content) {
    if (block.type === 'text') {
      if (typeof block.text !== 'string') throw new Error('text block has non-string text');
    } else if (block.type === 'tool_use') {
      if (typeof block.name !== 'string' || typeof block.id !== 'string') {
        throw new Error(`malformed tool_use block: ${JSON.stringify(block)}`);
      }
    } else {
      throw new Error(`unexpected block type: ${JSON.stringify(block.type)}`);
    }
  }
  if (!r.usage || !Number.isFinite(r.usage.input_tokens) || !Number.isFinite(r.usage.output_tokens)) {
    throw new Error(`malformed usage: ${JSON.stringify(r.usage)}`);
  }
}

function tryParse(s) {
  try { return JSON.parse(s); } catch { return undefined; }
}

export function fuzz(data) {
  const s = data.toString('utf8');
  const model = s.slice(0, 24) || 'gpt-4o';

  // Raw fuzz body parsed as a response, when it happens to be JSON.
  const parsed = tryParse(s);
  if (parsed && typeof parsed === 'object') checkResponse(openAIToAnthropicResponse(parsed, model));

  // Hostile-but-structured bodies: real response shape with fuzz-derived
  // values in each field the mapper reaches into.
  const hostiles = [
    { id: s.slice(0, 16), choices: [{ message: { content: s }, finish_reason: s.slice(0, 12) }], usage: { prompt_tokens: 1, completion_tokens: 2 } },
    { id: s.slice(0, 16), choices: [{ message: { refusal: s } }] },
    { choices: [{ message: { tool_calls: [{ id: s.slice(0, 8), function: { name: s.slice(0, 16), arguments: s } }] } }] },
    { choices: [{ message: { tool_calls: [null, s, { function: { arguments: '{' } }] } }] },
    { choices: s.length },
    { choices: [{ message: { content: s.slice(0, 8) } }], usage: s },
    { ['__proto__']: { polluted: true }, choices: [{ message: { content: s.slice(0, 8) } }] },
  ];
  for (const h of hostiles) checkResponse(openAIToAnthropicResponse(h, model));

  if (({}).polluted !== undefined) {
    throw new Error('prototype pollution via openAIToAnthropicResponse');
  }
}
