// Fuzz the OpenAI chat-completions → Anthropic SSE stream translator — the
// push(chunk)/end() state machine byom drives with each parsed upstream chunk
// to produce the Anthropic Messages stream a client consumes. The upstream
// stream is wire input byom does not control; the contract: NEVER throw on a
// hostile chunk (events whose delta/tool_calls fields are missing, primitives,
// or prototype-named junk, out-of-order tool indices, an empty stream), every
// emitted event serializes to a well-formed `event: <type>\ndata: <json>\n\n`
// frame, and content-block indices are strictly increasing so the client's
// block bookkeeping never corrupts. A throw here kills a client's live stream.
import {
  openAIStreamToAnthropicSSE,
  parseOpenAISSELine,
  formatAnthropicSSE,
} from '../dist/translate-chat.js';

function checkEvents(events) {
  if (!Array.isArray(events)) throw new Error('translator returned a non-array');
  let lastIndex = -1;
  for (const ev of events) {
    if (!ev || typeof ev.type !== 'string') {
      throw new Error(`event without a string type: ${JSON.stringify(ev)}`);
    }
    // Serialization must produce a parseable SSE frame.
    const frame = formatAnthropicSSE(ev);
    if (typeof frame !== 'string' || !frame.startsWith('event: ')) {
      throw new Error(`formatAnthropicSSE produced a non-SSE frame: ${JSON.stringify(String(frame).slice(0, 80))}`);
    }
    for (const line of frame.split('\n')) {
      if (line.startsWith('data: ')) JSON.parse(line.slice(6));
    }
    // Block indices only ever increase.
    if (ev.type === 'content_block_start') {
      if (typeof ev.index === 'number') {
        if (ev.index <= lastIndex) throw new Error(`non-increasing block index ${ev.index} after ${lastIndex}`);
        lastIndex = ev.index;
      }
    }
  }
}

export function fuzz(data) {
  const s = data.toString('utf8');

  // Parse the fuzz input as a stream of SSE lines and push whatever chunks
  // come out — parser + translator exercised end to end.
  const t1 = openAIStreamToAnthropicSSE({ requestModel: s.slice(0, 24) || 'gpt-4o' });
  for (const line of s.split('\n')) {
    const chunk = parseOpenAISSELine(line);
    if (chunk) checkEvents(t1.push(chunk));
  }
  checkEvents(t1.end());

  // Hostile-but-structured chunks: real chunk fields with adversarial values,
  // including out-of-order and prototype-named tool-call shapes.
  const t2 = openAIStreamToAnthropicSSE();
  const hostiles = [
    { id: s.slice(0, 16), choices: [{ delta: { role: 'assistant' } }] },
    { choices: [{ delta: { content: s } }] },
    { choices: [{ delta: { content: s.length } }] },
    { choices: [{ delta: { refusal: s } }] },
    { choices: [{ delta: { tool_calls: [{ index: 7, id: s.slice(0, 8), function: { name: s.slice(0, 16), arguments: s } }] } }] },
    { choices: [{ delta: { tool_calls: [{ index: 3, function: { arguments: s } }] } }] },
    { choices: [{ delta: { tool_calls: s } }] },
    { choices: [{ delta: { content: s.slice(0, 4) }, finish_reason: s.slice(0, 12) }] },
    { choices: [{ delta: { ['__proto__']: { polluted: true } } }] },
    { choices: s.length },
  ];
  for (const h of hostiles) checkEvents(t2.push(h));
  checkEvents(t2.end());
  // end() is idempotent — a second call must stay safe.
  checkEvents(t2.end());

  if (({}).polluted !== undefined) {
    throw new Error('prototype pollution via the stream translator');
  }
}
