// Fuzz parseOpenAISSELine — the parser byom runs over every raw line of the
// upstream chat-completions SSE stream to recover one chat.completion.chunk.
// Upstream stream lines are wire input byom does not control. Contract: NEVER
// throw on a hostile line (broken JSON, a bare `data:`, CR-terminated junk,
// prototype-named payloads), and return only `null` or a parsed object — never
// a primitive that a downstream `chunk?.choices` access would crash on. A
// throw here kills a client's live stream mid-response.
import { parseOpenAISSELine } from '../dist/translate-chat.js';

function check(out) {
  if (out === null) return;
  if (typeof out !== 'object') {
    throw new Error(`parseOpenAISSELine returned a non-object: ${JSON.stringify(out)}`);
  }
}

export function fuzz(data) {
  const s = data.toString('utf8');

  // Raw fuzz line, plus variants forced through the `data:` gate and the
  // CR/`[DONE]` edges the parser special-cases.
  check(parseOpenAISSELine(s));
  check(parseOpenAISSELine(`data: ${s}`));
  check(parseOpenAISSELine(`data:${s}\r`));
  check(parseOpenAISSELine(`data: [DONE]`));

  // Hostile-but-parseable chunk shapes: real chunk fields carrying adversarial
  // values, so any later property access runs on prototype-named junk too.
  const hostiles = [
    { id: s.slice(0, 16), choices: [{ delta: { content: s } }] },
    { id: s.slice(0, 16), choices: [{ delta: { tool_calls: [{ index: 0, id: s.slice(0, 8), function: { name: s.slice(0, 16), arguments: s } }] } }] },
    { choices: s },
    { ['__proto__']: { polluted: true }, choices: [{ delta: { content: s.slice(0, 8) } }] },
  ];
  for (const h of hostiles) check(parseOpenAISSELine(`data: ${JSON.stringify(h)}`));

  if (({}).polluted !== undefined) {
    throw new Error('prototype pollution via parseOpenAISSELine');
  }
}
