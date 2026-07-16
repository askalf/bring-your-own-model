#!/usr/bin/env node
/**
 * byom CLI — start the Claude Code → OpenAI bridge.
 *
 *   byom --model gpt-5.6-sol [--port 8788] [--base-url URL] [-v]
 *
 * Reads OPENAI_API_KEY from the environment. Prints a startup banner with
 * the exact env vars to point Claude Code at the bridge.
 */

import {
  createBridgeServer,
  pickApi,
  DEFAULT_PORT,
  DEFAULT_BASE_URL,
  type BridgeConfig,
} from './proxy.js';

interface CliArgs {
  model?: string;
  fastModel?: string;
  port: number;
  baseUrl: string;
  verbose: boolean;
  help: boolean;
}

const USAGE = `byom — run Claude Code on any OpenAI / OpenAI-compatible model

Usage:
  byom --model <id> [options]

Options:
  --model <id>        Target model to route every request to (required).
                      e.g. gpt-5.6-sol, gpt-4o, llama-3.3-70b
  --fast-model <id>   Cheaper model for Claude Code's haiku-tier sub-agent
                      requests (Explore/Task fan-outs). Unset = everything
                      runs on --model. e.g. --fast-model gpt-4o
  --port <n>          Port to listen on (default: ${DEFAULT_PORT}).
  --base-url <url>    OpenAI-compatible base URL
                      (default: ${DEFAULT_BASE_URL}).
                      Point this at Groq / OpenRouter / LiteLLM / a local
                      Ollama exposing /v1, etc.
  -v, --verbose       Log every request's target model + usage/reasoning
                      tokens (proof it ran on the other model).
  -h, --help          Show this help.

Environment:
  OPENAI_API_KEY      Backend API key (required). For a local provider that
                      needs no key, set it to any non-empty placeholder.

Example:
  export OPENAI_API_KEY=sk-...
  byom --model gpt-5.6-sol --port 8788
`;

/** Parse argv (no dependencies). Throws on malformed / unknown flags. */
export function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    port: DEFAULT_PORT,
    baseUrl: DEFAULT_BASE_URL,
    verbose: false,
    help: false,
  };

  // Pull a value for `--flag value` or `--flag=value`. `inlineValue` is the
  // part after `=` when present; otherwise the next argv token is consumed.
  const readValue = (flag: string, inlineValue: string | undefined, next: string | undefined): string => {
    if (inlineValue !== undefined) return inlineValue;
    if (next === undefined) throw new Error(`Missing value for ${flag}`);
    return next;
  };

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    const eq = token.indexOf('=');
    const flag = token.startsWith('--') && eq > 0 ? token.slice(0, eq) : token;
    const inline = token.startsWith('--') && eq > 0 ? token.slice(eq + 1) : undefined;

    switch (flag) {
      case '--model': {
        const raw = readValue(flag, inline, argv[i + 1]);
        if (inline === undefined) i++;
        args.model = raw;
        break;
      }
      case '--fast-model': {
        const raw = readValue(flag, inline, argv[i + 1]);
        if (inline === undefined) i++;
        args.fastModel = raw;
        break;
      }
      case '--port': {
        const raw = readValue(flag, inline, argv[i + 1]);
        if (inline === undefined) i++;
        const port = Number.parseInt(raw, 10);
        if (!Number.isInteger(port) || port < 1 || port > 65535) {
          throw new Error(`Invalid --port: ${raw} (expected 1-65535)`);
        }
        args.port = port;
        break;
      }
      case '--base-url': {
        const raw = readValue(flag, inline, argv[i + 1]);
        if (inline === undefined) i++;
        try {
          new URL(raw); // validate only — throws on a malformed URL
        } catch {
          throw new Error(`Invalid --base-url: ${raw}`);
        }
        args.baseUrl = raw;
        break;
      }
      case '-v':
      case '--verbose':
        args.verbose = true;
        break;
      case '-h':
      case '--help':
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${token}`);
    }
  }

  return args;
}

function main(): void {
  let args: CliArgs;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`Error: ${err instanceof Error ? err.message : String(err)}\n`);
    process.stderr.write(USAGE);
    process.exit(2);
    return;
  }

  if (args.help) {
    process.stdout.write(USAGE);
    process.exit(0);
    return;
  }

  if (!args.model) {
    console.error('Error: --model <id> is required.\n');
    process.stderr.write(USAGE);
    process.exit(2);
    return;
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error(
      'Error: OPENAI_API_KEY is not set.\n' +
      '  Set your backend key, e.g.  export OPENAI_API_KEY=sk-...\n' +
      '  (For a local provider that needs no key, set any non-empty placeholder.)',
    );
    process.exit(1);
    return;
  }

  const config: BridgeConfig = {
    model: args.model,
    fastModel: args.fastModel,
    baseUrl: args.baseUrl,
    apiKey,
    verbose: args.verbose,
  };

  const server = createBridgeServer(config);

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Error: port ${args.port} is already in use. Pick another with --port.`);
    } else {
      console.error(`Error: ${err.message}`);
    }
    process.exit(1);
  });

  server.listen(args.port, () => {
    const url = `http://localhost:${args.port}`;
    const api = pickApi(args.baseUrl, args.model!);
    const banner = [
      '',
      'byom — Claude Code → OpenAI bridge',
      `  listening    ${url}`,
      `  model        ${args.model}`,
      args.fastModel ? `  fast model   ${args.fastModel} (haiku-tier sub-agents)` : '',
      `  base url     ${args.baseUrl}`,
      `  api surface  ${api}`,
      args.verbose ? '  verbose      on' : '',
      '',
      'Point Claude Code at it (in a separate shell):',
      `  export ANTHROPIC_BASE_URL=${url}`,
      '  export ANTHROPIC_API_KEY=any-placeholder',
      '  claude',
      '',
    ].filter((line) => line !== '');
    process.stdout.write(banner.join('\n') + '\n');
  });

  // Clean shutdown on Ctrl-C.
  const shutdown = (): void => {
    server.close(() => process.exit(0));
    // Force-exit if connections linger.
    setTimeout(() => process.exit(0), 1000).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
