import type { OpenAITool, OpenAIToolCall } from './types';
import { generateId } from './arnaru';

function allowedNames(tools?: OpenAITool[]): Set<string> {
  return new Set(
    (tools || [])
      .filter(t => t?.type === 'function' && !!t.function?.name)
      .map(t => t.function.name)
  );
}

function normalizeArgs(value: unknown): Record<string, unknown> {
  if (value == null) return {};
  if (typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === 'string') {
    const raw = value.trim();
    if (!raw) return {};
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {}
    return { input: value };
  }
  return { input: value };
}

function makeCall(name: string, args: unknown, id?: unknown): OpenAIToolCall {
  return {
    id: typeof id === 'string' && id ? id : `call_${generateId()}`,
    type: 'function',
    function: {
      name,
      arguments: JSON.stringify(normalizeArgs(args))
    }
  };
}

function collectFromObject(
  obj: any,
  allowed: Set<string>,
  out: OpenAIToolCall[]
) {
  if (!obj || typeof obj !== 'object') return;

  const add = (name: unknown, args: unknown, id?: unknown) => {
    if (typeof name !== 'string' || !allowed.has(name)) return;
    const call = makeCall(name, args, id);
    const duplicate = out.some(
      x => x.function.name === call.function.name &&
           x.function.arguments === call.function.arguments
    );
    if (!duplicate) out.push(call);
  };

  if (Array.isArray(obj.tool_calls)) {
    for (const c of obj.tool_calls) {
      add(
        c?.function?.name || c?.name || c?.tool,
        c?.function?.arguments ?? c?.arguments ?? c?.parameters,
        c?.id
      );
    }
  }

  if (obj.tool_call) {
    const c = obj.tool_call;
    add(
      c?.function?.name || c?.name || c?.tool,
      c?.function?.arguments ?? c?.arguments ?? c?.parameters,
      c?.id
    );
  }

  if (obj.function_call) {
    const c = obj.function_call;
    add(c?.name || c?.function?.name, c?.arguments ?? c?.function?.arguments, c?.id);
  }

  if (obj.name || obj.tool) {
    let args = obj.arguments ?? obj.parameters;
    if (args === undefined) {
      const copy = { ...obj };
      delete copy.name;
      delete copy.tool;
      delete copy.id;
      args = copy;
    }
    add(obj.name || obj.tool, args, obj.id);
  }
}

function balancedJsonCandidates(text: string): string[] {
  const result: string[] = [];
  let start = -1;
  let depth = 0;
  let quote = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') quote = false;
      continue;
    }
    if (c === '"') {
      quote = true;
      continue;
    }
    if (c === '{') {
      if (depth === 0) start = i;
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0 && start >= 0) {
        result.push(text.slice(start, i + 1));
        start = -1;
      }
    }
  }
  return result;
}

/**
 * Parses the formats documented by FreeDeepseekAPI plus DeepSeek DSML.
 * Everything is normalized into OpenAI tool_calls.
 */
export function parseToolCalls(text: string, tools?: OpenAITool[]): OpenAIToolCall[] {
  const allowed = allowedNames(tools);
  if (!allowed.size || !text) return [];

  const out: OpenAIToolCall[] = [];

  // 1. Native-looking JSON envelopes.
  for (const raw of balancedJsonCandidates(text)) {
    try {
      collectFromObject(JSON.parse(raw), allowed, out);
    } catch {}
  }

  // 2. Explicit TOOL_CALL: JSON.
  for (const m of text.matchAll(/TOOL_CALL\s*:\s*([\s\S]+?)(?=\n|$)/gi)) {
    try {
      collectFromObject(JSON.parse(m[1].trim()), allowed, out);
    } catch {}
  }

  // 3. XML <tool_call>...</tool_call>.
  for (const m of text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi)) {
    try {
      collectFromObject(JSON.parse(m[1].trim()), allowed, out);
    } catch {}
  }

  // 4. DeepSeek DSML. Handle full wrapper and also invoke blocks when
  // the wrapper token is missing (a known long-context failure mode).
  const invokeRe =
    /<｜DSML｜invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/｜DSML｜invoke>/g;
  for (const m of text.matchAll(invokeRe)) {
    const args: Record<string, unknown> = {};
    const body = m[2];

    const paramRe =
      /<｜DSML｜parameter\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>([\s\S]*?)<\/｜DSML｜parameter>/g;

    for (const p of body.matchAll(paramRe)) {
      const key = p[1];
      const isString = p[2] === 'true';
      const raw = p[3].replace(/^\n|\n$/g, '');
      if (isString) {
        args[key] = raw;
      } else {
        try {
          args[key] = JSON.parse(raw);
        } catch {
          args[key] = raw;
        }
      }
    }

    if (allowed.has(m[1])) {
      const call = makeCall(m[1], args);
      if (!out.some(x => x.function.name === call.function.name &&
                         x.function.arguments === call.function.arguments)) {
        out.push(call);
      }
    }
  }

  // 5. Common "Called function NAME" fallback.
  const called = [...text.matchAll(/Called function\s+([A-Za-z0-9_.:-]+)/gi)];
  if (called.length) {
    const candidates = balancedJsonCandidates(text);
    for (const m of called) {
      if (!allowed.has(m[1])) continue;
      let args: unknown = {};
      for (const raw of candidates) {
        try {
          const obj = JSON.parse(raw);
          args = obj.arguments ?? obj.parameters ?? obj;
          break;
        } catch {}
      }
      const call = makeCall(m[1], args);
      if (!out.some(x => x.function.name === call.function.name &&
                         x.function.arguments === call.function.arguments)) {
        out.push(call);
      }
    }
  }

  return out;
}

export function hasToolMarkup(text: string): boolean {
  return /TOOL_CALL\s*:|<tool_call\b|<｜DSML｜(?:tool_calls|toolcalls|tool)>|<｜DSML｜invoke\b|["'](?:tool_call|tool_calls|function_call)["']\s*:/i.test(text || '');
}
