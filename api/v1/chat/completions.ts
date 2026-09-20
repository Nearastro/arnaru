import type { VercelRequest, VercelResponse } from '@vercel/node';
import type { OpenAIChatRequest, OpenAITool, OpenAIToolCall } from '../../../lib/types';
import { parseSSE, parseSSEStream } from '../../../lib/sse-parser';
import {
  buildArnaruRequest,
  callArnaruChat,
  callArnaruChatWithFiles,
  generateId,
  getTimestamp
} from '../../../lib/arnaru';
import {
  appendPromptInstruction,
  isEmptyContent,
  isContextTooLongError,
  looksLikeToolCallMarkup,
  MAX_EMPTY_RETRIES,
  MAX_UPSTREAM_PROMPT_CHARS,
  MIN_UPSTREAM_PROMPT_CHARS,
  sanitizeContent,
  truncatePromptMiddle
} from '../../../lib/recovery';

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb'
    }
  }
};

const VALID_MODELS = new Set([
  'claude-fable-5',
  'claude-haiku-4.5',
  'claude-opus-4.6',
  'claude-opus-4.7',
  'claude-opus-4.8',
  'claude-sonnet-4',
  'claude-sonnet-4.6',
  'claude-sonnet-5',
  'deepseek-r1',
  'deepseek-v3.1',
  'deepseek-v3.2',
  'deepseek-v3.2-online',
  'deepseek-v3.2-think',
  'deepseek-v4-flash',
  'deepseek-v4-pro',
  'gemini-2.0-flash',
  'gemini-2.5-flash',
  'gemini-2.5-pro',
  'gemini-3-flash',
  'gemini-3-pro',
  'gemini-3.1-flash',
  'gemini-3.1-pro',
  'gemini-3.5-flash',
  'gemini-3.5-flash-lite',
  'gemini-3.6-flash',
  'gpt-4.1',
  'gpt-4.1-mini',
  'gpt-4o',
  'gpt-5',
  'gpt-5-mini',
  'gpt-5-nano',
  'gpt-5.1',
  'gpt-5.2',
  'gpt-5.4',
  'gpt-5.5',
  'gpt-5.6-luna',
  'gpt-5.6-sol',
  'gpt-o3-mini',
  'grok-3',
  'grok-3-reasoner',
  'grok-4',
  'grok-4-fast',
  'grok-4-reasoning',
  'grok-4.1',
  'grok-4.1-fast',
  'grok-4.1-reasoning',
  'grok-4.2',
  'grok-4.2-reasoning',
  'grok-4.3-pro',
  'grok-4.3-reasoning',
  'grok-4.5',
  'kimi-k3',
  'llama-4',
  'llama-4.1',
  'mistral-small-3.2',
  'mistral-small-creative',
  'qwen-vl-max',
  'qwen3-235b',
  'qwen3-max',
  'skylark-pro',
  'step-3.5-flash',
  'step-3.5-flash-free'
]);

function validateModel(model: string): string {
  return VALID_MODELS.has(model)
    ? model
    : (process.env.DEFAULT_MODEL || 'claude-fable-5');
}

function getToolNames(tools?: OpenAITool[]): Set<string> {
  return new Set(
    (tools || [])
      .filter(tool => tool.type === 'function' && !!tool.function?.name)
      .map(tool => tool.function.name)
  );
}

function findJsonObjects(text: string): string[] {
  const results: string[] = [];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/gi);

  if (fenced) {
    for (const block of fenced) {
      const cleaned = block
        .replace(/^```(?:json)?/i, '')
        .replace(/```$/i, '')
        .trim();
      if (cleaned) results.push(cleaned);
    }
  }

  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];

    if (inString) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') inString = false;
      continue;
    }

    if (c === '"') {
      inString = true;
      continue;
    }

    if (c === '{') {
      if (depth === 0) start = i;
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0 && start >= 0) {
        results.push(text.slice(start, i + 1));
        start = -1;
      }
    }
  }

  return results;
}

function normalizeArguments(value: any): Record<string, any> {
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {
      return { input: value };
    }
  }

  if (value && typeof value === 'object') return value;
  return {};
}

function makeToolCall(name: string, args: any, suppliedId?: string): OpenAIToolCall {
  return {
    id: suppliedId || `call_${generateId()}`,
    type: 'function',
    function: {
      name,
      arguments: JSON.stringify(normalizeArguments(args))
    }
  };
}

function parseToolCalls(text: string, tools?: OpenAITool[]): OpenAIToolCall[] {
  const allowed = getToolNames(tools);
  if (!allowed.size) return [];

  const calls: OpenAIToolCall[] = [];
  const seen = new Set<string>();

  function add(name: any, args: any, id?: any) {
    if (typeof name !== 'string' || !allowed.has(name)) return;

    const normalized = normalizeArguments(args);
    const key = `${name}:${JSON.stringify(normalized)}`;
    if (seen.has(key)) return;

    seen.add(key);
    calls.push(
      makeToolCall(
        name,
        normalized,
        typeof id === 'string' ? id : undefined
      )
    );
  }

  const xmlMatches = text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi);
  if (xmlMatches) {
    for (const block of xmlMatches) {
      const cleaned = block
        .replace(/^<tool_call>\s*/i, '')
        .replace(/\s*<\/tool_call>$/i, '')
        .trim();

      try {
        const obj = JSON.parse(cleaned);

        if (obj?.name) {
          add(obj.name, obj.arguments ?? obj.parameters ?? {}, obj.id);
        }

        if (Array.isArray(obj?.tool_calls)) {
          for (const call of obj.tool_calls) {
            add(
              call?.function?.name || call?.name || call?.tool,
              call?.function?.arguments ?? call?.arguments ?? call?.parameters ?? {},
              call?.id
            );
          }
        }
      } catch {
        // Continue
      }
    }
  }

  const prefixMatches = [...text.matchAll(/TOOL_CALL\s*:\s*([\s\S]+)/gi)];
  for (const match of prefixMatches) {
    try {
      const obj = JSON.parse(match[1].trim());
      add(
        obj?.name || obj?.tool || obj?.function?.name,
        obj?.arguments ?? obj?.parameters ?? obj?.function?.arguments ?? {},
        obj?.id
      );
    } catch {
      // Continue
    }
  }

  const jsons = findJsonObjects(text);
  for (const raw of jsons) {
    try {
      const obj = JSON.parse(raw);
      if (!obj || typeof obj !== 'object') continue;

      if (Array.isArray(obj.tool_calls)) {
        for (const call of obj.tool_calls) {
          add(
            call?.function?.name || call?.name || call?.tool,
            call?.function?.arguments ?? call?.arguments ?? call?.parameters ?? {},
            call?.id
          );
        }
        continue;
      }

      if (obj.tool_call && typeof obj.tool_call === 'object') {
        const call = obj.tool_call;
        add(
          call.name || call.tool || call.function?.name,
          call.arguments ?? call.parameters ?? call.function?.arguments ?? {},
          call.id
        );
        continue;
      }

      if (obj.function_call && typeof obj.function_call === 'object') {
        const call = obj.function_call;
        add(call.name, call.arguments ?? {}, call.id);
        continue;
      }

      const name = typeof obj.name === 'string'
        ? obj.name
        : typeof obj.tool === 'string'
          ? obj.tool
          : undefined;

      if (!name) continue;

      let args = obj.arguments !== undefined ? obj.arguments : obj.parameters;
      if (args === undefined) {
        const copy = { ...obj };
        delete copy.name;
        delete copy.tool;
        args = copy;
      }

      add(name, args, obj.id);
    } catch {
      // Not valid JSON
    }
  }

  const calledMatches = [...text.matchAll(/Called function\s+([A-Za-z0-9_.:-]+)/gi)];
  if (calledMatches.length) {
    const jsonsFromCalled = findJsonObjects(text);
    for (const match of calledMatches) {
      const name = match[1];
      for (const raw of jsonsFromCalled) {
        try {
          const obj = JSON.parse(raw);
          add(name, obj);
          break;
        } catch {
          // Continue
        }
      }
    }
  }

  return calls;
}

function createCompletion(
  id: string,
  model: string,
  content: string,
  toolCalls: OpenAIToolCall[] = []
) {
  if (toolCalls.length) {
    return {
      id,
      object: 'chat.completion',
      created: getTimestamp(),
      model,
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            tool_calls: toolCalls
          },
          finish_reason: 'tool_calls'
        }
      ],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };
  }

  return {
    id,
    object: 'chat.completion',
    created: getTimestamp(),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: content || ''
        },
        finish_reason: 'stop'
      }
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };
}

function createStreamChunk(
  id: string,
  model: string,
  delta: any,
  finishReason: string | null = null
) {
  return `data: ${JSON.stringify({
    id,
    object: 'chat.completion.chunk',
    created: getTimestamp(),
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason
      }
    ]
  })}\n\n`;
}

function hasToolResult(body: OpenAIChatRequest): boolean {
  return body.messages.some((message: any) => message?.role === 'tool');
}

function hasAssistantToolCall(body: OpenAIChatRequest): boolean {
  return body.messages.some(
    (message: any) =>
      message?.role === 'assistant' &&
      Array.isArray(message?.tool_calls) &&
      message.tool_calls.length > 0
  );
}

function appendToolProtocol(systemPrompt: string, tools?: OpenAITool[]): string {
  if (!tools || !tools.length) return systemPrompt;

  const toolList = tools
    .filter(tool => tool.type === 'function' && tool.function?.name)
    .map(tool => JSON.stringify(tool))
    .join('\n');

  if (!toolList) return systemPrompt;

  return [
    systemPrompt,
    '',
    '=== AVAILABLE AGENT TOOLS ===',
    toolList,
    '',
    '=== TOOL CALL PROTOCOL ===',
    'When a tool is required, output ONLY one JSON object:',
    '{"name":"TOOL_NAME","arguments":{"argument":"value"}}',
    '',
    'TOOL_NAME must exactly match an available tool.',
    'Do not fabricate tool results.',
    'The external agent executes the tool.',
    'After a tool result is supplied, continue the ORIGINAL task.',
    'If the task is complete, answer normally.',
    'Never return an empty response.',
    '=== END TOOL PROTOCOL ==='
  ]
    .filter(Boolean)
    .join('\n');
}

function appendContinuation(systemPrompt: string): string {
  return [
    systemPrompt,
    '',
    '=== FINAL AGENT CONTINUATION ===',
    'A tool has already executed.',
    'Its real result is present in the conversation.',
    'Continue the original user task now.',
    '',
    'Return a normal natural-language answer if the task is complete.',
    'Use the tool result as factual context.',
    'Do not invent results.',
    'Do not say you are waiting for a tool.',
    'Do not return an empty response.',
    '=== END FINAL AGENT CONTINUATION ==='
  ].join('\n');
}

async function callArnaruRequest(arnaruBody: any, files: any[]): Promise<Response> {
  return files.length
    ? callArnaruChatWithFiles(arnaruBody, files)
    : callArnaruChat(arnaruBody);
}

async function readArnaruResponse(response: Response) {
  const rawText = await response.text();
  const parsed = parseSSE(rawText);

  if (parsed.fullMessage) {
    parsed.fullMessage = sanitizeContent(parsed.fullMessage);
  }

  return { rawText, parsed };
}

function makeDiagnostic(parsed: any): string {
  if (parsed?.errorMessage) {
    return `Arnaru returned an error: ${parsed.errorMessage}`;
  }
  return 'Arnaru returned an empty response after the tool execution.';
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed. Use POST.'
    });
  }

  const proxyApiKey = process.env.PROXY_API_KEY;
  const authHeader = req.headers.authorization;

  if (proxyApiKey && authHeader !== `Bearer ${proxyApiKey}`) {
    return res.status(401).json({
      error: 'Invalid or missing API key'
    });
  }

  try {
    const body = req.body as OpenAIChatRequest;

    if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'messages is required',
          type: 'invalid_request_error'
        }
      });
    }

    const model = validateModel(body.model || '');
    const requestId = generateId();

    const { arnaruBody, files } = await buildArnaruRequest({
      ...body,
      model
    });

    arnaruBody.systemPrompt = appendToolProtocol(
      arnaruBody.systemPrompt || '',
      body.tools
    );

    const continuation = hasToolResult(body);
    if (continuation) {
      arnaruBody.systemPrompt = appendContinuation(
        arnaruBody.systemPrompt || ''
      );
    }

    arnaruBody.systemPrompt = truncatePromptMiddle(
      arnaruBody.systemPrompt || '',
      Math.floor(MAX_UPSTREAM_PROMPT_CHARS * 0.4),
      0.35
    );

    arnaruBody.question = truncatePromptMiddle(
      arnaruBody.question || '',
      Math.floor(MAX_UPSTREAM_PROMPT_CHARS * 0.6),
      0.25
    );

    console.log(
      '[Arnaru proxy]',
      JSON.stringify({
        model,
        messageCount: body.messages.length,
        hasToolResult: continuation,
        hasAssistantToolCall: hasAssistantToolCall(body),
        toolCount: body.tools?.length || 0,
        hasFiles: files.length > 0
      })
    );

    let arnaruResponse = await callArnaruRequest(arnaruBody, files);

    if (!arnaruResponse.ok) {
      const errorText = await arnaruResponse.text();
      console.error('Arnaru API error:', arnaruResponse.status, errorText.slice(0, 1000));

      return res.status(arnaruResponse.status).json({
        error: {
          message: `Arnaru API error: ${arnaruResponse.status}`,
          type: 'api_error'
        }
      });
    }

    let { rawText, parsed } = await readArnaruResponse(arnaruResponse);
    let toolCalls = parseToolCalls(parsed.fullMessage || '', body.tools);

    if (toolCalls.length) {
      const completion = createCompletion(requestId, model, '', toolCalls);
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Access-Control-Allow-Origin', '*');
      return res.status(200).json(completion);
    }

    let retryAttempt = 0;

    const hasMalformedToolMarkup = (): boolean =>
      !!(body.tools && body.tools.length) &&
      !parseToolCalls(parsed.fullMessage || '', body.tools).length &&
      looksLikeToolCallMarkup(parsed.fullMessage || '');

    while (
      retryAttempt < MAX_EMPTY_RETRIES &&
      (isEmptyContent(parsed.fullMessage) || hasMalformedToolMarkup())
    ) {
      retryAttempt++;

      const contextTooLong = isContextTooLongError(parsed.errorMessage);
      const retryRatio = contextTooLong
        ? Math.max(0.35, 0.8 - retryAttempt * 0.2)
        : Math.max(0.5, 1 - retryAttempt * 0.2);

      const retryBudget = Math.max(
        MIN_UPSTREAM_PROMPT_CHARS,
        Math.floor(MAX_UPSTREAM_PROMPT_CHARS * retryRatio)
      );

      const malformedToolMarkup = hasMalformedToolMarkup();
      let recoverySystemPrompt = arnaruBody.systemPrompt || '';

      if (malformedToolMarkup) {
        recoverySystemPrompt = appendPromptInstruction(
          recoverySystemPrompt,
          '[STRICT INSTRUCTION] Your previous response contained incomplete tool-call markup. Keep arguments short and output ONLY strict JSON: {"tool_call":{"name":"<function>","arguments":{...}}}',
          retryBudget
        );
      } else {
        recoverySystemPrompt = [
          recoverySystemPrompt,
          '',
          '=== RECOVERY ===',
          'Your previous response was empty.',
          contextTooLong ? 'The previous context may have been too long.' : '',
          'A tool may already have completed; its real result is in the conversation above.',
          'Now provide the final answer to the original user.',
          'Return plain natural-language text.',
          'Do not call a tool unless absolutely necessary.',
          'Never return an empty response.',
          '=== END RECOVERY ==='
        ]
          .filter(Boolean)
          .join('\n');
      }

      recoverySystemPrompt = truncatePromptMiddle(
        recoverySystemPrompt,
        Math.floor(retryBudget * 0.4),
        0.35
      );

      const recoveryQuestion = truncatePromptMiddle(
        arnaruBody.question || '',
        Math.floor(retryBudget * 0.6),
        0.25
      );

      console.warn(
        '[Arnaru proxy] Recovery attempt',
        `${retryAttempt}/${MAX_EMPTY_RETRIES}`,
        malformedToolMarkup ? '(malformed tool markup)' : '(empty response)',
        `prompt=${recoveryQuestion.length}`
      );

      await new Promise(r => setTimeout(r, Math.min(500 * retryAttempt, 1500)));

      const retryBody = {
        ...arnaruBody,
        systemPrompt: recoverySystemPrompt,
        question: recoveryQuestion,
        conversationId: undefined
      };

      const retryResponse = await callArnaruRequest(retryBody, files);
      if (!retryResponse.ok) continue;

      const retryParsed = await readArnaruResponse(retryResponse);
      rawText = retryParsed.rawText;
      parsed = retryParsed.parsed;

      toolCalls = parseToolCalls(parsed.fullMessage || '', body.tools);

      if (toolCalls.length) {
        const completion = createCompletion(requestId, model, '', toolCalls);
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Access-Control-Allow-Origin', '*');
        return res.status(200).json(completion);
      }

      if (!isEmptyContent(parsed.fullMessage) && !hasMalformedToolMarkup()) {
        break;
      }
    }

    if (parsed.hasError && !parsed.fullMessage?.trim()) {
      return res.status(502).json({
        error: {
          message: parsed.errorMessage || 'Arnaru returned an upstream error',
          type: 'upstream_error'
        }
      });
    }

    if (body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('Access-Control-Allow-Origin', '*');

      res.write(createStreamChunk(requestId, model, { role: 'assistant' }));

      for (const chunk of parseSSEStream(rawText)) {
        if (chunk.error) {
          res.write(`data: ${JSON.stringify({ error: chunk.error })}\n\n`);
          continue;
        }

        if (chunk.text) {
          res.write(
            createStreamChunk(requestId, model, {
              content: sanitizeContent(chunk.text)
            })
          );
        }
      }

      res.write(createStreamChunk(requestId, model, {}, 'stop'));
      res.write('data: [DONE]\n\n');
      return res.end();
    }

    const finalContent = parsed.fullMessage?.trim() || '';

    if (!finalContent && continuation) {
      console.error('[Arnaru proxy] Final response remained empty after recovery.');
      return res.status(502).json({
        error: {
          message: makeDiagnostic(parsed),
          type: 'empty_upstream_response'
        }
      });
    }

    const completion = createCompletion(requestId, model, finalContent);
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(200).json(completion);

  } catch (error) {
    console.error('Proxy error:', error);
    return res.status(500).json({
      error: {
        message: error instanceof Error ? error.message : 'Internal server error',
        type: 'internal_error'
      }
    });
  }
}
