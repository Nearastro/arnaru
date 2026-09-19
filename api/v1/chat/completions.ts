import type {
  VercelRequest,
  VercelResponse
} from '@vercel/node';

import type {
  OpenAIChatRequest,
  OpenAITool,
  OpenAIToolCall
} from '../../../lib/types';

import {
  parseSSE,
  parseSSEStream
} from '../../../lib/sse-parser';

import {
  buildArnaruRequest,
  callArnaruChat,
  callArnaruChatWithFiles,
  generateId,
  getTimestamp
} from '../../../lib/arnaru';

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

function validateModel(
  model: string
): string {
  return VALID_MODELS.has(model)
    ? model
    : (
        process.env.DEFAULT_MODEL ||
        'claude-fable-5'
      );
}

function getToolNames(
  tools?: OpenAITool[]
): Set<string> {
  return new Set(
    (tools || [])
      .filter(
        x =>
          x.type === 'function' &&
          !!x.function?.name
      )
      .map(
        x =>
          x.function.name
      )
  );
}

function findJsonObjects(
  text: string
): string[] {
  const results: string[] = [];

  /*
   * Fenced JSON.
   */
  const fenced =
    text.match(
      /```(?:json)?\s*([\s\S]*?)\s*```/gi
    );

  if (fenced) {
    for (
      const block of fenced
    ) {
      const cleaned =
        block
          .replace(
            /^```(?:json)?/i,
            ''
          )
          .replace(
            /```$/i,
            ''
          )
          .trim();

      if (cleaned) {
        results.push(cleaned);
      }
    }
  }

  /*
   * Balanced JSON objects.
   */
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (
    let i = 0;
    i < text.length;
    i++
  ) {
    const c = text[i];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (
        c === '\\'
      ) {
        escaped = true;
      } else if (
        c === '"'
      ) {
        inString = false;
      }

      continue;
    }

    if (c === '"') {
      inString = true;
      continue;
    }

    if (c === '{') {
      if (depth === 0) {
        start = i;
      }

      depth++;
    }

    if (c === '}') {
      depth--;

      if (
        depth === 0 &&
        start >= 0
      ) {
        results.push(
          text.slice(
            start,
            i + 1
          )
        );

        start = -1;
      }
    }
  }

  return results;
}

function normalizeArguments(
  value: any
): Record<string, any> {
  if (
    typeof value === 'string'
  ) {
    try {
      const parsed =
        JSON.parse(value);

      if (
        parsed &&
        typeof parsed === 'object'
      ) {
        return parsed;
      }
    } catch {
      return {
        input: value
      };
    }
  }

  if (
    value &&
    typeof value === 'object'
  ) {
    return value;
  }

  return {};
}

function makeToolCall(
  name: string,
  args: any
): OpenAIToolCall {
  return {
    id:
      `call_${generateId()}`,

    type:
      'function',

    function: {
      name,

      arguments:
        JSON.stringify(
          normalizeArguments(args)
        )
    }
  };
}

/**
 * Parse the various tool-call formats that
 * an upstream model may emit.
 *
 * Supported:
 *
 * {"name":"shell","arguments":{...}}
 * {"tool":"shell","arguments":{...}}
 * {"tool_call":{...}}
 * {"function_call":{...}}
 * {"tool_calls":[...]}
 * <tool_call>{...}</tool_call>
 * TOOL_CALL: {...}
 * Called function shell + JSON
 */
function parseToolCalls(
  text: string,
  tools?: OpenAITool[]
): OpenAIToolCall[] {
  const allowed =
    getToolNames(tools);

  if (!allowed.size) {
    return [];
  }

  const calls:
    OpenAIToolCall[] = [];

  const seen =
    new Set<string>();

  function add(
    name: any,
    args: any
  ) {
    if (
      typeof name !== 'string' ||
      !allowed.has(name)
    ) {
      return;
    }

    const normalized =
      normalizeArguments(args);

    const key =
      `${name}:${JSON.stringify(normalized)}`;

    if (seen.has(key)) {
      return;
    }

    seen.add(key);

    calls.push(
      makeToolCall(
        name,
        normalized
      )
    );
  }

  /*
   * Explicit <tool_call>...</tool_call>
   */
  const xmlMatches =
    text.match(
      /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi
    );

  if (xmlMatches) {
    for (
      const block of xmlMatches
    ) {
      const cleaned =
        block
          .replace(
            /^<tool_call>\s*/i,
            ''
          )
          .replace(
            /\s*<\/tool_call>$/i,
            ''
          )
          .trim();

      try {
        const obj =
          JSON.parse(cleaned);

        if (
          obj?.name
        ) {
          add(
            obj.name,
            obj.arguments ??
              obj.parameters ??
              {}
          );
        }

        if (
          Array.isArray(
            obj?.tool_calls
          )
        ) {
          for (
            const call of
            obj.tool_calls
          ) {
            add(
              call?.function?.name ||
                call?.name ||
                call?.tool,
              call?.function?.arguments ??
                call?.arguments ??
                {}
            );
          }
        }
      } catch {
        // Continue with other parsers.
      }
    }
  }

  /*
   * Called function NAME
   */
  const calledMatches =
    [
      ...text.matchAll(
        /Called function\s+([A-Za-z0-9_.:-]+)/gi
      )
    ];

  /*
   * JSON objects.
   */
  const jsons =
    findJsonObjects(text);

  for (
    const raw of jsons
  ) {
    try {
      const obj =
        JSON.parse(raw);

      if (
        !obj ||
        typeof obj !== 'object'
      ) {
        continue;
      }

      /*
       * OpenAI-style:
       *
       * {
       *   "tool_calls": [...]
       * }
       */
      if (
        Array.isArray(
          obj.tool_calls
        )
      ) {
        for (
          const call of
          obj.tool_calls
        ) {
          add(
            call?.function?.name ||
              call?.name ||
              call?.tool,
            call?.function?.arguments ??
              call?.arguments ??
              {}
          );
        }

        continue;
      }

      /*
       * Envelope:
       *
       * {"tool_call": {...}}
       */
      if (
        obj.tool_call &&
        typeof obj.tool_call ===
          'object'
      ) {
        const call =
          obj.tool_call;

        add(
          call.name ||
            call.tool ||
            call.function?.name,

          call.arguments ??
            call.parameters ??
            call.function?.arguments ??
            {}
        );

        continue;
      }

      /*
       * Envelope:
       *
       * {"function_call": {...}}
       */
      if (
        obj.function_call &&
        typeof obj.function_call ===
          'object'
      ) {
        const call =
          obj.function_call;

        add(
          call.name,

          call.arguments ??
            {}
        );

        continue;
      }

      /*
       * Direct:
       *
       * {"name":"shell","arguments":{}}
       *
       * {"tool":"shell","arguments":{}}
       */
      let name:
        | string
        | undefined;

      if (
        typeof obj.name ===
        'string'
      ) {
        name = obj.name;
      }

      if (
        typeof obj.tool ===
        'string'
      ) {
        name = obj.tool;
      }

      /*
       * If this is:
       *
       * Called function shell
       *
       * followed by JSON args.
       */
      if (
        !name &&
        calledMatches.length
      ) {
        name =
          calledMatches[0][1];
      }

      if (name) {
        let args =
          obj.arguments;

        if (
          args === undefined
        ) {
          args =
            obj.parameters;
        }

        if (
          args === undefined
        ) {
          const copy = {
            ...obj
          };

          delete copy.name;
          delete copy.tool;

          args = copy;
        }

        add(
          name,
          args
        );
      }
    } catch {
      continue;
    }
  }

  /*
   * TOOL_CALL: prefix.
   */
  const prefixMatches =
    [
      ...text.matchAll(
        /TOOL_CALL\s*:\s*([\s\S]+)/gi
      )
    ];

  for (
    const match of
    prefixMatches
  ) {
    const payload =
      match[1].trim();

    try {
      const obj =
        JSON.parse(payload);

      add(
        obj?.name ||
          obj?.tool ||
          obj?.function?.name,

        obj?.arguments ??
          obj?.parameters ??
          obj?.function?.arguments ??
          {}
      );
    } catch {
      // JSON parser above may already
      // have handled it.
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
  if (
    toolCalls.length
  ) {
    return {
      id,

      object:
        'chat.completion',

      created:
        getTimestamp(),

      model,

      choices: [
        {
          index: 0,

          message: {
            role:
              'assistant',

            content:
              null,

            tool_calls:
              toolCalls
          },

          finish_reason:
            'tool_calls'
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

    object:
      'chat.completion',

    created:
      getTimestamp(),

    model,

    choices: [
      {
        index: 0,

        message: {
          role:
            'assistant',

          content:
            content || ''
        },

        finish_reason:
          'stop'
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
  finishReason:
    | string
    | null = null
) {
  return (
    `data: ${JSON.stringify({
      id,

      object:
        'chat.completion.chunk',

      created:
        getTimestamp(),

      model,

      choices: [
        {
          index: 0,

          delta,

          finish_reason:
            finishReason
        }
      ]
    })}\n\n`
  );
}

function hasToolResult(
  body: OpenAIChatRequest
): boolean {
  return body.messages.some(
    (msg: any) =>
      msg?.role === 'tool'
  );
}

function addContinuationPrompt(
  requestBody: any
) {
  const continuation = `

=== FINAL AGENT CONTINUATION ===

A tool has already been executed by the external agent.

The tool result is included in the conversation.

Continue the user's ORIGINAL task now.

Rules:
1. Use the tool result as factual context.
2. Do not pretend to execute another tool yourself.
3. If the task is complete, answer the user normally.
4. Clearly explain what was completed.
5. If another tool is genuinely required, request it using the available tool-call JSON format.
6. NEVER return an empty response.
7. NEVER respond with only whitespace.
8. NEVER say that you are waiting for a tool when the tool result is already present.

=== END FINAL AGENT CONTINUATION ===
`;

  requestBody.systemPrompt =
    (
      requestBody.systemPrompt ||
      ''
    ) +
    continuation;
}

async function callArnaruRequest(
  arnaruBody: any,
  files: any[]
): Promise<Response> {
  return files.length
    ? callArnaruChatWithFiles(
        arnaruBody,
        files
      )
    : callArnaruChat(
        arnaruBody
      );
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  if (
    req.method ===
    'OPTIONS'
  ) {
    res.setHeader(
      'Access-Control-Allow-Origin',
      '*'
    );

    res.setHeader(
      'Access-Control-Allow-Methods',
      'POST, OPTIONS'
    );

    res.setHeader(
      'Access-Control-Allow-Headers',
      'Content-Type, Authorization'
    );

    return res
      .status(200)
      .end();
  }

  if (
    req.method !==
    'POST'
  ) {
    return res
      .status(405)
      .json({
        error:
          'Method not allowed. Use POST.'
      });
  }

  const proxyApiKey =
    process.env.PROXY_API_KEY;

  const authHeader =
    req.headers.authorization;

  if (
    proxyApiKey &&
    authHeader !==
      `Bearer ${proxyApiKey}`
  ) {
    return res
      .status(401)
      .json({
        error:
          'Invalid or missing API key'
      });
  }

  try {
    const body =
      req.body as OpenAIChatRequest;

    if (
      !body ||
      !Array.isArray(
        body.messages
      ) ||
      body.messages.length === 0
    ) {
      return res
        .status(400)
        .json({
          error: {
            message:
              'messages is required',

            type:
              'invalid_request_error'
          }
        });
    }

    const model =
      validateModel(
        body.model || ''
      );

    const requestId =
      generateId();

    const {
      arnaruBody,
      files
    } =
      await buildArnaruRequest({
        ...body,
        model
      });

    /*
     * Tell Arnaru about available tools.
     */
    if (
      body.tools &&
      body.tools.length
    ) {
      const toolList =
        body.tools
          .filter(
            x =>
              x.type ===
                'function' &&
              x.function?.name
          )
          .map(
            x =>
              JSON.stringify(
                x
              )
          )
          .join('\n');

      const toolInstruction = `

AVAILABLE AGENT TOOLS:

${toolList}

TOOL CALL PROTOCOL:

When you need to use a tool, output ONLY:

{"name":"TOOL_NAME","arguments":{"argument":"value"}}

TOOL_NAME must exactly match one of the available tools.

You may request another tool after a previous tool result.

Do not fabricate tool results.

The external agent executes tools and sends their results back.

When a tool result is already present in the conversation, CONTINUE THE ORIGINAL TASK and produce a normal final answer unless another tool is actually required.

NEVER output an empty response.
`;

      arnaruBody.systemPrompt =
        (
          arnaruBody.systemPrompt ||
          ''
        ) +
        toolInstruction;
    }

    /*
     * Explicit continuation when AnyClaw
     * has already executed a tool.
     */
    const continuation =
      hasToolResult(body);

    if (continuation) {
      addContinuationPrompt(
        arnaruBody
      );
    }

    /*
     * First upstream request.
     */
    let arnaruResponse =
      await callArnaruRequest(
        arnaruBody,
        files
      );

    if (
      !arnaruResponse.ok
    ) {
      const errorText =
        await arnaruResponse.text();

      console.error(
        'Arnaru API error:',
        arnaruResponse.status,
        errorText
      );

      return res
        .status(
          arnaruResponse.status
        )
        .json({
          error: {
            message:
              `Arnaru API error: ${arnaruResponse.status}`,

            type:
              'api_error'
          }
        });
    }

    let rawText =
      await arnaruResponse.text();

    let parsed =
      parseSSE(rawText);

    /*
     * If upstream returned an error with no content.
     */
    if (
      parsed.hasError &&
      !parsed.fullMessage
    ) {
      return res
        .status(500)
        .json({
          error: {
            message:
              parsed.errorMessage ||
              'Unknown Arnaru error',

            type:
              'api_error'
          }
        });
    }

    /*
     * Parse native/fallback tool calls.
     */
    let toolCalls =
      parseToolCalls(
        parsed.fullMessage,
        body.tools
      );

    /*
     * Return tool calls immediately.
     */
    if (
      toolCalls.length
    ) {
      const completion =
        createCompletion(
          requestId,
          model,
          '',
          toolCalls
        );

      res.setHeader(
        'Content-Type',
        'application/json'
      );

      res.setHeader(
        'Access-Control-Allow-Origin',
        '*'
      );

      return res
        .status(200)
        .json(completion);
    }

    /*
     * CRITICAL EMPTY-RESPONSE RECOVERY
     *
     * If AnyClaw has already returned a tool result
     * but Arnaru gives us an empty response, ask Arnaru
     * one more time with an explicit final-answer prompt.
     *
     * This is intentionally limited to ONE retry.
     */
    if (
      continuation &&
      !parsed.fullMessage?.trim()
    ) {
      console.warn(
        'Arnaru returned empty after tool result; retrying continuation'
      );

      const retryBody = {
        ...arnaruBody,

        systemPrompt:
          `${arnaruBody.systemPrompt || ''}

FINAL RESPONSE RETRY:

Your previous response was empty.

The tool has already completed.

You MUST now respond to the user.

Do not output an empty message.

If the tool succeeded, explain the result and what you completed.

If the tool failed, explain the failure clearly.

Return plain natural-language assistant text.`
      };

      arnaruResponse =
        await callArnaruRequest(
          retryBody,
          files
        );

      if (
        arnaruResponse.ok
      ) {
        rawText =
          await arnaruResponse.text();

        parsed =
          parseSSE(rawText);

        toolCalls =
          parseToolCalls(
            parsed.fullMessage,
            body.tools
          );

        if (
          toolCalls.length
        ) {
          const completion =
            createCompletion(
              requestId,
              model,
              '',
              toolCalls
            );

          res.setHeader(
            'Content-Type',
            'application/json'
          );

          res.setHeader(
            'Access-Control-Allow-Origin',
            '*'
          );

          return res
            .status(200)
            .json(completion);
        }
      }
    }

    /*
     * NORMAL STREAM
     */
    if (
      body.stream
    ) {
      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.setHeader(
        'Access-Control-Allow-Origin',
        '*'
      );

      res.write(
        createStreamChunk(
          requestId,
          model,
          {
            role:
              'assistant'
          }
        )
      );

      for (
        const chunk of
        parseSSEStream(
          rawText
        )
      ) {
        if (
          chunk.error
        ) {
          res.write(
            `data: ${JSON.stringify({
              error:
                chunk.error
            })}\n\n`
          );

          continue;
        }

        if (
          chunk.text
        ) {
          res.write(
            createStreamChunk(
              requestId,
              model,
              {
                content:
                  chunk.text
              }
            )
          );
        }
      }

      res.write(
        createStreamChunk(
          requestId,
          model,
          {},
          'stop'
        )
      );

      res.write(
        'data: [DONE]\n\n'
      );

      return res.end();
    }

    const finalContent =
      parsed.fullMessage?.trim() ||
      '';

    /*
     * We do NOT fabricate a fake model answer.
     *
     * If the upstream still gives empty after
     * the recovery attempt, expose a diagnostic
     * instead of silently returning "".
     */
    const safeContent =
      finalContent ||
      (
        continuation
          ? 'The tool completed, but the model did not return a final response.'
          : ''
      );

    const completion =
      createCompletion(
        requestId,
        model,
        safeContent
      );

    res.setHeader(
      'Content-Type',
      'application/json'
    );

    res.setHeader(
      'Access-Control-Allow-Origin',
      '*'
    );

    return res
      .status(200)
      .json(completion);

  } catch (error) {
    console.error(
      'Proxy error:',
      error
    );

    return res
      .status(500)
      .json({
        error: {
          message:
            error instanceof Error
              ? error.message
              : 'Internal server error',

          type:
            'internal_error'
        }
      });
  }
}
