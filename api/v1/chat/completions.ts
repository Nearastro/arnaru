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
          x.type ===
            'function' &&
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
        results.push(
          cleaned
        );
      }
    }
  }

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

    if (
      inString
    ) {
      if (
        escaped
      ) {
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

function parseToolCall(
  text: string,
  tools?: OpenAITool[]
): OpenAIToolCall | null {
  const allowed =
    getToolNames(tools);

  if (!allowed.size) {
    return null;
  }

  /*
   * Supports:
   *
   * {"name":"shell","arguments":{...}}
   *
   * {"tool":"shell","arguments":{...}}
   *
   * {"command":"pwd"}
   * Called function shell
   */

  const called =
    text.match(
      /Called function\s+([A-Za-z0-9_.:-]+)/i
    );

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
        typeof obj !==
          'object'
      ) {
        continue;
      }

      let name:
        | string
        | null = null;

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

      if (
        !name &&
        called?.[1]
      ) {
        name =
          called[1];
      }

      if (
        !name ||
        !allowed.has(name)
      ) {
        continue;
      }

      let args =
        obj.arguments;

      if (
        args ===
        undefined
      ) {
        const copy = {
          ...obj
        };

        delete copy.name;
        delete copy.tool;

        args = copy;
      }

      if (
        typeof args ===
        'string'
      ) {
        try {
          args =
            JSON.parse(
              args
            );
        } catch {
          args = {
            input: args
          };
        }
      }

      if (
        args === null ||
        typeof args !==
          'object'
      ) {
        args = {};
      }

      return {
        id:
          `call_${generateId()}`,

        type:
          'function',

        function: {
          name,

          arguments:
            JSON.stringify(
              args
            )
        }
      };
    } catch {
      continue;
    }
  }

  /*
   * If model emitted:
   *
   * Called function shell
   *
   * but JSON parser couldn't find
   * a valid object, don't fabricate
   * arguments.
   */
  return null;
}

function createCompletion(
  id: string,
  model: string,
  content: string,
  toolCall?: OpenAIToolCall | null
) {
  if (toolCall) {
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
            role: 'assistant',
            content: null,
            tool_calls: [
              toolCall
            ]
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
          role: 'assistant',
          content
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
     * Tell Arnaru exactly what tools
     * are available.
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

IMPORTANT:
When you need to use a tool, output ONLY a JSON object in this exact form:

{"name":"TOOL_NAME","arguments":{"argument":"value"}}

TOOL_NAME must be one of the available tools.

Do not say "I ran the tool".
Do not invent tool results.
The external agent will execute the tool and send the result back to you.
`;

      arnaruBody.systemPrompt =
        (
          arnaruBody.systemPrompt ||
          ''
        ) +
        toolInstruction;
    }

    const arnaruResponse =
      files.length
        ? await callArnaruChatWithFiles(
            arnaruBody,
            files
          )
        : await callArnaruChat(
            arnaruBody
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

    const rawText =
      await arnaruResponse.text();

    /*
     * We intentionally parse the complete
     * upstream response before returning
     * because tool_calls need to be atomic.
     */
    const parsed =
      parseSSE(rawText);

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

    const toolCall =
      parseToolCall(
        parsed.fullMessage,
        body.tools
      );

    /*
     * NATIVE TOOL CALL
     */
    if (
      toolCall
    ) {
      const completion =
        createCompletion(
          requestId,
          model,
          '',
          toolCall
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

    const completion =
      createCompletion(
        requestId,
        model,
        parsed.fullMessage
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
