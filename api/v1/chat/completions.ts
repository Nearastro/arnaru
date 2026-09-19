import type {
  VercelRequest,
  VercelResponse
} from '@vercel/node';

import type {
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAITool
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
      sizeLimit: '4mb'
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
    : (
        process.env.DEFAULT_MODEL ||
        'claude-fable-5'
      );
}

/*
 * Models like Arnaru may emit something like:

{
  "name": "shell",
  "arguments": {
    "command": "ls -la"
  }
}

or:

{
  "tool": "shell",
  "command": "ls -la"
}

This function converts those text forms into
real OpenAI tool_calls.
*/

function extractToolCall(
  text: string,
  tools?: OpenAITool[]
) {
  if (!tools?.length) {
    return null;
  }

  const allowed = new Set(
    tools
      .filter(
        tool =>
          tool.type === 'function' &&
          tool.function?.name
      )
      .map(
        tool =>
          tool.function.name
      )
  );

  if (!allowed.size) {
    return null;
  }

  const candidates: string[] = [];

  /*
   * 1. Markdown JSON block.
   */
  const fenced = text.match(
    /```(?:json)?\s*([\s\S]*?)\s*```/i
  );

  if (fenced?.[1]) {
    candidates.push(fenced[1]);
  }

  /*
   * 2. Entire response.
   */
  candidates.push(text.trim());

  /*
   * 3. Extract first JSON object.
   */
  const firstBrace =
    text.indexOf('{');

  const lastBrace =
    text.lastIndexOf('}');

  if (
    firstBrace >= 0 &&
    lastBrace > firstBrace
  ) {
    candidates.push(
      text.slice(
        firstBrace,
        lastBrace + 1
      )
    );
  }

  for (const candidate of candidates) {
    try {
      const parsed =
        JSON.parse(candidate);

      if (
        !parsed ||
        typeof parsed !== 'object'
      ) {
        continue;
      }

      const name =
        typeof parsed.name === 'string'
          ? parsed.name
          : typeof parsed.tool === 'string'
            ? parsed.tool
            : null;

      if (
        !name ||
        !allowed.has(name)
      ) {
        continue;
      }

      let args: unknown =
        parsed.arguments;

      /*
       * Support models that emit:
       *
       * {"name":"shell","command":"ls"}
       *
       * instead of:
       *
       * {"name":"shell","arguments":{"command":"ls"}}
       */
      if (
        args === undefined
      ) {
        const copy = {
          ...parsed
        };

        delete copy.name;
        delete copy.tool;

        args = copy;
      }

      if (
        typeof args === 'string'
      ) {
        try {
          args = JSON.parse(args);
        } catch {
          args = {
            input: args
          };
        }
      }

      if (
        !args ||
        typeof args !== 'object'
      ) {
        args = {};
      }

      return {
        id:
          `call_${generateId()}`,
        type: 'function' as const,

        function: {
          name,
          arguments:
            JSON.stringify(args)
        }
      };
    } catch {
      // Try next candidate.
    }
  }

  return null;
}

function removeToolJson(
  text: string,
  toolCall: ReturnType<
    typeof extractToolCall
  >
): string {
  if (!toolCall) {
    return text;
  }

  /*
   * Don't expose the model's internal
   * JSON tool request to the user.
   */
  const fenced =
    text.replace(
      /```(?:json)?\s*[\s\S]*?\s*```/gi,
      ''
    );

  const firstBrace =
    fenced.indexOf('{');

  const lastBrace =
    fenced.lastIndexOf('}');

  if (
    firstBrace >= 0 &&
    lastBrace > firstBrace
  ) {
    return (
      fenced.slice(
        0,
        firstBrace
      ) +
      fenced.slice(
        lastBrace + 1
      )
    ).trim();
  }

  return fenced.trim();
}

function createCompletion(
  id: string,
  model: string,
  content: string,
  toolCalls?: any[]
) {
  const message: any = {
    role: 'assistant',
    content:
      toolCalls?.length
        ? null
        : content
  };

  if (toolCalls?.length) {
    message.tool_calls =
      toolCalls;
  }

  return {
    id,
    object: 'chat.completion',
    created: getTimestamp(),
    model,

    choices: [
      {
        index: 0,
        message,

        finish_reason:
          toolCalls?.length
            ? 'tool_calls'
            : 'stop'
      }
    ],

    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };
}

function streamChunk(
  id: string,
  model: string,
  delta: any,
  finishReason: string | null = null
) {
  return (
    `data: ${JSON.stringify({
      id,
      object:
        'chat.completion.chunk',
      created: getTimestamp(),
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
  if (req.method === 'OPTIONS') {
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

    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({
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
    return res.status(401).json({
      error:
        'Invalid or missing API key'
    });
  }

  try {
    const body =
      req.body as OpenAIChatRequest;

    if (
      !body ||
      !Array.isArray(body.messages) ||
      body.messages.length === 0
    ) {
      return res.status(400).json({
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

    /*
     * IMPORTANT:
     *
     * AnyClaw owns the tools.
     *
     * Arnaru only needs to return a
     * structured tool_call.
     */
    const tools =
      Array.isArray(body.tools)
        ? body.tools
        : [];

    /*
     * Tell Arnaru what tools are available.
     *
     * This is intentionally appended to
     * the system prompt because the
     * upstream Arnaru API currently
     * accepts "question", not native
     * OpenAI tool definitions.
     */
    let systemPrompt =
      body.systemPrompt || '';

    if (tools.length > 0) {
      const toolDescriptions =
        tools
          .filter(
            tool =>
              tool.type ===
                'function' &&
              tool.function?.name
          )
          .map(tool =>
            JSON.stringify(
              tool
            )
          )
          .join('\n');

      systemPrompt += `

You are connected to an external agent runtime.

The runtime has these tools:

${toolDescriptions}

When you need to use a tool, DO NOT explain the tool call.

Return ONLY one JSON object:

{
  "name": "exact_tool_name",
  "arguments": {
    "argument": "value"
  }
}

The "name" MUST exactly match one of the provided tools.

The "arguments" object MUST match that tool's parameters.

If no tool is needed, answer normally.

Never pretend that a tool was executed.
`;
    }

    const modifiedBody: OpenAIChatRequest = {
      ...body,
      model,
      systemPrompt
    };

    const {
      arnaruBody,
      files
    } =
      await buildArnaruRequest(
        modifiedBody
      );

    /*
     * IMPORTANT:
     *
     * If tools are present, inject the
     * tool schema into the question too.
     *
     * Arnaru's upstream endpoint receives
     * a flattened "question".
     */
    if (tools.length > 0) {
      arnaruBody.question = `
Available tools:

${tools
  .map(
    tool =>
      JSON.stringify(tool)
  )
  .join('\n')}

${arnaruBody.question}
`;
    }

    const arnaruResponse =
      files.length > 0
        ? await callArnaruChatWithFiles(
            arnaruBody,
            files
          )
        : await callArnaruChat(
            arnaruBody
          );

    if (!arnaruResponse.ok) {
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

    const parsed =
      parseSSE(rawText);

    if (
      parsed.hasError &&
      !parsed.fullMessage
    ) {
      return res.status(500).json({
        error: {
          message:
            parsed.errorMessage ||
            'Unknown Arnaru error',
          type:
            'api_error'
        }
      });
    }

    const fullText =
      parsed.fullMessage || '';

    /*
     * Detect model-generated tool JSON.
     */
    const toolCall =
      extractToolCall(
        fullText,
        tools
      );

    /*
     * ==========================
     * NATIVE TOOL CALL RESPONSE
     * ==========================
     */
    if (toolCall) {
      const cleanText =
        removeToolJson(
          fullText,
          toolCall
        );

      /*
       * Non-streaming.
       */
      if (!body.stream) {
        res.setHeader(
          'Content-Type',
          'application/json'
        );

        res.setHeader(
          'Access-Control-Allow-Origin',
          '*'
        );

        return res.status(200).json(
          createCompletion(
            requestId,
            model,
            cleanText,
            [toolCall]
          )
        );
      }

      /*
       * Streaming.
       *
       * Emit OpenAI-compatible
       * tool_calls delta.
       */
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
        streamChunk(
          requestId,
          model,
          {
            role: 'assistant'
          }
        )
      );

      res.write(
        streamChunk(
          requestId,
          model,
          {
            tool_calls: [
              {
                index: 0,
                id: toolCall.id,
                type: 'function',
                function: {
                  name:
                    toolCall.function.name,
                  arguments:
                    toolCall.function.arguments
                }
              }
            ]
          }
        )
      );

      res.write(
        streamChunk(
          requestId,
          model,
          {},
          'tool_calls'
        )
      );

      res.write(
        'data: [DONE]\n\n'
      );

      return res.end();
    }

    /*
     * ==========================
     * NORMAL RESPONSE
     * ==========================
     */

    if (body.stream) {
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
        streamChunk(
          requestId,
          model,
          {
            role: 'assistant'
          }
        )
      );

      for (
        const chunk of
        parseSSEStream(rawText)
      ) {
        if (chunk.error) {
          res.write(
            `data: ${JSON.stringify({
              error:
                chunk.error
            })}\n\n`
          );

          continue;
        }

        if (chunk.text) {
          res.write(
            streamChunk(
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
        streamChunk(
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

    res.setHeader(
      'Content-Type',
      'application/json'
    );

    res.setHeader(
      'Access-Control-Allow-Origin',
      '*'
    );

    return res.status(200).json(
      createCompletion(
        requestId,
        model,
        fullText
      )
    );

  } catch (error) {
    console.error(
      'Proxy error:',
      error
    );

    return res.status(500).json({
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
