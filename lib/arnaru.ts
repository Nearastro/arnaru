import {
  OpenAIMessage,
  OpenAIContentPart,
  ArnaruFileAttachment
} from './types';

const ARNARU_BASE_URL =
  process.env.ARNARU_BASE_URL ||
  'https://arnaru-ai.vercel.app';

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  'claude-fable-5';

const DEFAULT_WEB_SEARCH =
  process.env.DEFAULT_WEB_SEARCH === 'true';

const MAX_FILES = 9;

const MIME_EXT_MAP: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/jpg': 'jpg',
  'image/webp': 'webp',
  'image/gif': 'gif',
  'image/heic': 'heic',
  'application/pdf': 'pdf',
  'text/plain': 'txt',
  'text/csv': 'csv',
  'application/json': 'json',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
    'docx'
};

function guessExtension(
  mimeType: string
): string {
  return (
    MIME_EXT_MAP[mimeType] ||
    (mimeType.split('/')[1] || 'bin')
      .split('+')[0]
  );
}

function parseDataUri(
  uri: string
): {
  mimeType: string;
  buffer: Buffer;
} | null {
  const match = uri.match(
    /^data:([^;,]+);base64,(.+)$/s
  );

  if (!match) {
    return null;
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(
      match[2],
      'base64'
    )
  };
}

async function fetchAsBuffer(
  url: string
): Promise<{
  mimeType: string;
  buffer: Buffer;
} | null> {
  try {
    const resp = await fetch(url);

    if (!resp.ok) {
      return null;
    }

    const mimeType = (
      resp.headers.get(
        'content-type'
      ) ||
      'application/octet-stream'
    )
      .split(';')[0]
      .trim();

    const arrayBuf =
      await resp.arrayBuffer();

    return {
      mimeType,
      buffer: Buffer.from(arrayBuf)
    };
  } catch {
    return null;
  }
}

async function resolveAttachment(
  url: string,
  filenameHint?: string
): Promise<ArnaruFileAttachment | null> {
  if (!url) {
    return null;
  }

  let resolved:
    | {
        mimeType: string;
        buffer: Buffer;
      }
    | null = null;

  if (url.startsWith('data:')) {
    resolved = parseDataUri(url);
  } else if (
    /^https?:\/\//i.test(url)
  ) {
    resolved = await fetchAsBuffer(url);
  }

  if (
    !resolved ||
    resolved.buffer.length === 0
  ) {
    return null;
  }

  const ext = guessExtension(
    resolved.mimeType
  );

  const filename =
    filenameHint ||
    `upload_${Date.now()}_${Math.random()
      .toString(36)
      .slice(2, 8)}.${ext}`;

  return {
    buffer: resolved.buffer,
    filename,
    mimeType: resolved.mimeType
  };
}

/**
 * Convert assistant tool calls into deterministic
 * text that the Arnaru model can understand.
 */
function stringifyToolCalls(
  msg: OpenAIMessage
): string {
  if (
    !msg.tool_calls ||
    msg.tool_calls.length === 0
  ) {
    return '';
  }

  return msg.tool_calls
    .map(call => {
      const name =
        call.function?.name || 'unknown';

      const args =
        call.function?.arguments || '{}';

      return [
        `Assistant requested tool: ${name}`,
        `Tool call ID: ${call.id}`,
        `Arguments: ${args}`
      ].join('\n');
    })
    .join('\n\n');
}

/**
 * Extract OpenAI messages into the simple
 * question/systemPrompt contract expected by Arnaru.
 *
 * Important:
 * tool results are preserved and explicitly marked.
 * This is what allows the second model pass to
 * continue after AnyClaw executes a tool.
 */
export async function extractMessageContent(
  messages: OpenAIMessage[]
): Promise<{
  question: string;
  systemPrompt?: string;
  files: ArnaruFileAttachment[];
}> {
  let systemPrompt = '';

  const parts: string[] = [];

  const files: ArnaruFileAttachment[] = [];

  let hasToolResult = false;

  for (const msg of messages) {
    /*
     * SYSTEM
     */
    if (msg.role === 'system') {
      if (
        typeof msg.content === 'string'
      ) {
        if (msg.content.trim()) {
          systemPrompt =
            systemPrompt
              ? `${systemPrompt}\n\n${msg.content}`
              : msg.content;
        }
      } else {
        const textFragments: string[] = [];

        for (
          const part of
          (msg.content as OpenAIContentPart[]) ||
          []
        ) {
          if (
            part &&
            typeof part === 'object' &&
            part.type === 'text' &&
            part.text
          ) {
            textFragments.push(
              part.text
            );
          }
        }

        const text =
          textFragments.join('\n').trim();

        if (text) {
          systemPrompt =
            systemPrompt
              ? `${systemPrompt}\n\n${text}`
              : text;
        }
      }

      continue;
    }

    /*
     * TOOL RESULT
     *
     * This is critical for the second pass.
     */
    if (msg.role === 'tool') {
      hasToolResult = true;

      const result =
        typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(
              msg.content ?? ''
            );

      const callId =
        msg.tool_call_id ||
        'unknown';

      parts.push(
        [
          '=== TOOL RESULT ===',
          `Tool call ID: ${callId}`,
          'Result:',
          result,
          '=== END TOOL RESULT ==='
        ].join('\n')
      );

      continue;
    }

    /*
     * ASSISTANT MESSAGE WITH TOOL CALLS
     */
    if (
      msg.role === 'assistant' &&
      msg.tool_calls?.length
    ) {
      const text =
        typeof msg.content === 'string'
          ? msg.content.trim()
          : '';

      if (text) {
        parts.push(
          `Assistant: ${text}`
        );
      }

      const calls =
        stringifyToolCalls(msg);

      if (calls) {
        parts.push(calls);
      }

      continue;
    }

    /*
     * NORMAL STRING CONTENT
     */
    if (
      typeof msg.content === 'string'
    ) {
      const text =
        msg.content.trim();

      if (!text) {
        continue;
      }

      if (msg.role === 'user') {
        parts.push(
          `User: ${text}`
        );
      } else if (
        msg.role === 'assistant'
      ) {
        parts.push(
          `Assistant: ${text}`
        );
      }

      continue;
    }

    /*
     * MULTIMODAL CONTENT
     */
    const textFragments: string[] = [];

    for (
      const part of
      (msg.content as OpenAIContentPart[]) ||
      []
    ) {
      if (
        !part ||
        typeof part !== 'object'
      ) {
        continue;
      }

      if (
        part.type === 'text'
      ) {
        if (part.text) {
          textFragments.push(
            part.text
          );
        }
      } else if (
        part.type === 'image_url' &&
        msg.role === 'user'
      ) {
        if (
          files.length >= MAX_FILES
        ) {
          continue;
        }

        const attachment =
          await resolveAttachment(
            part.image_url?.url
          );

        if (attachment) {
          files.push(attachment);
        }
      } else if (
        part.type === 'file' &&
        msg.role === 'user'
      ) {
        if (
          files.length >= MAX_FILES
        ) {
          continue;
        }

        const fileData =
          part.file?.file_data;

        if (fileData) {
          const attachment =
            await resolveAttachment(
              fileData,
              part.file?.filename
            );

          if (attachment) {
            files.push(attachment);
          }
        }
      }
    }

    const text =
      textFragments
        .join('\n')
        .trim();

    if (!text) {
      continue;
    }

    if (msg.role === 'user') {
      parts.push(
        `User: ${text}`
      );
    } else if (
      msg.role === 'assistant'
    ) {
      parts.push(
        `Assistant: ${text}`
      );
    }
  }

  /*
   * When AnyClaw has just returned a tool result,
   * explicitly tell the model to continue.
   *
   * Without this, some upstream models stop after
   * the tool call and return an empty completion.
   */
  if (hasToolResult) {
    parts.push(
      [
        '=== AGENT CONTINUATION ===',
        'A previously requested tool has finished executing.',
        'The tool result above is real and must be used.',
        'Continue the original user task now.',
        'If the task is complete, respond to the user in natural language.',
        'Explain what happened and what was done.',
        'If another tool is required, request it using the available tool format.',
        'NEVER return an empty response.',
        'Do not merely repeat the tool result.',
        '=== END AGENT CONTINUATION ==='
      ].join('\n')
    );
  }

  const question =
    parts.length === 1 &&
    parts[0].startsWith('User: ')
      ? parts[0].slice(6)
      : parts.join('\n\n');

  return {
    question,
    systemPrompt:
      systemPrompt || undefined,
    files
  };
}

export async function buildArnaruRequest(
  body: any
): Promise<{
  arnaruBody: any;
  files: ArnaruFileAttachment[];
}> {
  const {
    question,
    systemPrompt,
    files
  } =
    await extractMessageContent(
      body.messages
    );

  const arnaruBody = {
    question,

    model:
      body.model ||
      DEFAULT_MODEL,

    conversationId:
      body.conversationId,

    webSearch:
      body.webSearch ??
      DEFAULT_WEB_SEARCH,

    systemPrompt:
      body.systemPrompt ||
      systemPrompt
  };

  return {
    arnaruBody,
    files
  };
}

export async function callArnaruChat(
  requestBody: any
): Promise<Response> {
  return fetch(
    `${ARNARU_BASE_URL}/api/chat`,
    {
      method: 'POST',

      headers: {
        'Content-Type':
          'application/json'
      },

      body:
        JSON.stringify(
          requestBody
        )
    }
  );
}

export async function callArnaruChatWithFiles(
  requestBody: any,
  files: ArnaruFileAttachment[]
): Promise<Response> {
  const formData =
    new FormData();

  formData.append(
    'question',
    requestBody.question ?? ''
  );

  if (requestBody.model) {
    formData.append(
      'model',
      requestBody.model
    );
  }

  if (
    requestBody.conversationId
  ) {
    formData.append(
      'conversationId',
      requestBody.conversationId
    );
  }

  if (
    requestBody.webSearch !==
    undefined
  ) {
    formData.append(
      'webSearch',
      String(
        requestBody.webSearch
      )
    );
  }

  if (
    requestBody.systemPrompt
  ) {
    formData.append(
      'systemPrompt',
      requestBody.systemPrompt
    );
  }

  for (
    const file of files.slice(
      0,
      MAX_FILES
    )
  ) {
    const blob =
      new Blob(
        [
          new Uint8Array(
            file.buffer
          )
        ],
        {
          type:
            file.mimeType
        }
      );

    formData.append(
      'files',
      blob,
      file.filename
    );
  }

  return fetch(
    `${ARNARU_BASE_URL}/api/chat`,
    {
      method: 'POST',
      body: formData
    }
  );
}

export function generateId(): string {
  return (
    'chatcmpl-' +
    Math.random()
      .toString(36)
      .substring(2, 15)
  );
}

export function getTimestamp(): number {
  return Math.floor(
    Date.now() / 1000
  );
}
