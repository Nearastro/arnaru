import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  buildArnaruRequest,
  callArnaruChat,
  callArnaruChatWithFiles,
  hasToolCall,
  hasToolResult,
  updateArnaruSession,
  markArnaruFailure,
  type ArnaruMessage,
} from "@/lib/arnaru";

/* =========================================================
 * Models
 * ========================================================= */

const VALID_MODELS = new Set([
  "claude-fable-5",
  "claude-fable",
  "gpt-5",
  "gpt-5-mini",
  "gpt-5.1",
  "gpt-5.2",
  "gpt-5.3",
  "gpt-5.4",
  "gemini",
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "deepseek",
]);

/* =========================================================
 * Types
 * ========================================================= */

type ChatMessage = ArnaruMessage & {
  role:
    | "system"
    | "user"
    | "assistant"
    | "tool";

  content?: any;

  tool_calls?: any[];

  tool_call_id?: string;
};

type ChatBody = {
  model?: string;

  messages?: ChatMessage[];

  tools?: any[];

  tool_choice?: any;

  stream?: boolean;

  temperature?: number;

  max_tokens?: number;

  conversationId?: string;

  webSearch?: boolean;

  user?: string;

  session_id?: string;
};

/* =========================================================
 * Utility
 * ========================================================= */

function json(
  data: unknown,
  status = 200,
  headers?: HeadersInit
) {
  return NextResponse.json(
    data,
    {
      status,
      headers,
    }
  );
}

function generateId() {
  return `chatcmpl-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

function generateToolId() {
  return `call_${Date.now()}_${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

function safeString(
  value: unknown
): string {
  if (
    value === undefined ||
    value === null
  ) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeArguments(
  value: unknown
): string {
  if (
    typeof value === "string"
  ) {
    const trimmed =
      value.trim();

    if (!trimmed) {
      return "{}";
    }

    try {
      JSON.parse(trimmed);
      return trimmed;
    } catch {
      return JSON.stringify({
        input: trimmed,
      });
    }
  }

  if (
    value &&
    typeof value === "object"
  ) {
    try {
      return JSON.stringify(
        value
      );
    } catch {
      return "{}";
    }
  }

  return "{}";
}

/* =========================================================
 * Tool parsing
 * ========================================================= */

function findJsonObjects(
  text: string
): any[] {
  const results: any[] = [];

  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (
    let i = 0;
    i < text.length;
    i++
  ) {
    const char =
      text[i];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (
      char === "\\" &&
      inString
    ) {
      escaped = true;
      continue;
    }

    if (char === '"') {
      inString =
        !inString;
      continue;
    }

    if (inString) {
      continue;
    }

    if (char === "{") {
      if (depth === 0) {
        start = i;
      }

      depth++;
    }

    if (char === "}") {
      depth--;

      if (
        depth === 0 &&
        start >= 0
      ) {
        const candidate =
          text.slice(
            start,
            i + 1
          );

        try {
          results.push(
            JSON.parse(
              candidate
            )
          );
        } catch {}

        start = -1;
      }
    }
  }

  return results;
}

function makeToolCall(
  raw: any
) {
  const functionData =
    raw?.function || raw;

  const name =
    functionData?.name ||
    raw?.name ||
    raw?.tool;

  if (
    typeof name !== "string" ||
    !name.trim()
  ) {
    return null;
  }

  const args =
    functionData?.arguments ??
    functionData?.parameters ??
    raw?.arguments ??
    raw?.parameters ??
    {};

  return {
    id:
      raw?.id ||
      raw?.tool_call_id ||
      generateToolId(),

    type: "function",

    function: {
      name:
        name.trim(),

      arguments:
        normalizeArguments(
          args
        ),
    },
  };
}

function parseToolCalls(
  text: string
): any[] {
  if (!text?.trim()) {
    return [];
  }

  const calls: any[] = [];

  /*
   * 1. <tool_call>...</tool_call>
   */
  const tagRegex =
    /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi;

  let match: RegExpExecArray | null;

  while (
    (match =
      tagRegex.exec(text))
  ) {
    const objects =
      findJsonObjects(
        match[1]
      );

    for (const object of objects) {
      const call =
        makeToolCall(object);

      if (call) {
        calls.push(call);
      }
    }
  }

  /*
   * 2. TOOL_CALL:
   */
  const toolCallRegex =
    /TOOL_CALL\s*:\s*([\s\S]+)/gi;

  while (
    (match =
      toolCallRegex.exec(text))
  ) {
    const objects =
      findJsonObjects(
        match[1]
      );

    for (const object of objects) {
      const source =
        object?.tool_call ||
        object?.function_call ||
        object;

      const call =
        makeToolCall(source);

      if (call) {
        calls.push(call);
      }
    }
  }

  /*
   * 3. JSON envelope
   */
  const objects =
    findJsonObjects(text);

  for (const object of objects) {
    if (
      Array.isArray(
        object?.tool_calls
      )
    ) {
      for (
        const raw of
        object.tool_calls
      ) {
        const call =
          makeToolCall(raw);

        if (call) {
          calls.push(call);
        }
      }
    }

    if (
      object?.tool_call
    ) {
      const call =
        makeToolCall(
          object.tool_call
        );

      if (call) {
        calls.push(call);
      }
    }

    if (
      object?.function_call
    ) {
      const call =
        makeToolCall(
          object.function_call
        );

      if (call) {
        calls.push(call);
      }
    }

    /*
     * Direct:
     * {"name":"shell","arguments":...}
     */
    if (
      object?.name &&
      (
        object?.arguments !==
          undefined ||
        object?.parameters !==
          undefined
      )
    ) {
      const call =
        makeToolCall(
          object
        );

      if (call) {
        calls.push(call);
      }
    }
  }

  /*
   * 4. Called function shell
   */
  const calledRegex =
    /Called function\s+([A-Za-z0-9_.:-]+)[\s\S]*?(\{[\s\S]*\})/gi;

  while (
    (match =
      calledRegex.exec(text))
  ) {
    const name =
      match[1];

    const objects =
      findJsonObjects(
        match[2]
      );

    for (const object of objects) {
      const call =
        makeToolCall({
          name,
          arguments: object,
        });

      if (call) {
        calls.push(call);
      }
    }
  }

  /*
   * Deduplicate.
   */
  const unique =
    new Map<string, any>();

  for (const call of calls) {
    const key =
      [
        call.function.name,
        call.function.arguments,
      ].join(":");

    if (!unique.has(key)) {
      unique.set(
        key,
        call
      );
    }
  }

  return [...unique.values()];
}

/* =========================================================
 * Tool prompt
 * ========================================================= */

function appendToolProtocol(
  systemPrompt: string,
  tools: any[]
): string {
  if (
    !Array.isArray(tools) ||
    !tools.length
  ) {
    return systemPrompt;
  }

  const definitions =
    tools
      .map((tool) => {
        const fn =
          tool?.function ||
          tool;

        return JSON.stringify({
          name:
            fn?.name,
          description:
            fn?.description,
          parameters:
            fn?.parameters,
        });
      })
      .join("\n");

  return [
    systemPrompt,

    "=== AVAILABLE TOOLS ===",
    definitions,
    "=== END AVAILABLE TOOLS ===",

    "When you need an external tool, emit ONLY a tool call.",
    "Use this format:",
    '{"tool_calls":[{"id":"call_x","type":"function","function":{"name":"tool_name","arguments":"{\\"key\\":\\"value\\"}"}}]}',
    "",
    "Do not pretend a tool was executed.",
    "Wait for the tool result.",
  ]
    .filter(Boolean)
    .join("\n");
}

/* =========================================================
 * Continuation
 * ========================================================= */

function appendContinuation(
  systemPrompt: string
): string {
  return [
    systemPrompt,

    "=== TOOL CONTINUATION ===",
    "A tool requested by the assistant has completed.",
    "The conversation contains the assistant tool call and the real tool result.",
    "Continue the ORIGINAL user request.",
    "Interpret the tool result.",
    "Do not output an empty response.",
    "Do not repeat the tool call unless genuinely necessary.",
    "Return the final answer directly to the user.",
    "=== END TOOL CONTINUATION ===",
  ]
    .filter(Boolean)
    .join("\n");
}

/* =========================================================
 * OpenAI completion object
 * ========================================================= */

function createCompletion(
  model: string,
  text: string,
  toolCalls: any[],
  id: string
) {
  const created =
    Math.floor(
      Date.now() / 1000
    );

  if (
    toolCalls.length
  ) {
    return {
      id,
      object:
        "chat.completion",
      created,
      model,

      choices: [
        {
          index: 0,

          message: {
            role:
              "assistant",

            content:
              null,

            tool_calls:
              toolCalls,
          },

          finish_reason:
            "tool_calls",
        },
      ],
    };
  }

  return {
    id,

    object:
      "chat.completion",

    created,

    model,

    choices: [
      {
        index: 0,

        message: {
          role:
            "assistant",

          content:
            text,
        },

        finish_reason:
          "stop",
      },
    ],
  };
}

/* =========================================================
 * Empty response
 * ========================================================= */

function isEmpty(
  text: string
) {
  return !text ||
    !text.trim();
}

/* =========================================================
 * Session key
 * ========================================================= */

function getSessionKey(
  request: NextRequest,
  body: ChatBody
): string {
  const header =
    request.headers.get(
      "x-agent-session"
    ) ||
    request.headers.get(
      "x-session-id"
    );

  if (header?.trim()) {
    return header.trim();
  }

  if (
    body.session_id?.trim()
  ) {
    return body.session_id;
  }

  if (
    body.user?.trim()
  ) {
    return `user:${body.user.trim()}`;
  }

  if (
    body.conversationId?.trim()
  ) {
    return `conversation:${body.conversationId.trim()}`;
  }

  /*
   * IMPORTANT:
   *
   * Don't create a random session on every
   * tool continuation if AnyClaw gives no
   * explicit session.
   *
   * This request-level fallback remains stable
   * through the client conversation when
   * conversationId exists.
   */
  return "default";
}

/* =========================================================
 * API key
 * ========================================================= */

function isAuthorized(
  request: NextRequest
) {
  const expected =
    process.env.API_KEY ||
    process.env.OPENAI_API_KEY;

  if (!expected) {
    return true;
  }

  const authorization =
    request.headers.get(
      "authorization"
    );

  if (!authorization) {
    return false;
  }

  return (
    authorization ===
    `Bearer ${expected}`
  );
}

/* =========================================================
 * CORS
 * ========================================================= */

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin":
      "*",

    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, X-Agent-Session, X-Session-ID",

    "Access-Control-Allow-Methods":
      "POST, OPTIONS",
  };
}

/* =========================================================
 * Request
 * ========================================================= */

export async function POST(
  request: NextRequest
) {
  const headers =
    corsHeaders();

  if (
    !isAuthorized(request)
  ) {
    return json(
      {
        error: {
          message:
            "Unauthorized",
          type:
            "invalid_api_key",
        },
      },
      401,
      headers
    );
  }

  let body: ChatBody;

  try {
    body =
      await request.json();
  } catch {
    return json(
      {
        error: {
          message:
            "Invalid JSON body",
          type:
            "invalid_request_error",
        },
      },
      400,
      headers
    );
  }

  if (
    !Array.isArray(
      body.messages
    ) ||
    body.messages.length === 0
  ) {
    return json(
      {
        error: {
          message:
            "messages is required",
          type:
            "invalid_request_error",
        },
      },
      400,
      headers
    );
  }

  const model =
    body.model &&
    VALID_MODELS.has(
      body.model
    )
      ? body.model
      : "claude-fable-5";

  const messages =
    body.messages;

  const toolResultExists =
    hasToolResult(messages);

  const sessionId =
    getSessionKey(
      request,
      body
    );

  /*
   * IMPORTANT:
   *
   * Don't mutate the original body messages.
   */
  let systemPrompt =
    messages
      .filter(
        (message) =>
          message.role ===
          "system"
      )
      .map(
        (message) =>
          typeof message.content ===
          "string"
            ? message.content
            : safeString(
                message.content
              )
      )
      .join("\n\n");

  systemPrompt =
    appendToolProtocol(
      systemPrompt,
      body.tools || []
    );

  if (
    toolResultExists
  ) {
    systemPrompt =
      appendContinuation(
        systemPrompt
      );
  }

  const arnaruRequest =
    buildArnaruRequest(
      messages,
      {
        model,
        conversationId:
          body.conversationId,
        webSearch:
          body.webSearch,
        systemPrompt,
        sessionId,
      }
    );

  /*
   * Tool result means this is the SECOND+
   * turn of an agent execution.
   */
  const isContinuation =
    toolResultExists;

  /*
   * Empty response retries.
   *
   * Similar concept to FreeDeepseekAPI.
   */
  const maxRetries = isContinuation
    ? 2
    : 1;

  let lastError:
    | unknown
    | undefined;

  let finalText = "";

  let finalResponse:
    | Awaited<
        ReturnType<
          typeof callArnaruChat
        >
      >
    | undefined;

  let finalRequest =
    arnaruRequest;

  const files =
    messages
      .flatMap(
        (message: any) => {
          const content =
            message.content;

          if (
            !Array.isArray(content)
          ) {
            return [];
          }

          return content;
        }
      )
      .filter(
        (part: any) =>
          part?.type === "file" ||
          part?.type ===
            "input_file"
      );

  for (
    let attempt = 0;
    attempt <= maxRetries;
    attempt++
  ) {
    try {
      /*
       * Recovery prompt gets progressively
       * stronger.
       */
      let requestForAttempt = {
        ...finalRequest,
      };

      if (
        attempt > 0 &&
        isContinuation
      ) {
        requestForAttempt = {
          ...requestForAttempt,

          systemPrompt: [
            requestForAttempt.systemPrompt,

            "=== FINAL RECOVERY ===",
            "The external tool has already completed successfully.",
            "You MUST answer the user's original request now.",
            "The tool result is already present in the conversation.",
            "Do not call a tool.",
            "Do not return an empty response.",
            "Give a concise factual answer based on the tool result.",
            "=== END FINAL RECOVERY ===",
          ]
            .filter(Boolean)
            .join("\n"),
        };
      }

      /*
       * File support is retained.
       */
      let response;

      if (
        files.length
      ) {
        /*
         * buildArnaruRequest already
         * extracts actual file buffers.
         *
         * This branch intentionally falls
         * back to normal call if no actual
         * buffers are available.
         */
        response =
          await callArnaruChat(
            requestForAttempt
          );
      } else {
        response =
          await callArnaruChat(
            requestForAttempt
          );
      }

      finalResponse =
        response;

      if (
        response.conversationId
      ) {
        finalRequest = {
          ...requestForAttempt,

          conversationId:
            response.conversationId,
        };

        updateArnaruSession(
          sessionId,
          response
        );
      }

      const text =
        response.text
          ?.trim() || "";

      /*
       * FIRST:
       * parse tool calls.
       *
       * A model tool call must never be
       * mistaken for normal text.
       */
      const toolCalls =
        parseToolCalls(
          text
        );

      if (
        toolCalls.length
      ) {
        const completionId =
          generateId();

        return json(
          createCompletion(
            model,
            "",
            toolCalls,
            completionId
          ),
          200,
          headers
        );
      }

      /*
       * Normal answer.
       */
      if (
        !isEmpty(text)
      ) {
        finalText =
          text;

        break;
      }

      /*
       * Empty.
       */
      lastError =
        new Error(
          `Arnaru returned an empty response (attempt ${
            attempt + 1
          })`
        );

      markArnaruFailure(
        sessionId
      );

    } catch (error) {
      lastError =
        error;

      markArnaruFailure(
        sessionId
      );

      /*
       * Only retry continuation.
       */
      if (
        !isContinuation
      ) {
        break;
      }
    }
  }

  /*
   * We have a real answer.
   */
  if (
    !isEmpty(finalText)
  ) {
    const completionId =
      generateId();

    return json(
      createCompletion(
        model,
        finalText,
        [],
        completionId
      ),
      200,
      headers
    );
  }

  /*
   * NEVER silently return:
   *
   * ""
   *
   * That was the original bug.
   */
  return json(
    {
      error: {
        message:
          isContinuation
            ? "Arnaru returned an empty response after tool execution. The tool result was received, but the model failed to produce the final assistant message."
            : "Arnaru returned an empty response.",

        type:
          "empty_upstream_response",

        retryable:
          isContinuation,

        details:
          process.env.NODE_ENV ===
          "development"
            ? safeString(
                lastError
              )
            : undefined,
      },
    },
    502,
    headers
  );
}

/* =========================================================
 * OPTIONS
 * ========================================================= */

export async function OPTIONS() {
  return new NextResponse(
    null,
    {
      status: 204,
      headers:
        corsHeaders(),
    }
  );
}
