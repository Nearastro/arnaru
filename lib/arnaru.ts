const ARNARU_BASE_URL =
  process.env.ARNARU_BASE_URL || "https://arnaru-ai.vercel.app";

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL || "claude-fable-5";

const DEFAULT_WEB_SEARCH =
  process.env.DEFAULT_WEB_SEARCH !== "false";

const MAX_FILES = 9;

const MAX_HISTORY_MESSAGES = Number(
  process.env.ARNARU_MAX_HISTORY_MESSAGES || 24
);

const MAX_HISTORY_CHARS = Number(
  process.env.ARNARU_MAX_HISTORY_CHARS || 12000
);

const MAX_QUESTION_CHARS = Number(
  process.env.ARNARU_MAX_QUESTION_CHARS || 80000
);

export type ArnaruMessage = {
  role: string;
  content?: unknown;
  tool_calls?: any[];
  tool_call_id?: string;
  name?: string;
};

export type ArnaruFile = {
  filename: string;
  mimeType: string;
  data: Buffer;
};

export type ArnaruRequest = {
  question: string;
  model: string;
  conversationId?: string;
  webSearch?: boolean;
  systemPrompt?: string;
};

export type ArnaruResponse = {
  text: string;
  conversationId?: string;
  raw?: unknown;
};

type SessionState = {
  id: string;
  createdAt: number;
  updatedAt: number;

  parentMessageId?: string;

  history: ArnaruMessage[];

  lastQuestion?: string;

  lastToolResult?: string;

  failures: number;
};

const sessions = new Map<string, SessionState>();

/* ---------------------------------------------------------
 * Generic helpers
 * --------------------------------------------------------- */

function now() {
  return Date.now();
}

function safeString(value: unknown): string {
  if (value === null || value === undefined) return "";

  if (typeof value === "string") {
    return value;
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeText(value: unknown): string {
  return safeString(value)
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value;

  return (
    value.slice(0, Math.max(0, max - 200)) +
    "\n\n...[context truncated]...\n\n" +
    value.slice(-180)
  );
}

function makeSessionId(seed?: string) {
  if (seed) {
    return seed
      .replace(/[^a-zA-Z0-9._:-]/g, "_")
      .slice(0, 160);
  }

  return `arnaru-${cryptoRandomId()}`;
}

function cryptoRandomId() {
  return `${Date.now()}-${Math.random()
    .toString(36)
    .slice(2, 12)}`;
}

/* ---------------------------------------------------------
 * Session store
 * --------------------------------------------------------- */

function getSession(id?: string): SessionState {
  const sessionId = makeSessionId(id);

  let session = sessions.get(sessionId);

  if (!session) {
    session = {
      id: sessionId,
      createdAt: now(),
      updatedAt: now(),
      history: [],
      failures: 0,
    };

    sessions.set(sessionId, session);
  }

  session.updatedAt = now();

  return session;
}

export function resetArnaruSession(id: string) {
  sessions.delete(id);
}

export function clearArnaruSessions() {
  sessions.clear();
}

/* ---------------------------------------------------------
 * Tool-call serialization
 *
 * IMPORTANT:
 * Keep this chronological.
 * --------------------------------------------------------- */

function stringifyToolCalls(toolCalls: any[]): string {
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
    return "";
  }

  return toolCalls
    .map((call) => {
      const functionName =
        call?.function?.name ||
        call?.name ||
        "unknown";

      const args =
        call?.function?.arguments ??
        call?.arguments ??
        "{}";

      const id =
        call?.id ||
        call?.tool_call_id ||
        "tool-call";

      return [
        "TOOL_CALL",
        `id: ${id}`,
        `name: ${functionName}`,
        `arguments: ${safeString(args)}`,
        "END_TOOL_CALL",
      ].join("\n");
    })
    .join("\n\n");
}

/* ---------------------------------------------------------
 * Message → Arnaru text
 * --------------------------------------------------------- */

function messageToText(message: ArnaruMessage): string {
  const role = message.role;

  /* system */
  if (role === "system") {
    return `System:\n${normalizeText(message.content)}`;
  }

  /* tool result */
  if (role === "tool") {
    const id =
      message.tool_call_id ||
      "unknown-tool-call";

    return [
      "TOOL_RESULT",
      `tool_call_id: ${id}`,
      "result:",
      normalizeText(message.content),
      "END_TOOL_RESULT",
    ].join("\n");
  }

  /* assistant tool call */
  if (
    role === "assistant" &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length
  ) {
    const toolCalls =
      stringifyToolCalls(message.tool_calls);

    const normalText =
      normalizeText(message.content);

    if (normalText) {
      return [
        "Assistant:",
        normalText,
        toolCalls,
      ].join("\n");
    }

    return `Assistant:\n${toolCalls}`;
  }

  /* normal assistant */
  if (role === "assistant") {
    return `Assistant:\n${normalizeText(message.content)}`;
  }

  /* user */
  if (role === "user") {
    return `User:\n${normalizeText(message.content)}`;
  }

  return `${role}:\n${normalizeText(message.content)}`;
}

/* ---------------------------------------------------------
 * History normalization
 * --------------------------------------------------------- */

function cloneMessage(message: ArnaruMessage): ArnaruMessage {
  return {
    role: message.role,
    content: message.content,
    tool_calls: message.tool_calls,
    tool_call_id: message.tool_call_id,
    name: message.name,
  };
}

function pushHistory(
  session: SessionState,
  messages: ArnaruMessage[]
) {
  for (const message of messages) {
    session.history.push(cloneMessage(message));
  }

  /*
   * Keep chronological history.
   */
  while (
    session.history.length >
    MAX_HISTORY_MESSAGES
  ) {
    session.history.shift();
  }

  /*
   * Also cap character count.
   */
  let total = 0;

  const kept: ArnaruMessage[] = [];

  for (
    let i = session.history.length - 1;
    i >= 0;
    i--
  ) {
    const text = messageToText(session.history[i]);

    total += text.length;

    if (total > MAX_HISTORY_CHARS) {
      break;
    }

    kept.unshift(session.history[i]);
  }

  session.history = kept;
}

/* ---------------------------------------------------------
 * Tool-result detection
 * --------------------------------------------------------- */

export function hasToolResult(
  messages: ArnaruMessage[]
): boolean {
  return messages.some(
    (message) => message.role === "tool"
  );
}

export function hasToolCall(
  messages: ArnaruMessage[]
): boolean {
  return messages.some(
    (message) =>
      message.role === "assistant" &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0
  );
}

/* ---------------------------------------------------------
 * Build chronological conversation
 * --------------------------------------------------------- */

function buildConversation(
  messages: ArnaruMessage[]
): string {
  const parts: string[] = [];

  for (const message of messages) {
    const text = messageToText(message);

    if (text.trim()) {
      parts.push(text);
    }
  }

  return parts.join("\n\n");
}

/* ---------------------------------------------------------
 * Recovery history
 * --------------------------------------------------------- */

function buildRecoveryHistory(
  session: SessionState
): string {
  if (!session.history.length) {
    return "";
  }

  const parts = session.history.map(messageToText);

  let result = parts.join("\n\n");

  result = truncate(
    result,
    MAX_HISTORY_CHARS
  );

  return [
    "=== RECOVERY CONTEXT ===",
    result,
    "=== END RECOVERY CONTEXT ===",
  ].join("\n");
}

/* ---------------------------------------------------------
 * Agent continuation
 * --------------------------------------------------------- */

function buildContinuationPrompt(): string {
  return [
    "The external tool execution has completed.",
    "",
    "Continue the original user request now.",
    "",
    "Use the tool result that appears immediately before this instruction.",
    "",
    "Do not request the same tool again unless the original task genuinely requires it.",
    "",
    "Return the final assistant response to the user.",
    "",
    "IMPORTANT:",
    "Never return an empty response.",
  ].join("\n");
}

/* ---------------------------------------------------------
 * File helpers
 * --------------------------------------------------------- */

function isFilePart(value: any): boolean {
  if (!value || typeof value !== "object") {
    return false;
  }

  return (
    value.type === "file" ||
    value.type === "input_file" ||
    value.type === "document"
  );
}

function extractFiles(
  messages: ArnaruMessage[]
): ArnaruFile[] {
  const files: ArnaruFile[] = [];

  for (const message of messages) {
    const content = message.content;

    if (!Array.isArray(content)) {
      continue;
    }

    for (const part of content) {
      if (!isFilePart(part)) {
        continue;
      }

      const filename =
        part.filename ||
        part.name ||
        "attachment";

      const mimeType =
        part.mime_type ||
        part.mimeType ||
        "application/octet-stream";

      let data: Buffer | null = null;

      if (typeof part.data === "string") {
        try {
          data = Buffer.from(
            part.data,
            "base64"
          );
        } catch {
          data = null;
        }
      }

      if (data) {
        files.push({
          filename,
          mimeType,
          data,
        });
      }

      if (files.length >= MAX_FILES) {
        return files;
      }
    }
  }

  return files;
}

/* ---------------------------------------------------------
 * Content extraction
 * --------------------------------------------------------- */

function extractContent(
  content: unknown
): string {
  if (typeof content === "string") {
    return content;
  }

  if (content === null || content === undefined) {
    return "";
  }

  if (Array.isArray(content)) {
    return content
      .map((part: any) => {
        if (typeof part === "string") {
          return part;
        }

        if (
          part?.type === "text" &&
          typeof part.text === "string"
        ) {
          return part.text;
        }

        if (
          typeof part?.content === "string"
        ) {
          return part.content;
        }

        return "";
      })
      .filter(Boolean)
      .join("\n");
  }

  return safeString(content);
}

/* ---------------------------------------------------------
 * Normalize question length
 * --------------------------------------------------------- */

function compactQuestion(
  question: string,
  messages: ArnaruMessage[],
  session: SessionState
): string {
  if (
    question.length <=
    MAX_QUESTION_CHARS
  ) {
    return question;
  }

  /*
   * Preserve:
   * 1. original user task
   * 2. recent conversation
   * 3. fresh tool result
   * 4. recovery instructions
   */

  const firstUser =
    messages.find(
      (message) => message.role === "user"
    );

  const firstTask = firstUser
    ? messageToText(firstUser)
    : "";

  const recent =
    session.history
      .slice(-10)
      .map(messageToText)
      .join("\n\n");

  const toolResults =
    session.history
      .filter(
        (message) =>
          message.role === "tool"
      )
      .slice(-4)
      .map(messageToText)
      .join("\n\n");

  const compacted = [
    "=== ORIGINAL TASK ===",
    firstTask,

    "=== RECENT CONTEXT ===",
    recent,

    "=== FRESH TOOL RESULTS ===",
    toolResults,

    "=== CONTINUATION ===",
    buildContinuationPrompt(),
  ].join("\n\n");

  return truncate(
    compacted,
    MAX_QUESTION_CHARS
  );
}

/* ---------------------------------------------------------
 * Main message extraction
 * --------------------------------------------------------- */

export function extractMessageContent(
  messages: ArnaruMessage[],
  sessionId?: string
) {
  const session = getSession(sessionId);

  pushHistory(session, messages);

  const conversation =
    buildConversation(messages);

  const hasTool =
    hasToolResult(messages);

  const recovery =
    hasTool
      ? buildRecoveryHistory(session)
      : "";

  const continuation =
    hasTool
      ? buildContinuationPrompt()
      : "";

  let question = conversation;

  /*
   * Recovery is only used when there is a tool
   * continuation or when explicitly needed.
   */
  if (hasTool) {
    question = [
      conversation,
      "",
      recovery,
      "",
      "=== AGENT CONTINUATION ===",
      continuation,
      "=== END AGENT CONTINUATION ===",
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  question = compactQuestion(
    question,
    messages,
    session
  );

  const files =
    extractFiles(messages);

  session.lastQuestion =
    question;

  const lastTool =
    [...messages]
      .reverse()
      .find(
        (message) =>
          message.role === "tool"
      );

  if (lastTool) {
    session.lastToolResult =
      normalizeText(lastTool.content);
  }

  return {
    question,
    files,
    session,
  };
}

/* ---------------------------------------------------------
 * Build Arnaru request
 * --------------------------------------------------------- */

export function buildArnaruRequest(
  messages: ArnaruMessage[],
  options?: {
    model?: string;
    conversationId?: string;
    webSearch?: boolean;
    systemPrompt?: string;
    sessionId?: string;
  }
): ArnaruRequest {
  const session =
    getSession(
      options?.sessionId ||
      options?.conversationId
    );

  const extracted =
    extractMessageContent(
      messages,
      session.id
    );

  let systemPrompt =
    options?.systemPrompt || "";

  if (
    hasToolResult(messages)
  ) {
    systemPrompt = [
      systemPrompt,
      buildContinuationPrompt(),
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  return {
    question: extracted.question,
    model:
      options?.model ||
      DEFAULT_MODEL,

    conversationId:
      options?.conversationId ||
      session.parentMessageId,

    webSearch:
      options?.webSearch ??
      DEFAULT_WEB_SEARCH,

    systemPrompt:
      systemPrompt || undefined,
  };
}

/* ---------------------------------------------------------
 * SSE / Arnaru response parser
 * --------------------------------------------------------- */

function extractTextFromObject(
  value: any
): string {
  if (!value) return "";

  const candidates = [
    value.answer,
    value.text,
    value.response,
    value.message,
    value.content,
    value.output,
    value.result,
    value.generated_text,
    value.delta,
  ];

  for (const candidate of candidates) {
    if (
      typeof candidate === "string" &&
      candidate.trim()
    ) {
      return candidate;
    }
  }

  if (
    value.message &&
    typeof value.message === "object"
  ) {
    const nested =
      extractTextFromObject(
        value.message
      );

    if (nested) return nested;
  }

  if (
    Array.isArray(value.choices)
  ) {
    for (const choice of value.choices) {
      const text =
        extractTextFromObject(choice);

      if (text) return text;
    }
  }

  return "";
}

export function parseArnaruResponse(
  raw: string
): ArnaruResponse {
  const textChunks: string[] = [];

  let conversationId: string | undefined;

  const lines =
    raw.split(/\r?\n/);

  for (const line of lines) {
    let value =
      line.startsWith("data:")
        ? line.slice(5).trim()
        : line.trim();

    if (!value) continue;

    if (value === "[DONE]") {
      continue;
    }

    try {
      const json =
        JSON.parse(value);

      if (
        typeof json.conversationId ===
          "string"
      ) {
        conversationId =
          json.conversationId;
      }

      if (
        typeof json.newConversationId ===
          "string"
      ) {
        conversationId =
          json.newConversationId;
      }

      if (
        typeof json.conversation_id ===
          "string"
      ) {
        conversationId =
          json.conversation_id;
      }

      const text =
        extractTextFromObject(json);

      if (text) {
        textChunks.push(text);
      }
    } catch {
      /*
       * Some upstream variants may emit
       * plain text after data:.
       */
      if (
        !line.startsWith("event:") &&
        value.length > 0
      ) {
        textChunks.push(value);
      }
    }
  }

  /*
   * If SSE parsing found nothing,
   * attempt the complete body as JSON.
   */
  if (!textChunks.length) {
    try {
      const json =
        JSON.parse(raw);

      const text =
        extractTextFromObject(json);

      if (text) {
        textChunks.push(text);
      }

      conversationId =
        conversationId ||
        json.conversationId ||
        json.newConversationId ||
        json.conversation_id;
    } catch {
      /*
       * Last fallback:
       * plain response body.
       */
      const plain =
        raw.trim();

      if (
        plain &&
        !plain.startsWith("<")
      ) {
        textChunks.push(plain);
      }
    }
  }

  /*
   * Deduplicate accidental repeated
   * SSE snapshots.
   */
  const result =
    textChunks
      .filter(Boolean)
      .reduce(
        (acc, chunk) => {
          if (!acc) return chunk;

          if (chunk === acc) {
            return acc;
          }

          if (
            chunk.startsWith(acc)
          ) {
            return chunk;
          }

          return acc + chunk;
        },
        ""
      );

  return {
    text: result.trim(),
    conversationId,
    raw,
  };
}

/* ---------------------------------------------------------
 * Fetch Arnaru
 * --------------------------------------------------------- */

async function fetchArnaru(
  request: ArnaruRequest,
  files: ArnaruFile[]
): Promise<ArnaruResponse> {
  const url =
    `${ARNARU_BASE_URL}/api/chat`;

  let response: Response;

  if (files.length) {
    const form =
      new FormData();

    form.append(
      "question",
      request.question
    );

    form.append(
      "model",
      request.model
    );

    if (request.conversationId) {
      form.append(
        "conversationId",
        request.conversationId
      );
    }

    form.append(
      "webSearch",
      String(
        request.webSearch ?? false
      )
    );

    if (request.systemPrompt) {
      form.append(
        "systemPrompt",
        request.systemPrompt
      );
    }

    for (const file of files) {
      form.append(
        "files",
        new Blob(
          [file.data],
          { type: file.mimeType }
        ),
        file.filename
      );
    }

    response =
      await fetch(url, {
        method: "POST",
        body: form,
      });
  } else {
    response =
      await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json",
        },
        body: JSON.stringify({
          question:
            request.question,

          model:
            request.model,

          conversationId:
            request.conversationId,

          webSearch:
            request.webSearch,

          systemPrompt:
            request.systemPrompt,
        }),
      });
  }

  const raw =
    await response.text();

  if (!response.ok) {
    const error =
      new Error(
        `Arnaru HTTP ${response.status}: ${raw.slice(
          0,
          1000
        )}`
      );

    (error as any).status =
      response.status;

    throw error;
  }

  const parsed =
    parseArnaruResponse(raw);

  if (
    parsed.conversationId
  ) {
    request.conversationId =
      parsed.conversationId;
  }

  return parsed;
}

/* ---------------------------------------------------------
 * Public call
 * --------------------------------------------------------- */

export async function callArnaruChat(
  request: ArnaruRequest
) {
  return fetchArnaru(
    request,
    []
  );
}

export async function callArnaruChatWithFiles(
  request: ArnaruRequest,
  files: ArnaruFile[]
) {
  return fetchArnaru(
    request,
    files
  );
}

/* ---------------------------------------------------------
 * Session response bookkeeping
 * --------------------------------------------------------- */

export function updateArnaruSession(
  sessionId: string,
  response: ArnaruResponse
) {
  const session =
    sessions.get(sessionId);

  if (!session) return;

  session.updatedAt =
    now();

  session.failures = 0;

  if (
    response.conversationId
  ) {
    session.parentMessageId =
      response.conversationId;
  }
}

/* ---------------------------------------------------------
 * Failure bookkeeping
 * --------------------------------------------------------- */

export function markArnaruFailure(
  sessionId: string
) {
  const session =
    sessions.get(sessionId);

  if (!session) return;

  session.failures++;

  /*
   * Don't let a broken session stay forever.
   */
  if (
    session.failures >= 3
  ) {
    sessions.delete(sessionId);
  }
}

/* ---------------------------------------------------------
 * IDs / timestamp compatibility
 * --------------------------------------------------------- */

export function generateId() {
  return cryptoRandomId();
}

export function getTimestamp() {
  return Math.floor(
    Date.now() / 1000
  );
}
