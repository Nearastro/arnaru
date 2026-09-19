import { ArnaruSSEData } from './types';

export interface SSEParseResult {
  fullMessage: string;
  newConversationId: string | null;
  hasError: boolean;
  errorMessage: string | null;
}

function extractText(data: any): string {
  if (data == null) return '';

  if (typeof data === 'string') {
    return data === '[DONE]' ? '' : data;
  }

  // OpenAI-style choices
  if (Array.isArray(data.choices) && data.choices.length > 0) {
    const choice = data.choices[0];

    if (choice.delta) {
      const value = extractText(choice.delta);
      if (value) return value;
    }

    if (choice.message) {
      const value = extractText(choice.message);
      if (value) return value;
    }

    if (typeof choice.text === 'string') {
      return choice.text;
    }

    if (typeof choice.content === 'string') {
      return choice.content;
    }
  }

  // Common response fields
  const fields = [
    'answer',
    'text',
    'response',
    'content',
    'message',
    'output',
    'result',
    'generated_text',
    'delta',
    'token'
  ];

  for (const field of fields) {
    if (data[field] == null) continue;

    const value = data[field];

    if (typeof value === 'string') {
      if (value !== '[DONE]') return value;
      continue;
    }

    if (Array.isArray(value)) {
      const joined = value
        .map((item: any) => extractText(item))
        .filter(Boolean)
        .join('');

      if (joined) return joined;
    }

    if (typeof value === 'object') {
      const nested = extractText(value);
      if (nested) return nested;
    }
  }

  // Nested data object
  if (data.data !== undefined && data.data !== data) {
    const nested = extractText(data.data);
    if (nested) return nested;
  }

  // Some APIs wrap the actual result inside `response.data`
  if (data.response && typeof data.response === 'object') {
    const nested = extractText(data.response);
    if (nested) return nested;
  }

  return '';
}

function parseDataLine(dataStr: string): {
  text: string;
  conversationId: string | null;
  error: string | null;
} {
  if (!dataStr || dataStr === '[DONE]') {
    return {
      text: '',
      conversationId: null,
      error: null
    };
  }

  try {
    const data: any = JSON.parse(dataStr);

    if (data.data === '[DONE]') {
      return {
        text: '',
        conversationId: data.conversationId || null,
        error: null
      };
    }

    const conversationId =
      typeof data.conversationId === 'string'
        ? data.conversationId
        : typeof data.conversation_id === 'string'
          ? data.conversation_id
          : null;

    let error: string | null = null;

    if (data.error) {
      error =
        typeof data.error === 'string'
          ? data.error
          : data.error.message || JSON.stringify(data.error);
    }

    const text = extractText(data);

    return {
      text,
      conversationId,
      error
    };
  } catch {
    // Not JSON. Treat it as a raw SSE payload.
    const cleanText = dataStr
      .replace(/^["']+|["']+$/g, '')
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'")
      .replace(/\[DONE\]/gi, '')
      .trim();

    return {
      text: cleanText,
      conversationId: null,
      error: null
    };
  }
}

function parseRawResponse(rawText: string): SSEParseResult {
  let fullMessage = '';
  let newConversationId: string | null = null;
  let hasError = false;
  let errorMessage: string | null = null;

  const trimmedRaw = rawText.trim();

  if (!trimmedRaw) {
    return {
      fullMessage: '',
      newConversationId: null,
      hasError: false,
      errorMessage: null
    };
  }

  // ---------------------------------------------------------
  // 1. Normal SSE
  // ---------------------------------------------------------
  const lines = rawText.split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed || !trimmed.startsWith('data:')) {
      continue;
    }

    const dataStr = trimmed.slice(5).trim();

    if (!dataStr || dataStr === '[DONE]') {
      continue;
    }

    const parsed = parseDataLine(dataStr);

    if (parsed.conversationId) {
      newConversationId = parsed.conversationId;
    }

    if (parsed.error) {
      hasError = true;
      errorMessage = parsed.error;
    }

    if (parsed.text) {
      fullMessage += parsed.text;
    }
  }

  // ---------------------------------------------------------
  // 2. Raw JSON response
  // ---------------------------------------------------------
  if (!fullMessage.trim()) {
    try {
      const data: any = JSON.parse(trimmedRaw);

      if (typeof data.conversationId === 'string') {
        newConversationId = data.conversationId;
      }

      if (typeof data.conversation_id === 'string') {
        newConversationId = data.conversation_id;
      }

      if (data.error) {
        hasError = true;
        errorMessage =
          typeof data.error === 'string'
            ? data.error
            : data.error.message || JSON.stringify(data.error);
      }

      fullMessage = extractText(data);
    } catch {
      // Not JSON.
    }
  }

  // ---------------------------------------------------------
  // 3. Raw text containing SSE without line breaks
  // ---------------------------------------------------------
  if (!fullMessage.trim() && trimmedRaw.includes('data:')) {
    const matches = trimmedRaw.match(/data:\s*(.+?)(?=\s+data:|$)/g);

    if (matches) {
      for (const match of matches) {
        const dataStr = match.replace(/^data:\s*/, '').trim();

        if (!dataStr || dataStr === '[DONE]') continue;

        const parsed = parseDataLine(dataStr);

        if (parsed.conversationId) {
          newConversationId = parsed.conversationId;
        }

        if (parsed.error) {
          hasError = true;
          errorMessage = parsed.error;
        }

        if (parsed.text) {
          fullMessage += parsed.text;
        }
      }
    }
  }

  // ---------------------------------------------------------
  // 4. Last-resort plain response
  // ---------------------------------------------------------
  if (!fullMessage.trim()) {
    const cleaned = trimmedRaw
      .replace(/^data:\s*/gm, '')
      .replace(/\[DONE\]/gi, '')
      .trim();

    if (cleaned && !cleaned.startsWith('{')) {
      fullMessage = cleaned;
    }
  }

  return {
    fullMessage: fullMessage
      .replace(/\[DONE\]/gi, '')
      .trim(),
    newConversationId,
    hasError,
    errorMessage
  };
}

export function parseSSE(rawText: string): SSEParseResult {
  return parseRawResponse(rawText);
}

export function* parseSSEStream(
  rawText: string
): Generator<{
  text: string;
  conversationId?: string;
  error?: string;
}> {
  const lines = rawText.split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed || !trimmed.startsWith('data:')) {
      continue;
    }

    const dataStr = trimmed.slice(5).trim();

    if (!dataStr || dataStr === '[DONE]') {
      continue;
    }

    const parsed = parseDataLine(dataStr);

    if (parsed.text || parsed.error || parsed.conversationId) {
      yield {
        text: parsed.text,
        ...(parsed.conversationId
          ? { conversationId: parsed.conversationId }
          : {}),
        ...(parsed.error
          ? { error: parsed.error }
          : {})
      };
    }
  }

  // If Arnaru returned normal JSON instead of SSE,
  // don't silently produce an empty stream.
  if (!lines.some(line => line.trim().startsWith('data:'))) {
    const parsed = parseRawResponse(rawText);

    if (
      parsed.fullMessage ||
      parsed.errorMessage ||
      parsed.newConversationId
    ) {
      yield {
        text: parsed.fullMessage,
        ...(parsed.newConversationId
          ? { conversationId: parsed.newConversationId }
          : {}),
        ...(parsed.errorMessage
          ? { error: parsed.errorMessage }
          : {})
      };
    }
  }
}
