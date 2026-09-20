export const PROMPT_COMPACTION_MARKER = '\n\n[Earlier context compacted by Arnaru proxy]\n\n';

export const MIN_UPSTREAM_PROMPT_CHARS = 16000;

export const MAX_UPSTREAM_PROMPT_CHARS = (() => {
  const configured = Number(process.env.MAX_UPSTREAM_PROMPT_CHARS);
  return Number.isFinite(configured)
    ? Math.max(MIN_UPSTREAM_PROMPT_CHARS, Math.floor(configured))
    : 200000;
})();

export const MAX_EMPTY_RETRIES = (() => {
  const configured = Number(process.env.MAX_EMPTY_RETRIES);
  return Number.isFinite(configured) && configured > 0 ? Math.floor(configured) : 4;
})();

/**
 * Strip surrogate code units and other broken Unicode so the
 * upstream gateway (and Telegram-style sinks) never crash on them.
 */
export function sanitizeContent(text: unknown): string {
  return String(text || '').replace(/[\ud800-\udfff]/g, '');
}

/**
 * Keep the head and tail of an over-long prompt and mark the
 * dropped middle. Mirrors FreeDeepseekAPI's compaction behavior.
 */
export function truncatePromptMiddle(
  value: string,
  maxChars: number,
  headRatio = 0.5
): string {
  const text = String(value || '');

  if (maxChars <= 0) return '';
  if (text.length <= maxChars) return text;
  if (maxChars <= PROMPT_COMPACTION_MARKER.length) {
    return text.substring(text.length - maxChars);
  }

  const payloadChars = maxChars - PROMPT_COMPACTION_MARKER.length;
  const headChars = Math.max(0, Math.floor(payloadChars * headRatio));
  const tailChars = Math.max(0, payloadChars - headChars);

  return (
    text.substring(0, headChars) +
    PROMPT_COMPACTION_MARKER +
    text.substring(text.length - tailChars)
  );
}

/**
 * Append a trailing instruction (e.g. a recovery/continuation note)
 * while keeping the whole prompt inside the upstream budget.
 */
export function appendPromptInstruction(
  promptText: string,
  instruction: string,
  maxChars = MAX_UPSTREAM_PROMPT_CHARS
): string {
  const suffix = `\n\n${String(instruction || '').trim()}`;
  const baseBudget = Math.max(0, maxChars - suffix.length);

  return truncatePromptMiddle(promptText, baseBudget, 0.35) + suffix;
}

export function isContextTooLongError(error: unknown): boolean {
  const message =
    typeof error === 'string'
      ? error
      : `${(error as any)?.content || ''} ${(error as any)?.message || ''} ${(error as any)?.finish_reason || ''} ${(error as any)?.type || ''}`;

  return /(?:content|prompt|context).{0,40}(?:too\s+long|too\s+large|length|limit|maximum)|maximum.{0,30}(?:context|token)|too\s+many\s+tokens|содержани[ея]\s+слишком\s+длин|контекст.{0,30}(?:длин|лимит)|内容.{0,12}(?:过长|太长)|上下文.{0,12}(?:过长|超出)/i.test(
    message
  );
}

/**
 * Detect tool-call markup (legacy, XML, DSML, or JSON envelope)
 * that was emitted but could not be parsed.
 */
export function looksLikeToolCallMarkup(text: unknown): boolean {
  return /TOOL_CALL:\s*[\w-]+|<\s*tool_call\b|[|｜]+\s*DSML\s*[|｜]+|[<＜]\s*\/?\s*(?:DSML)?(?:[\w.-]+:)?(?:tool[\s_-]*calls|function[\s_-]*calls|invoke)\b|["'](?:tool_call|tool_calls|function_call)["']\s*:/i.test(
    String(text || '')
  );
}

export function isEmptyContent(text: string | null | undefined): boolean {
  return !text || !String(text).trim();
}

export const RETRY_INITIAL_DELAY_MS = 2000;
export const RETRY_BACKOFF_FACTOR = 2;
export const RETRY_JITTER_FACTOR = 0.25;
export const RETRY_MAX_DELAY_MS = 30000;

/**
 * Exponential backoff with jitter, capped at RETRY_MAX_DELAY_MS.
 * Mirrors opencode's retry policy (2s initial, x2 growth, 25% jitter).
 */
export function retryDelay(
  attempt: number,
  random: () => number = Math.random
): number {
  const safeAttempt = Math.max(1, Math.floor(attempt));
  const base =
    RETRY_INITIAL_DELAY_MS *
    Math.pow(RETRY_BACKOFF_FACTOR, safeAttempt - 1);
  const capped = Math.min(base, RETRY_MAX_DELAY_MS);
  const jitter =
    capped * RETRY_JITTER_FACTOR * (random() * 2 - 1);
  return Math.max(0, Math.min(capped + jitter, RETRY_MAX_DELAY_MS));
}

const RETRYABLE_PATTERNS: RegExp[] = [
  /\b(?:429|500|502|503|504|524)\b/,
  /rate limit|rate-limit|rate_limit|too many requests/i,
  /overloaded|service unavailable|service_unavailable|internal server error|server error|server_error|provider returned error|provider_returned_error/i,
  /terminated|fetch failed|failed to fetch|network[-_\s]error|upstream connect|connection refused|connection reset|socket hang up|econnrefused|econnreset|etimedout|enotfound/i,
  /\btimeout\b|timed out|time out/i,
  /try your request again|retry your request|resource exhausted|resource_exhausted/i,
  /try again (?:later|in\b)|(?:currently|temporarily) at capacity/i
];

/**
 * Decide whether an upstream error is worth retrying. An empty or
 * unknown error is treated as retryable so empty responses keep recovering.
 */
export function isRetryableError(error: unknown): boolean {
  const message =
    typeof error === 'string'
      ? error
      : `${(error as any)?.content || ''} ${
          (error as any)?.message || ''
        } ${(error as any)?.type || ''}`;
  const text = String(message || '').trim();
  if (!text) return true;
  return RETRYABLE_PATTERNS.some(pattern => pattern.test(text));
}

export const TOOL_OUTPUT_MAX_CHARS = (() => {
  const configured = Number(process.env.TOOL_OUTPUT_MAX_CHARS);
  return Number.isFinite(configured) && configured > 0
    ? Math.floor(configured)
    : 8000;
})();

const TOOL_OUTPUT_TRUNCATION_MARKER =
  '\n\n... [OUTPUT TRUNCATED DUE TO LENGTH - PLEASE CONTINUE] ...\n\n';

/**
 * Keep the head and tail of an oversized tool output so the model
 * still sees the beginning and the most recent lines.
 */
export function truncateToolOutput(
  value: unknown,
  maxChars = TOOL_OUTPUT_MAX_CHARS
): string {
  const text = typeof value === 'string' ? value : String(value ?? '');
  if (text.length <= maxChars) return text;
  if (maxChars <= TOOL_OUTPUT_TRUNCATION_MARKER.length) {
    return text.substring(0, maxChars);
  }
  const payloadChars = maxChars - TOOL_OUTPUT_TRUNCATION_MARKER.length;
  const headChars = Math.floor(payloadChars * 0.7);
  const tailChars = payloadChars - headChars;
  return (
    text.substring(0, headChars) +
    TOOL_OUTPUT_TRUNCATION_MARKER +
    text.substring(text.length - tailChars)
  );
}
