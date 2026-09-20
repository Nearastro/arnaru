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
