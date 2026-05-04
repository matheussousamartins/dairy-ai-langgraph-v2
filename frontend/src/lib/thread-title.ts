const GREETING_PATTERNS = [
  /^oi+!?$/i,
  /^ol[aá]+!?$/i,
  /^ola+!?$/i,
  /^bom dia!?$/i,
  /^boa tarde!?$/i,
  /^boa noite!?$/i,
  /^e ai+!?$/i,
  /^opa!?$/i,
  /^hey!?$/i,
  /^hello!?$/i,
  /^hi!?$/i,
  /^teste!?$/i,
  /^testando!?$/i,
  /^ok!?$/i,
];

const LEADING_PREFIXES = [
  /^quais?\s+s[aã]o\s+/i,
  /^qual\s+[éeoáaõ]+\s+/i,
  /^me\s+explique\s+/i,
  /^explique\s+/i,
  /^fale\s+sobre\s+/i,
  /^quero\s+saber\s+sobre\s+/i,
  /^preciso\s+saber\s+sobre\s+/i,
  /^pode\s+me\s+ajudar\s+com\s+/i,
  /^como\s+funciona\s+/i,
];

function normalizeWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function trimTrailingPunctuation(value: string) {
  return value.replace(/[.?!,:;]+$/g, "").trim();
}

export function isWeakThreadPrompt(value?: string | null) {
  const normalized = normalizeWhitespace(value ?? "");
  if (!normalized) return true;
  if (normalized.length < 8) return true;
  return GREETING_PATTERNS.some((pattern) => pattern.test(normalized));
}

export function summarizeThreadTitleFromText(value?: string | null) {
  const normalized = trimTrailingPunctuation(normalizeWhitespace(value ?? ""));
  if (!normalized || isWeakThreadPrompt(normalized)) return null;

  let concise = normalized;
  for (const prefix of LEADING_PREFIXES) {
    concise = concise.replace(prefix, "");
  }

  concise = trimTrailingPunctuation(normalizeWhitespace(concise));
  if (!concise) {
    concise = normalized;
  }

  if (concise.length > 52) {
    const cut = concise.slice(0, 52);
    const lastSpace = cut.lastIndexOf(" ");
    concise = `${(lastSpace > 20 ? cut.slice(0, lastSpace) : cut).trim()}...`;
  }

  return concise.charAt(0).toUpperCase() + concise.slice(1);
}

export function summarizeThreadTitleFromMessages(messages: Array<{ role?: string; content?: string }>, fallbackId?: string) {
  const userMessages = messages.filter((message) => message.role === "user").map((message) => message.content ?? "");
  const bestPrompt = userMessages.find((content) => !isWeakThreadPrompt(content)) ?? userMessages.find(Boolean);
  const summarized = summarizeThreadTitleFromText(bestPrompt);
  void fallbackId;
  return summarized ?? "Nova sessão";
}

export function shouldReplaceSessionTitle(currentTitle?: string | null) {
  const normalized = normalizeWhitespace(currentTitle ?? "");
  if (!normalized) return true;
  if (/^thread\s+/i.test(normalized)) return true;
  if (/^nova sess[aã]o$/i.test(normalized)) return true;
  return isWeakThreadPrompt(normalized);
}
