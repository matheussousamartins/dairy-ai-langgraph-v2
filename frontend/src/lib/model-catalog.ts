export interface ModelCatalogItem {
  id: string;
  label: string;
  provider: string;
  description: string;
  family: string;
  familySubtitle: string;
  compatibilityStatus?: "ready" | "requires_adapter";
  compatibilityMessage?: string;
  setupHint?: string;
  selectable?: boolean;
  input_cost?: number;
  output_cost?: number;
}

export const MODEL_CATALOG: ModelCatalogItem[] = [
  {
    id: "openai/gpt-4.1",
    label: "GPT-4.1",
    provider: "OpenAI",
    description: "Modelo mais forte para respostas longas, raciocínio e consolidação.",
    family: "GPT-4.1",
    familySubtitle: "Mais capazes para tarefas complexas",
    input_cost: 2,
    output_cost: 8,
  },
];

const MODEL_ALIASES: Record<string, string> = {
  "gpt-4o-mini": "openai/gpt-4o-mini",
  "gpt-4o": "openai/gpt-4o",
  "gpt-4.1-mini": "openai/gpt-4.1-mini",
  "gpt-4.1": "openai/gpt-4.1",
  "gpt-4.1-nano": "openai/gpt-4.1-nano",
  "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
  "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
};

function titleCaseSlug(value: string) {
  const withoutProvider = value.includes("/") ? value.split("/").slice(1).join("/") : value;
  return withoutProvider
    .split(/[-_/]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function inferProvider(modelId: string) {
  const id = modelId.toLowerCase();
  if (id.startsWith("openai/") || id.startsWith("gpt-")) return "OpenAI";
  if (id.startsWith("anthropic/") || id.includes("claude")) return "Anthropic";
  if (id.startsWith("google/") || id.includes("gemini")) return "Google";
  if (id.startsWith("meta-llama/") || id.includes("llama")) return "Meta";
  if (id.startsWith("deepseek/")) return "DeepSeek";
  return "Custom";
}

function inferFamily(modelId: string, provider?: string) {
  const id = modelId.toLowerCase();
  if (id.includes("gpt-4o")) return "GPT-4o";
  if (id.includes("gpt-4.1")) return "GPT-4.1";
  if (id.includes("claude")) return "Claude";
  if (id.includes("gemini")) return "Gemini";
  if (id.includes("llama")) return "Llama";
  if (id.includes("deepseek")) return "DeepSeek";
  return provider ?? "Outros";
}

function inferFamilySubtitle(family: string) {
  if (family === "GPT-4o") return "Mais econômicos e com baixa latência";
  if (family === "GPT-4.1") return "Mais capazes para tarefas complexas";
  if (family === "Claude") return "Boa escrita e raciocínio consistente";
  if (family === "Gemini") return "Modelos Google com contexto amplo";
  if (family === "Llama") return "Alternativas abertas para testes";
  if (family === "DeepSeek") return "Modelos eficientes para raciocinio e codigo";
  return "Modelos disponíveis nesta categoria";
}

export function resolveEnabledModelCatalog(allowedIds: string[]): ModelCatalogItem[] {
  const normalized = allowedIds.map((item) => item.trim()).filter(Boolean);
  const fallback = normalized.length > 0 ? normalized : [MODEL_CATALOG[0]?.id ?? "gpt-4o-mini"];

  return fallback.map((rawModelId) => {
    const modelId = MODEL_ALIASES[rawModelId] ?? rawModelId;
    const known = MODEL_CATALOG.find((item) => item.id === modelId);
    if (known) return { ...known, id: rawModelId };
    const provider = inferProvider(modelId);
    const family = inferFamily(modelId, provider);
    return {
      id: rawModelId,
      label: titleCaseSlug(modelId),
      provider,
      description: "Modelo liberado pela whitelist do backend.",
      family,
      familySubtitle: inferFamilySubtitle(family),
      compatibilityStatus: "requires_adapter",
      compatibilityMessage: "Verifique a compatibilidade do provider configurado no backend.",
      setupHint: "Não foi possível validar o backend em tempo real.",
      selectable: false,
      input_cost: 0,
      output_cost: 0,
    };
  });
}
