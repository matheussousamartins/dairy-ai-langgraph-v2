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
    id: "gpt-4o-mini",
    label: "GPT-4o Mini",
    provider: "OpenAI",
    description: "Equilibrado para testes rápidos, baixo custo e boa latência.",
    family: "GPT-4o",
    familySubtitle: "Mais econômicos e com baixa latência",
    input_cost: 0.3,
    output_cost: 0.6,
  },
  {
    id: "gpt-4o",
    label: "GPT-4o",
    provider: "OpenAI",
    description: "Mais qualidade para respostas complexas e consolidação.",
    family: "GPT-4o",
    familySubtitle: "Mais econômicos e com baixa latência",
    input_cost: 2.5,
    output_cost: 10,
  },
  {
    id: "gpt-4.1-mini",
    label: "GPT-4.1 Mini",
    provider: "OpenAI",
    description: "Boa opção para testes gerais com foco em custo e consistência.",
    family: "GPT-4.1",
    familySubtitle: "Mais capazes para tarefas complexas",
    input_cost: 0.4,
    output_cost: 1.6,
  },
  {
    id: "gpt-4.1",
    label: "GPT-4.1",
    provider: "OpenAI",
    description: "Modelo mais forte para respostas longas, raciocínio e consolidação.",
    family: "GPT-4.1",
    familySubtitle: "Mais capazes para tarefas complexas",
    input_cost: 2,
    output_cost: 8,
  },
  {
    id: "gpt-4.1-nano",
    label: "GPT-4.1 Nano",
    provider: "OpenAI",
    description: "Versão leve para testes rápidos e tarefas simples.",
    family: "GPT-4.1",
    familySubtitle: "Mais capazes para tarefas complexas",
    input_cost: 0.1,
    output_cost: 0.4,
  },
  {
    id: "claude-3.5-sonnet",
    label: "Claude 3.5 Sonnet",
    provider: "Anthropic",
    description: "Disponível quando o backend estiver apontando para um provider compatível.",
    family: "Claude",
    familySubtitle: "Boa escrita e raciocínio consistente",
    input_cost: 3,
    output_cost: 15,
  },
  {
    id: "llama-3.1-70b",
    label: "Llama 3.1 70B",
    provider: "Meta",
    description: "Disponível quando o backend estiver apontando para um provider compatível.",
    family: "Llama",
    familySubtitle: "Alternativas abertas para testes",
    input_cost: 0.7,
    output_cost: 0.8,
  },
];

function titleCaseSlug(value: string) {
  return value
    .split(/[-_/]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function inferFamily(modelId: string, provider?: string) {
  const id = modelId.toLowerCase();
  if (id.includes("gpt-4o")) return "GPT-4o";
  if (id.includes("gpt-4.1")) return "GPT-4.1";
  if (id.includes("claude")) return "Claude";
  if (id.includes("llama")) return "Llama";
  return provider ?? "Outros";
}

function inferFamilySubtitle(family: string) {
  if (family === "GPT-4o") return "Mais econômicos e com baixa latência";
  if (family === "GPT-4.1") return "Mais capazes para tarefas complexas";
  if (family === "Claude") return "Boa escrita e raciocínio consistente";
  if (family === "Llama") return "Alternativas abertas para testes";
  return "Modelos disponíveis nesta categoria";
}

export function resolveEnabledModelCatalog(allowedIds: string[]): ModelCatalogItem[] {
  const normalized = allowedIds.map((item) => item.trim()).filter(Boolean);
  const fallback = normalized.length > 0 ? normalized : [MODEL_CATALOG[0]?.id ?? "gpt-4o-mini"];

  return fallback.map((modelId) => {
    const known = MODEL_CATALOG.find((item) => item.id === modelId);
    if (known) return known;
    const family = inferFamily(modelId, "Custom");
    return {
      id: modelId,
      label: titleCaseSlug(modelId),
      provider: "Custom",
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
