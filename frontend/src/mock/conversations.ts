import type { GenesisMessage, GenesisSession } from "@/state/useGenesisUI";

export const mockSessions: GenesisSession[] = [
  {
    id: "thread-briefing",
    title: "Briefing Inicial",
    createdAt: Date.now() - 1000 * 60 * 60 * 18,
  },
  {
    id: "thread-scout",
    title: "Pesquisa Tavily",
    createdAt: Date.now() - 1000 * 60 * 60 * 6,
  },
];

export const mockMessages: Record<string, GenesisMessage[]> = {
  "thread-briefing": [
    {
      id: "m1",
      role: "user",
      content: "Preciso do status do pipeline de vendas.",
      timestamp: Date.now() - 1000 * 60 * 60 * 18,
      modelId: "openrouter/gpt-4.1-mini",
    },
    {
      id: "m2",
      role: "assistant",
      content: "Status geral: 12 oportunidades em andamento. Quer detalhar por região ou etapa?",
      timestamp: Date.now() - 1000 * 60 * 60 * 18 + 30 * 1000,
      modelId: "openrouter/gpt-4.1-mini",
    },
  ],
  "thread-scout": [
    {
      id: "m3",
      role: "user",
      content: "Liste insights recentes sobre IA generativa aplicada em varejo.",
      timestamp: Date.now() - 1000 * 60 * 60 * 6,
      modelId: "openrouter/meta-llama/llama-3.1-70b-instruct",
    },
    {
      id: "m4",
      role: "assistant",
      content: "Pesquisei tavily e encontrei três relatórios relevantes. Quer um resumo comparativo?",
      timestamp: Date.now() - 1000 * 60 * 60 * 6 + 45 * 1000,
      modelId: "openrouter/meta-llama/llama-3.1-70b-instruct",
      usedTavily: true,
    },
  ],
};
