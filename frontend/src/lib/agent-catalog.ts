export interface AgentCatalogItem {
  id: string;
  label: string;
  endpoint: string;
  input_cost?: number;
  output_cost?: number;
}

export const AGENT_CATALOG: AgentCatalogItem[] = [
  {
    id: "orquestrador",
    label: "Assistente Geral (Orquestrador)",
    endpoint: "/webhook/orquestrador",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-1",
    label: "Tecnologia de Queijos",
    endpoint: "/webhook/agente-1",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-2",
    label: "Fermentados",
    endpoint: "/webhook/agente-2",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-3",
    label: "Regulatórios por País",
    endpoint: "/webhook/agente-3",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-4",
    label: "Qualidade do Leite",
    endpoint: "/webhook/agente-4",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-5",
    label: "Diagnóstico de Defeitos",
    endpoint: "/webhook/agente-5",
    input_cost: 0,
    output_cost: 0,
  },
  {
    id: "agente-6",
    label: "Formulação e Desenvolvimento",
    endpoint: "/webhook/agente-6",
    input_cost: 0,
    output_cost: 0,
  },
];

export function getAgentById(id: string) {
  return AGENT_CATALOG.find((item) => item.id === id);
}

