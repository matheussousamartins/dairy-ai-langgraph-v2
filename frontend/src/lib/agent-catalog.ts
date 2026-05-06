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
    label: "Orquestrador",
    endpoint: "/webhook/orquestrador",
    input_cost: 0,
    output_cost: 0,
  },
];

export function getAgentById(id: string) {
  return AGENT_CATALOG.find((item) => item.id === id);
}

