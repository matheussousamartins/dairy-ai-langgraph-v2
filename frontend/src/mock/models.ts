import type { ModelOption } from "@/state/useGenesisUI";

export const modelCatalog: ModelOption[] = [
  {
    id: "openrouter/gpt-4.1-mini",
    label: "GPT-4.1 Mini",
    inputCost: 0.3,
    outputCost: 0.6,
  },
  {
    id: "openrouter/anthropic/claude-3.5-sonnet",
    label: "Claude 3.5 Sonnet",
    inputCost: 3,
    outputCost: 15,
  },
  {
    id: "openrouter/meta-llama/llama-3.1-70b-instruct",
    label: "Llama 3.1 70B",
    inputCost: 0.7,
    outputCost: 0.8,
  },
];
